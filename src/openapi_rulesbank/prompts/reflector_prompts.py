# src/prompts/reflector_prompts.py

"""
Prompt templates and output schemas for the Reflector Node.

The Reflector applies Chain-of-Thought self-reflection to each RawRule produced
by the Extractor. For each rule it receives:
  - The rule itself (section_id, rule_text, openapi_mapping)
  - Relevant OpenAPI reference chunks retrieved via RAG

It returns a ReflectionResult with confidence score, reasoning trace, and flag.

Classes:
    ReflectionResult — structured LLM output for a single rule reflection

Constants:
    reflector_prompt — ChatPromptTemplate used to invoke the LLM per rule
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured output schema
# The LLM is asked to return JSON matching ReflectionResult via
# llm.with_structured_output(ReflectionResult).
# ---------------------------------------------------------------------------

class ReflectionResult(BaseModel):
    """Self-reflection assessment produced by the Reflector Node for one rule."""

    reflection_confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Confidence score between 0.0 and 1.0. "
            "1.0 = rule is fully grounded in the section content and aligns with the "
            "OpenAPI specification. 0.0 = rule is uncertain, hallucinated, or incorrect."
        )
    )
    reflection_reasoning: str = Field(
        description=(
            "Chain-of-Thought explanation of why this rule was extracted, "
            "whether its openapi_mapping is correct, and how the retrieved "
            "OpenAPI reference context supports or contradicts it."
        )
    )
    reflection_flagged: bool = Field(
        description=(
            "True if this rule should receive priority attention during validation. "
            "Flag when: confidence < 0.7, the openapi_mapping is ambiguous, "
            "the rule is not explicitly stated in the section, or the RAG context "
            "suggests a different mapping."
        )
    )
    split_suggestion: str = Field(
        default="",
        description=(
            "If this rule should be split into multiple more specific rules, "
            "describe exactly how. Empty string if no split needed. "
            "Example: 'Split into two rules: one for 400 and one for 500'. "
            "For a path_operation rule: a split is only valid if the rule combines "
            "multiple HTTP methods in one field (e.g. 'put, patch'). In that case "
            "each split target must also be a path_operation, one per method. "
            "Do NOT suggest splitting a path_operation into request_body, response, "
            "or other rule_types — those are separate complementary rules, not a "
            "decomposition of path_operation."
        )
    )
    discard_suggestion: bool = Field(
        default=False,
        description=(
            "True if this rule should not exist — e.g. it describes the absence "
            "of a construct, which cannot be mapped to OpenAPI."
        )
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your task is to review an OpenAPI rule that was automatically extracted from a 3GPP
document and assess its quality using Chain-of-Thought reasoning.

For each rule, reason through the following questions:
  1. Is the rule explicitly stated or clearly implied by the section content?
  2. Is the openapi_mapping (object, field, value) correct and complete?
  3. Does the OpenAPI reference context below confirm or contradict this rule?
  4. How confident are you that this rule is accurate and actionable?
  5. Should this rule be split into multiple more specific rules?
     For example: a rule about '4xx/5xx responses' should be split into individual
     rules per status code if the section content lists them separately.
  6. Should this rule be discarded entirely?
     Discard if: the rule describes the absence of a construct (e.g. 'no query
     parameters'), or if the rule_text contains no mappable OpenAPI information.

Then produce:
  - A confidence score (0.0 to 1.0)
  - A reasoning trace explaining your assessment step by step
  - A flag indicating whether the rule needs priority validation
  - A split suggestion if the rule should be broken into multiple rules
  - A discard suggestion if the rule should not exist at all

Relevant OpenAPI Specification context (retrieved for this rule):
{openapi_reference_context}
"""

_USER = """\
Rule to assess:

  Section ID     : {section_id}
  Section title  : {section_title}
  Rule type      : {rule_type}
  Rule text      : {rule_text}
  OpenAPI object : {openapi_object}
  OpenAPI field  : {openapi_field}
  OpenAPI value  : {openapi_value}

Section content (the source from which this rule was extracted):
{section_content}

Apply Chain-of-Thought reasoning and return your assessment.
"""

reflector_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])


# ---------------------------------------------------------------------------
# Section-level completeness assessment (Fase 2)
# Invoked once per section after all individual rules have been reflected.
# ---------------------------------------------------------------------------

class SectionReflection(BaseModel):
    """
    Completeness assessment for a full section — produced once per section
    after all individual rules have been reflected (Fase 2).
    """

    missing_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Rules genuinely missing from this section. "
            "Each entry is a direct instruction for the Extractor. "
            "Example: 'Extract rule for HTTP 201 Created response for PUT operation'. "
            "Do NOT include rules already present in current_rules or already_validated_rules."
        )
    )
    reasoning: str = Field(
        description=(
            "Why these rules are missing and what in the section content "
            "supports their existence."
        )
    )


_SECTION_SYSTEM = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your task is to assess whether all OpenAPI rules have been extracted from a 3GPP section.

You will receive:
  - The full section content
  - Rules just extracted and reflected in this iteration (current_rules)
  - Rules already validated in previous iterations (already_validated_rules)
  - Rules identified as missing in the previous iteration (previous_missing_rules)

SYSTEMATIC COMPLETENESS CHECK:
For each rule_type below, check whether the section content implies it should exist.
If yes, verify that at least one rule of that type is present in current_rules OR
already_validated_rules. If not present, report it as missing.

  path_operation  → Did the section define an HTTP method (GET, PUT, POST, DELETE, PATCH)
                    on a resource path? If yes → must have one path_operation rule per method.

  request_body    → Did the section describe a request body, input data structure, or
                    mandatory payload? If yes → must have at least one request_body rule.

  response        → Did the section describe response codes or response data structures?
                    If yes → must have one rule per response code or wildcard range (4XX, 5XX).
                    Check each code mentioned: 200, 201, 204, 4xx/5xx, etc.

  path_parameter  → Did the section describe path variables (e.g. {{id}}, {{version}}, {{className}})?
                    If yes → must have one path_parameter rule per variable.

  query_parameter → Did the section describe named query parameters (not their absence)?
                    If yes → must have one query_parameter rule per named parameter.

  schema_property → Did the section describe data model attributes or fields?
                    If yes → must have one schema_property rule per attribute.

  security_scheme → Did the section describe authentication or security requirements?
                    If yes → must have at least one security_scheme rule.

Go through this checklist for every rule_type. Do not skip any.

A rule is MISSING if ALL of the following are true:
  1. The section content clearly implies it should exist as an OpenAPI construct
  2. It is NOT present in current_rules
  3. It is NOT present in already_validated_rules

Do NOT suggest:
  - Rules already covered by current_rules or already_validated_rules
  - Rules about the absence of a construct (e.g. 'no query parameters')
  - Vague or non-actionable rules
  - Rules that duplicate what is already extracted under a different phrasing

For previous_missing_rules: check whether each one was resolved by current_rules.
Only carry forward what is genuinely still absent.

Be specific and actionable. Each missing rule must be a direct instruction
to the Extractor describing exactly what to extract and how to map it.
"""

_SECTION_USER = """\
Section ID: {section_id}

Section content:
{section_content}

Rules extracted and reflected in this iteration:
{current_rules}

Rules already validated in previous iterations (do NOT suggest these as missing):
{already_validated_rules}

Rules identified as missing in the previous iteration (check if now resolved):
{previous_missing_rules}

What is still genuinely missing from this section?
"""

reflector_section_prompt = ChatPromptTemplate.from_messages([
    ("system", _SECTION_SYSTEM),
    ("human",  _SECTION_USER),
])
