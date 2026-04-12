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
            "Example: 'Split into two rules: one for 400 and one for 500'"
        )
    )
    discard_suggestion: bool = Field(
        default=False,
        description=(
            "True if this rule should not exist — e.g. it describes the absence "
            "of a construct, which cannot be mapped to OpenAPI."
        )
    )
    missing_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Rules that appear to be missing from this section. "
            "Each entry is a short description of what should be extracted. "
            "Only populate if you can clearly identify missing rules from the section content."
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
  7. Are there rules missing from this section that were not extracted?
     Look at the full section content and identify OpenAPI constructs that should
     have generated rules but didn't. Only report concrete missing rules, not vague ones.

Then produce:
  - A confidence score (0.0 to 1.0)
  - A reasoning trace explaining your assessment step by step
  - A flag indicating whether the rule needs priority validation
  - A split suggestion if the rule should be broken into multiple rules
  - A discard suggestion if the rule should not exist at all
  - A list of missing rules visible in the section content

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
