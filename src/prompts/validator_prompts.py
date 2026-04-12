# src/prompts/validator_prompts.py

"""
Prompt templates and output schemas for the Validator Node.

The Validator applies two-stage validation to each ReflectedRule:
  1. Structural — Pydantic schema check (no LLM call).
  2. Semantic   — LLM checks whether the rule is consistent with the OpenAPI
                  specification and the 3GPP section it was extracted from.

Classes:
    ValidationVerdict — structured LLM output for one rule's semantic check

Constants:
    validator_prompt  — ChatPromptTemplate used for semantic validation
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class ValidationVerdict(BaseModel):
    """Semantic validation result produced by the Validator Node for one rule."""

    valid: bool = Field(
        description=(
            "True if the rule is semantically correct: the rule_text is grounded "
            "in the section content, the openapi_mapping is accurate, and the "
            "openapi_value is consistent with the OpenAPI specification. "
            "False if the rule contains an incorrect mapping, hallucinated content, "
            "or an openapi_value that contradicts the spec."
        )
    )
    error_type: str = Field(
        default="correction",
        description=(
            "Only relevant when valid=False. "
            "'correction' — the rule exists but mapping/text is wrong, fix it. "
            "'split' — the rule should be broken into multiple specific rules. "
            "'discard' — the rule should not exist at all. "
            "Default to 'correction' when valid=True."
        )
    )
    instruction: str = Field(
        default="",
        description=(
            "When valid=False: actionable instruction for the Extractor. "
            "Written as a direct command. Empty when valid=True. "
            "Examples:\n"
            "  correction: 'Fix openapi_field to parameters[in=query,name=scope]'\n"
            "  split: 'Split into three rules: one for 400, one for 404, one for 500'\n"
            "  discard: 'Discard — absence of parameters is not a valid OpenAPI rule'"
        )
    )
    agreed_with_missing: bool = Field(
        default=False,
        description=(
            "True if the Validator agrees with the Reflector's missing_rules suggestions "
            "for this section. When True, the Validator confirms these rules should be extracted."
        )
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your task is to perform semantic validation on an OpenAPI rule that was automatically
extracted from a 3GPP document and reviewed by a self-reflection step.

Verify all four points:
  1. Is the rule_text grounded in the section content (not hallucinated)?
  2. Is the openapi_object a valid OpenAPI construct for the given rule_type?
  3. Is the openapi_field valid and specific for that construct?
  4. Is the openapi_value correct and concrete for the rule_type?

Validation rules by rule_type:
  path_operation  : openapi_field must be a single lowercase HTTP method (get/put/post/delete/patch).
                    openapi_value must be the uppercase HTTP method (GET/PUT/POST/DELETE/PATCH).
                    Reject if multiple methods are combined in one field (e.g. "put, patch").
                    IMPORTANT: if the section assigns multiple HTTP methods to the same operation,
                    the correct extraction behavior is one separate rule per method. Do NOT reject
                    a rule for covering only one method when the section lists multiple methods for
                    the same operation — each single-method rule is valid and complete on its own.
  schema_property : openapi_field must be "properties.<name>". openapi_value must be a JSON Schema type.
  path_parameter  : openapi_field must be "parameters[in=path,name=<name>]".
  query_parameter : openapi_field must be "parameters[in=query,name=<name>]".
  response        : openapi_field must be an HTTP status code string ("200", "201", "400", etc.)
                    or an OpenAPI wildcard ("1XX"–"5XX").
  request_body    : openapi_field must be "content". openapi_value must be a media type.
  security_scheme : openapi_field must be "type". openapi_value must be a valid OAuth2/http/apiKey value.

Return valid=False if any check fails.

USING THE REFLECTOR'S ASSESSMENT:

The Reflector may have flagged this rule for splitting, discarding, or identified missing rules.
Use its suggestions as additional signals, but make your own judgment:

  - If the Reflector suggested a split and you agree:
      set valid=False, error_type='split'
      instruction='Split into N rules: [describe each one specifically]'
  - If the Reflector suggested discard and you agree:
      set valid=False, error_type='discard'
      instruction='Discard — [reason why this is not a valid OpenAPI rule]'
  - If the Reflector listed missing_rules and you agree they should exist:
      set agreed_with_missing=True (the current rule may still be valid=True)
  - If you disagree with any Reflector suggestion: explain why in the instruction field
    and state what should be done instead. Never silently ignore a suggestion.
    Example: "Reflector suggested split, but the section defines 4xx/5xx as a single
    wildcard range — keep as 4XX per OpenAPI spec. No action needed."

WRITING INSTRUCTIONS:
  Always write instruction as a direct command to the Extractor.
  Be specific: instead of 'fix the status code', write 'change openapi_field from 4XX to 400'.
  For splits: list exactly what each new rule should contain.
"""

_USER = """\
Rule to validate:

  Section ID     : {section_id}
  Section title  : {section_title}
  Rule type      : {rule_type}
  Rule text      : {rule_text}
  OpenAPI object : {openapi_object}
  OpenAPI field  : {openapi_field}
  OpenAPI value  : {openapi_value}

Reflector assessment:
  Confidence        : {reflection_confidence}
  Flagged           : {reflection_flagged}
  Reasoning         : {reflection_reasoning}
  Split suggestion  : {split_suggestion}
  Discard suggestion: {discard_suggestion}
  Missing rules     : {missing_rules}

Section content (source of the rule):
{section_content}

Return your semantic validation verdict.
"""

validator_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])
