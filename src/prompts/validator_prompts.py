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
    reason: str = Field(
        description=(
            "Brief explanation of the verdict. "
            "If valid=True: confirm what makes the rule correct (or leave empty). "
            "If valid=False: explain exactly what is wrong and what the correct "
            "mapping should be."
        )
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your task is to perform semantic validation on an OpenAPI rule that was automatically
extracted from a 3GPP document and reviewed by a self-reflection step.

Verify:
  1. Is the rule_text grounded in the section content (not hallucinated)?
  2. Is the openapi_object a valid OpenAPI construct (path, schema, operation, parameter, etc.)?
  3. Is the openapi_field a valid field within that OpenAPI object?
  4. Is the openapi_value correct and consistent with the OpenAPI specification?

Return valid=False if any of the above checks fail.
"""

_USER = """\
Rule to validate:

  Section ID     : {section_id}
  Section title  : {section_title}
  Rule text      : {rule_text}
  OpenAPI object : {openapi_object}
  OpenAPI field  : {openapi_field}
  OpenAPI value  : {openapi_value}

Reflector assessment:
  Confidence     : {confidence}
  Flagged        : {flagged}
  Reasoning      : {reasoning}

Section content (source of the rule):
{section_content}

Return your semantic validation verdict.
"""

validator_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])
