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

    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Confidence score between 0.0 and 1.0. "
            "1.0 = rule is fully grounded in the section content and aligns with the "
            "OpenAPI specification. 0.0 = rule is uncertain, hallucinated, or incorrect."
        )
    )
    reasoning: str = Field(
        description=(
            "Chain-of-Thought explanation of why this rule was extracted, "
            "whether its openapi_mapping is correct, and how the retrieved "
            "OpenAPI reference context supports or contradicts it."
        )
    )
    flagged: bool = Field(
        description=(
            "True if this rule should receive priority attention during validation. "
            "Flag when: confidence < 0.7, the openapi_mapping is ambiguous, "
            "the rule is not explicitly stated in the section, or the RAG context "
            "suggests a different mapping."
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

Then produce:
  - A confidence score (0.0 to 1.0)
  - A reasoning trace explaining your assessment step by step
  - A flag indicating whether the rule needs priority validation

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

Apply Chain-of-Thought reasoning and return your assessment.
"""

reflector_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])
