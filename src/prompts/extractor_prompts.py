# src/prompts/extractor_prompts.py

"""
Prompt templates and output schemas for the Extractor Node.

The Extractor processes one section at a time. For each section it receives:
  - The full section content from the 3GPP document
  - The extraction_focus defined by the Planner for that section
  - The OpenAPI spec overview as reference
  - Optional helper context from auxiliary 3GPP documents

It returns a structured list of RawRule objects.

Classes:
    SectionRules    — wrapper around list[RawRule] for structured LLM output

Constants:
    extractor_prompt — ChatPromptTemplate used to invoke the LLM per section
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from schemas.rules import RawRule


# ---------------------------------------------------------------------------
# Structured output schema
# LLM is asked to return JSON matching SectionRules via
# llm.with_structured_output(SectionRules).
# A wrapper is needed because with_structured_output requires a BaseModel,
# not a plain list.
# ---------------------------------------------------------------------------

class SectionRules(BaseModel):
    """Wrapper holding all rules extracted from a single 3GPP section."""

    rules: list[RawRule] = Field(
        description=(
            "List of OpenAPI rules extracted from this section. "
            "Return an empty list if no concrete rules are found."
        )
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your task is to extract concrete OpenAPI rules from a section of a 3GPP document.

A rule is a specific, actionable statement that describes how a 3GPP NRM element,
attribute, operation, or constraint maps to an OpenAPI construct (path, schema,
operation, parameter, response, etc.).

Guidelines:
  - Extract only rules that are explicitly stated or clearly implied by the text.
  - Do NOT infer or hallucinate rules that are not grounded in the section content.
  - Each rule must have a clear mapping to a specific OpenAPI object and field.
  - If the section contains a table, extract one rule per relevant row.
  - If the section defines multiple attributes, extract one rule per attribute.
  - Ignore introductory or background text that does not contain mappable rules.

OpenAPI Specification reference (for mapping guidance):
{openapi_reference_overview}
"""

_USER = """\
Section ID   : {section_id}
Section Title: {section_title}

Extraction focus (defined by the Planner):
{extraction_focus}

Section content:
{section_content}

Auxiliary context from supporting 3GPP documents (if relevant):
{helper_context}

Extract all OpenAPI rules present in this section.
Return an empty list if no concrete rules are found.
"""

extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])
