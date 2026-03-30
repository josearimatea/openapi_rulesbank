# src/prompts/planner_prompts.py

"""
Prompt templates and output schemas for the Planner Node.

The Planner is the first LLM call in the pipeline. It receives a list of
section titles and previews from the 3GPP document and decides which sections
are worth extracting OpenAPI rules from, and what to focus on in each one.

Classes:
    SectionPlan     — plan for a single section (Pydantic, used as structured output)
    ExtractionPlan  — full extraction plan output by the LLM (Pydantic)

Constants:
    planner_prompt  — ChatPromptTemplate used to invoke the LLM
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured output schemas
# The LLM is asked to return JSON matching these Pydantic models via
# llm.with_structured_output(ExtractionPlan).
# ---------------------------------------------------------------------------

class SectionPlan(BaseModel):
    """Extraction plan for a single 3GPP document section."""

    section_id: str = Field(
        description="The section_id value from the parsed_sections list."
    )
    title: str = Field(
        description="The section title as it appears in the document."
    )
    priority: str = Field(
        description=(
            "Extraction priority: 'high' if the section likely contains "
            "concrete OpenAPI rules (e.g. endpoint paths, HTTP methods, schemas), "
            "'medium' if it contains supporting context (e.g. data model descriptions), "
            "'low' if it contains background or reference information."
        )
    )
    extraction_focus: str = Field(
        description=(
            "Brief instruction for the Extractor Node: what kind of rules to look "
            "for in this section. E.g. 'Extract HTTP method and path definitions', "
            "'Extract attribute-to-schema mappings', 'Extract response code rules'."
        )
    )
    notes: str = Field(
        description="Any additional observation that may help the Extractor, or empty string."
    )


class ExtractionPlan(BaseModel):
    """Full extraction plan produced by the Planner Node."""

    document_summary: str = Field(
        description=(
            "One or two sentences describing what this 3GPP document covers "
            "and its relevance to OpenAPI rule extraction."
        )
    )
    sections_to_extract: list[SectionPlan] = Field(
        description=(
            "Ordered list of sections selected for rule extraction. "
            "Include only sections with priority 'high' or 'medium'. "
            "Ordered from highest to lowest priority."
        )
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your task is to analyze a list of sections from a 3GPP document and produce
an extraction plan that will guide the next stage of a rule extraction pipeline.

For each section you receive, decide:
  1. Whether it is relevant to OpenAPI rule extraction (skip irrelevant sections).
  2. What its priority is: high, medium, or low.
  3. What the Extractor should focus on when processing it.

Focus on sections that define:
  - REST API endpoints, HTTP methods, resource paths
  - Request/response schemas and data types
  - Attribute mappings between NRM and OpenAPI/JSON
  - YANG-to-OpenAPI or NRM-to-OpenAPI mappings
  - Operation definitions (GET, POST, PUT, DELETE, PATCH)
  - Error codes and response structures

The OpenAPI Specification overview below gives you context on the target format:
{openapi_spec_overview}
"""

_USER = """\
Document sections to analyze (section_id | title | content preview):
{sections_summary}

Auxiliary context (from supporting 3GPP documents, if any):
{helper_context}

Produce an ExtractionPlan with all relevant sections ordered by priority.
"""

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])
