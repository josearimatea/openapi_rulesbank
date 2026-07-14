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
            "Ordered list of ALL sections selected for rule extraction, "
            "including 'low' priority ones. Exclude only sections that are "
            "clearly irrelevant (e.g. cover page, table of contents, list of "
            "abbreviations, normative references). "
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

The pipeline extracts exactly these seven types of OpenAPI rules from each section:

  path_operation  — an HTTP method (GET, POST, PUT, DELETE, PATCH, etc.) on a resource path
  path_parameter  — a path variable in a URI template (e.g. the "id" in /resources/{{id}},
                  the "className" in /{{className}}={{id}}, the "version" in /api/{{version}})
  query_parameter — a named query string parameter for an HTTP operation
  request_body    — the media type and schema of an HTTP request body
  response        — an HTTP response code (including the "default" error response) and
                  its body schema for an operation
  schema_property — a field, attribute, or schema-level constraint in a data model
                  (becomes components/schemas in OpenAPI), including enumerations,
                  required fields, and composition keywords (oneOf/anyOf/allOf)
  security_scheme — an authentication or security requirement

For each section, ask: "Could this section contain at least one of these seven rule types?"
If yes, include it. If no, exclude it.

INCLUDE — assign priority based on directness:
  HIGH   — sections that directly define HTTP operations, paths, parameters,
            request/response structures, or operation-to-HTTP mappings
  MEDIUM — sections that define data types, structured types, enumerations,
            or attribute definitions (these become schema_property rules)
  LOW    — sections that provide mapping context or overview

EXCLUDE — sections with no contribution to any of the seven rule types:
  - Administrative: foreword, scope, references, abbreviations, void sections
  - State machines: triggering events, pre/post conditions, from/to state tables
  - Stage 2 definitions: 3GPP documents are structured in stages.
    Stage 2 defines operations in technology-agnostic terms — abstract input/output
    parameters, results, exceptions — with no HTTP binding, no URI, no status code,
    no media type, and no JSON type mapping. These contain none of the seven rule types.
    Stage 3 maps those definitions to a concrete protocol (HTTP/REST) and is the
    source of all seven rule types. Include Stage 3; exclude Stage 2.
    When uncertain: if the section mentions no concrete protocol artifact
    (no HTTP method, no URI pattern, no status code, no media type, no JSON schema),
    it is likely Stage 2 — exclude it.

The OpenAPI Specification overview below gives you context on the target format:
{openapi_reference_overview}
"""

_USER = """\
{contract_summary}

The SERVICE CONTRACT above (when present) defines the scope of the specific
OpenAPI YAML these rules are being built for. Treat it as a strong signal about
what belongs to this service — YOU still decide which sections to include, but
let the contract drive that decision:
  - A section is in scope when its content matches the service's purpose,
    capabilities, key_concepts, or inclusion_keywords → keep it and set priority
    by directness (HIGH/MEDIUM/LOW as defined above).
  - A section is likely out of scope when its content is dominated by
    exclusion_keywords or out_of_scope_signals (i.e. it belongs to another MnS)
    → exclude it unless it clearly also defines rules for THIS service.
  - extraction_focus should reference the relevant capabilities by name when applicable.
  - SCOPE IS AN INTERSECTION — a section qualifies ONLY if BOTH hold:
      (a) it belongs to THIS service (matches purpose/capabilities/inclusion, and
          is NOT dominated by exclusion_keywords/out_of_scope_signals), AND
      (b) it is in the stage 3 block named by the contract's "Stage scope"
          (e.g. clause 12.1.x for this service — NOT the stage 3 block of another
          service such as 12.2.x, 12.3.x, which belong to other MnS).
    Exclude a section if EITHER condition fails. In particular, the stage 3 block
    of a DIFFERENT service (another MnS) is out of scope — do not select it.
    Also exclude every stage 2 section (abstract operation/parameter definitions),
    even this service's, since stage 2 is not a source of OpenAPI rules.
When the contract is absent, fall back to the inclusion/exclusion rules above.

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
