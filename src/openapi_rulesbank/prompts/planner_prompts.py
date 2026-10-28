# src/prompts/planner_prompts.py

"""
Prompt templates and output schemas for the Planner Node.

The Planner is the first LLM call in the pipeline. It receives a list of
section titles and previews from the 3GPP document and decides which sections
are worth extracting OpenAPI rules from, and what to focus on in each one.

Classes:
    SectionDecision — the LLM's decision on one section (include or not, and why)
    SectionPlan     — plan for a single selected section
    ExtractionPlan  — the extraction plan handed to the rest of the pipeline

Functions:
    planner_decisions_schema — the LLM's answer schema: one required
                               SectionDecision per section offered

Constants:
    planner_prompt  — ChatPromptTemplate used to invoke the LLM
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

from openapi_rulesbank.prompts.rule_type_definitions import rule_type_definitions_for_template


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


class SectionDecision(BaseModel):
    """The Planner's decision on one section it was given."""

    include: bool = Field(description="True to extract rules from this section.")
    priority: str = Field(
        description="When included: 'high', 'medium' or 'low' (see INCLUDE). "
                    "When excluded: empty."
    )
    extraction_focus: str = Field(
        description="When included: what kind of rules to look for in this section. "
                    "When excluded: empty."
    )
    notes: str = Field(description="Any observation that may help the Extractor, or empty.")
    reason: str = Field(
        description="When excluded: one sentence naming the EXCLUDE rule that applies, "
                    "read from the row. When included: empty."
    )


def planner_decisions_schema(sections: list[dict]) -> type[BaseModel]:
    """The Planner's answer schema for these sections: one REQUIRED decision field
    per section offered, named s<section_id>, in the order of the rows.

    With two free lists (selected, excluded) the LLM wrote each from memory and,
    over a few dozen rows, now and then left one out of both — a section dropped
    with no decision (28/out/2026). A required field per row makes that answer
    invalid: under a strict schema it cannot be produced at all, otherwise it fails
    validation instead of passing as a silent exclusion.
    """
    fields: dict = {
        "document_summary": (str, Field(
            description="One or two sentences describing what this 3GPP document "
                        "covers and its relevance to OpenAPI rule extraction.")),
    }
    for s in sections:
        fields[f"s{s['section_id']}"] = (SectionDecision, Field(
            description=f"Decision for row [{s['section_id']}] {s['title'][:80]}"))
    return create_model("PlannerDecisions", **fields)


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
            "List of ALL sections selected for rule extraction, including 'low' "
            "priority ones. Exclude only sections that are clearly irrelevant "
            "(e.g. cover page, table of contents, list of abbreviations, normative "
            "references). Keep them in DOCUMENT ORDER (ascending clause number), "
            "NOT reordered by priority: a section often relies on one that precedes "
            "it (e.g. a 'General' clause declares the imported types a later 'Type' "
            "clause uses), and downstream processing follows this order."
        )
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your task is to analyze a list of sections from a 3GPP document and produce
an extraction plan that will guide the next stage of a rule extraction pipeline.

The pipeline extracts exactly eight types of OpenAPI rules: path_operation,
path_parameter, query_parameter, request_body, response, schema_property,
security_scheme, callback. What each one IS is defined below — the same
definitions every later stage uses:

""" + rule_type_definitions_for_template() + """

For each section, ask: "Could this section contain at least one rule of these
eight types, as defined above?" If yes, include it. If no, exclude it and say
why. A section that only describes something, without giving what a definition
above requires, contains none.

Your answer has one decision field per row, named s<section_id> (row [251] ->
field "s251"), in the order of the rows: fill every one of them.

Judge every section by its CONTENT, NOT by its title. Each row gives "path:" — the
titles above the section, so a clause titled only "POST" or "URI" shows the
resource it belongs to — the start of the content (the preview) and, when
present, two more things read by code over the WHOLE section: "tables:" — every
table it holds, with its caption and its column names, marked "(empty: n/a)" when
the table lists nothing; and "signals:" — what the whole text shows of each
rule type (methods, paths, status codes, media types, data types...), found with
the vocabulary of the definitions above, or "none". A section's decisive table is
often past the preview: read the tables and signals before judging it. Signals
are hints, not decisions — judge each section by the definitions.
3GPP clauses are hierarchical: a parent clause (e.g. "Resources", "Data type
definitions", "RESTful HTTP-based solution set") often only GROUPS its children,
so its preview carries no prose of its own — it is empty or shows just markdown
heading markers ("#", "##", "###") or a bare title. Such a section defines
nothing to extract, however relevant its title sounds — do NOT select it. The
rules live in its child subsections, which appear as their own rows (deeper
clause numbers) with real text in their previews. Conversely, a deep subsection
IS a rule source when its preview shows an HTTP method, a URI, a parameter or
data-type table, an enumeration, or an operation/notification mapping — select it
on that merit. Being a parent or a leaf decides nothing; only whether the
section's own preview shows something extractable does.

INCLUDE — assign priority based on directness:
  HIGH   — sections that directly define HTTP operations, paths, parameters,
            request/response structures, or operation-to-HTTP mappings
  MEDIUM — sections that define data types, structured types, enumerations,
            or attribute definitions (these become schema_property rules)
  LOW    — sections whose own table gives rules or facts later sections build on:
            an overview table of resources and methods, or a table listing the
            data types used and the specification each one is defined in

EXCLUDE — sections with no contribution to any of the eight rule types:
  - Prose-only clauses: decided by the row's "tables:" field, not by the title. A
    clause whose row lists NO table — or only tables marked "(empty: n/a)" — and
    whose preview shows no HTTP method, path, parameter, status code, media type
    or data type is prose only: exclude it, and do NOT select it as "context" or
    "support" for a sibling (context reaches the later stages through retrieval;
    selecting such a clause only spends extraction attempts that produce no
    rule). A clause whose row lists a table with content is NEVER excluded on
    this ground — not even when it summarises what a sibling details (an
    overview table) or its title is just "URI" or "Description".
  - Administrative: foreword, scope, references, abbreviations, void sections
  - State machines: triggering events, pre/post conditions, from/to state tables
  - Stage 2 definitions: 3GPP documents are structured in stages.
    Stage 2 defines operations in technology-agnostic terms — abstract input/output
    parameters, results, exceptions — with no HTTP binding, no URI, no status code,
    no media type, and no JSON type mapping. These contain none of the eight rule types.
    Stage 3 maps those definitions to a concrete protocol (HTTP/REST) and is the
    source of all eight rule types. Include Stage 3; exclude Stage 2.
    When uncertain: if the section mentions no concrete protocol artifact
    (no HTTP method, no URI pattern, no status code, no media type, no JSON schema),
    it is likely Stage 2 — exclude it.

The OpenAPI Specification overview below gives you context on the target format:
{openapi_reference_overview}
"""

_USER = """\
{contract_summary}

The SERVICE CONTRACT above (when present) defines the scope of the specific
OpenAPI YAML these rules are being built for. The sections below have ALREADY
been narrowed to this service's stage 3 block, so scope is mostly settled — use
the contract to make the finer content calls:
  - A section is in scope when its content matches the service's purpose,
    capabilities, key_concepts, or inclusion_keywords → keep it and set priority
    by directness (HIGH/MEDIUM/LOW as defined above).
  - A section is out of scope when its content is dominated by exclusion_keywords
    or out_of_scope_signals (it belongs to another MnS) → exclude it unless it
    clearly also defines rules for THIS service.
  - extraction_focus should reference the relevant capabilities by name when applicable.

Document sections to analyze (section_id | title | path: … | content preview |
tables: … | signals: …):
{sections_summary}

Auxiliary context (from supporting 3GPP documents, if any):
{helper_context}

Decide every section above, one decision field per row. The code keeps the
included ones in document order; a priority never changes that order.
"""



planner_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])
