# src/prompts/section_analyzer_prompts.py

"""
Prompts and schemas for the two-phase Section Analyzer the Extractor runs BEFORE
its main extraction.

The main extraction is one crowded call that reliably misses subtle judgements —
above all, that a defined type embeds a common type imported from ANOTHER spec
(e.g. NotificationHeader from TS 28.623), whose fields must become one `allOf
$ref`, not the type's own properties. The reference to that other spec and the
type's own field list live in DIFFERENT sections, so no single section carries
the whole picture.

Two phases, plus a running reference memory, solve this:

  Phase 1 (SectionImports) — reads the section and spots any reference to another
    spec. A reference is a trigger to INVESTIGATE: Phase 1 writes, per reference,
    a retrieval query to fetch what that type/spec is, and a short explanation of
    why the other spec is mentioned here. The node runs the queries (RAG).

  Phase 2 (SectionAnalysis) — sees the section, the reference memory, and the RAG
    results. It does two things: (A) MEMORY — say what to add about each reference
    (its fields / a note), so later sections know what it is without re-deriving;
    (B) EXTRACTOR — classify the section and draft the guidance the main prompt
    follows (schema home or not, imported base → allOf $ref + inherited fields).

Phase 1 finds and fetches; Phase 2 writes the memory and the extractor guidance.
The memory is generic ({type: {spec, fields, note}}), so any imported type from
any spec flows through the same path.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase 1 — find references to other specs and how to investigate them
# ---------------------------------------------------------------------------

class SpecReference(BaseModel):
    """One reference this section makes to another spec, with how to look it up."""

    type_name: str = Field(
        description="The referenced type's name, e.g. 'NotificationHeader'."
    )
    source_spec: str = Field(
        description="The spec it comes from, as written here, e.g. 'TS 28.623'."
    )
    key_terms: str = Field(
        default="",
        description=(
            "A few distinctive words about what this type IS (its role/domain, e.g. "
            "'notification header common fields' or 'error response body'), to help "
            "single its definition out from similarly named types. Just the terms — "
            "the retrieval query is assembled in code around the type name and the "
            "shape of an OpenAPI schema definition, so do NOT write a full sentence "
            "or guess field names here."
        )
    )


class SectionImports(BaseModel):
    """Phase 1 output: the other-spec references this section carries."""

    references: list[SpecReference] = Field(
        default_factory=list,
        description=(
            "Every reference this section makes to a type defined in ANOTHER spec. "
            "Sources: a 'Data types imported' table (Reference column names an "
            "external spec), or a data-type column citing a type the reference "
            "memory already attributes to another spec. One entry per referenced "
            "type. Empty when the section references no external types."
        )
    )
    explanation: str = Field(
        default="",
        description=(
            "A short plain-language note on WHY the other spec(s) are referenced "
            "here and what role those types play in this section — so the next phase "
            "understands the purpose of each lookup (e.g. 'This type reuses the "
            "common NotificationHeader from TS 28.623 as the header of the "
            "notification payload defined here')."
        )
    )


_STEP1_SYSTEM = """\
You triage ONE 3GPP specification section before OpenAPI rule extraction. Your job
here is narrow: find every reference this section makes to a type defined in
ANOTHER specification, and set up its investigation.

A reference to another spec is a TRIGGER to look it up — you do not resolve it
now, you say how to fetch it and why it is here:

  - Find references. They appear as a "Data types imported" table whose Reference
    column names an external spec (e.g. "NotificationHeader — TS 28.623",
    "ErrorResponse — TS 28.623"), or as a data type cited in an attribute table
    that the reference memory below already attributes to another spec.
  - For each referenced type, give its name, its source spec, and a few key_terms
    describing what it IS (its role/domain). You do NOT write the retrieval query —
    the code assembles it from the type name and the shape of a schema definition;
    your key_terms only help disambiguate similar names. Do not guess field names.
  - Write one explanation covering why these other specs are referenced in this
    section and what the referenced types are used for here.

USE THE REFERENCE MEMORY (below): types already seen in earlier sections, with the
spec each comes from. If this section uses a type the memory attributes to another
spec — even if this section does not restate the import — include it as a
reference so it gets looked up and understood in this section's context.

Judge from the section text plus the memory. Do not invent a type or a spec that
appears in neither. If the section references no external types, return an empty
list.
"""

_STEP1_USER = """\
Reference memory so far (types seen in earlier sections, 'type -> source spec'):
{ref_memory}

Section Title: {section_title}

Section content:
{section_content}

Return the external references this section makes, each with a fetch query, plus
one explanation of why they are referenced here.
"""

section_import_ref_prompt = ChatPromptTemplate.from_messages([
    ("system", _STEP1_SYSTEM),
    ("human",  _STEP1_USER),
])


# ---------------------------------------------------------------------------
# Phase 2 — update the reference memory and draft the extractor guidance
# ---------------------------------------------------------------------------

class ReferenceMemoryEntry(BaseModel):
    """What Phase 2 learned about one referenced type, to add to the memory."""

    type_name: str = Field(description="The referenced type's name.")
    source_spec: str = Field(default="", description="The spec it is defined in.")
    fields: list[str] = Field(
        default_factory=list,
        description=(
            "The type's own PROPERTY NAMES from the fetched definition — the members "
            "it declares (the keys under `properties:`, or the attribute-table rows), "
            "not the schema keywords that structure them (allOf, type, properties, "
            "required, items, $ref are structure, never property names). Empty if no "
            "definition was retrieved. These let a later section recognise, by field "
            "overlap, when it embeds this type as a base."
        )
    )
    note: str = Field(
        default="",
        description="One short line on what this type is / how it is used."
    )


class SectionAnalysis(BaseModel):
    """Phase 2 output: memory updates + guidance for the Extractor."""

    memory_updates: list[ReferenceMemoryEntry] = Field(
        default_factory=list,
        description=(
            "Entries to add/refresh in the running reference memory, built from the "
            "Phase 1 references, the RAG results and this section. Record each "
            "referenced type with its spec, its own fields (from the fetched "
            "definition), and a short note. Empty when the section referenced no "
            "external types."
        )
    )

    section_kind: str = Field(
        description=(
            "What this section fundamentally is, from its own content: "
            "'type_definition' (a 'Definition of type <Name>' attribute table), "
            "'enumeration' (defines an enum's values), 'mapping' (maps an IS "
            "operation/notification to its HTTP form — 'mapped to SS equivalents'), "
            "'resource' (a resource URI and its HTTP methods), or 'other'."
        )
    )
    emits_schema_properties: bool = Field(
        description=(
            "True ONLY when section_kind is 'type_definition' or 'enumeration' — the "
            "section is the HOME of a schema. False for 'mapping'/'resource'/'other': "
            "they yield HTTP-level rules but MUST NOT emit schema_property; those "
            "attributes belong to the type-definition section."
        )
    )
    schema_name: str = Field(
        default="",
        description=(
            "For a type_definition/enumeration, the canonical schema name: ONLY the "
            "<Name> part of the defining heading ('<clause> Type <Name>' / '<clause> "
            "Enumeration <Name>'), with its exact casing. Strip the leading clause "
            "number and the label word 'Type'/'Enumeration' — they are not part of "
            "the name (heading 'Enumeration PerfMetricDirection' → 'PerfMetric"
            "Direction'). Empty otherwise."
        )
    )
    base_ref: str = Field(
        default="",
        description=(
            "When this type embeds an imported common type as its base — its "
            "attribute table repeats that type's fields (known from the memory/RAG) "
            "— the OpenAPI $ref to it, e.g. "
            "\"$ref: 'TS28623_ComDefs.yaml#/components/schemas/NotificationHeader'\". "
            "Empty when there is no imported base."
        )
    )
    inherited_fields: list[str] = Field(
        default_factory=list,
        description=(
            "The attribute names this type INHERITS from base_ref (the ones in BOTH "
            "this type's table and the imported type's fields), which must NOT be "
            "emitted as own properties. Empty when base_ref is empty."
        )
    )
    guidance: str = Field(
        description=(
            "One or two imperative sentences telling the Extractor how to build this "
            "section's rules. It MUST stay consistent with emits_schema_properties: "
            "when that is False, the guidance must NOT tell the Extractor to emit a "
            "schema or its properties (say what it DOES emit — callback/request_body/"
            "response — and that schema_property is not this section's job). "
            "E.g. 'Mapping section: emit the callback/request_body/response; do NOT "
            "emit schema_property — the parameters belong to their Type section.' or "
            "'Type NotifyThresholdCrossing embeds NotificationHeader: emit one allOf "
            "$ref to it and only these own properties: <list>.'"
        )
    )


_STEP2_SYSTEM = """\
You are an expert in 3GPP specifications and OpenAPI. You finish triaging ONE
section before another agent extracts rules from it. You are given the section,
the running reference memory, Phase 1's explanation of the section's external
references, and the fetched definitions of those references (RAG). Two jobs:

A. UPDATE THE REFERENCE MEMORY. For each external type this section references,
   record what it is: its name, its source spec, its OWN property names, and a
   short note. Read the property names out of the fetched definition — they are the
   members the type declares, NOT the schema keywords that structure them (allOf,
   type, properties, required, items, $ref are structure, never property names).
   The fetched text may be split across several chunks and interleaved with other
   types; gather this type's own members from wherever they appear. This is generic
   — any imported type from any spec (imports and allOf composition are pervasive
   in these specs). Return empty when the section references no external types.

B. GUIDE THE EXTRACTOR:
   1. CLASSIFY the section by its content: type_definition, enumeration, mapping,
      resource, or other. emits_schema_properties is True ONLY for
      type_definition/enumeration. Keep `guidance` CONSISTENT with this flag —
      when emits_schema_properties is False, the guidance must not tell the
      Extractor to emit a schema or its properties (a mapping section maps
      parameters to a request body that references types defined elsewhere; it
      does not define those types here).
   2. RESOLVE AN EMBEDDED BASE. A type-definition section rarely names the common
      base it builds on — it just lists all its attributes, the inherited ones
      mixed in with its own. To find the base, compare the section's attribute
      names against the known types in the reference memory: go through each memory
      type that has a `fields` list, and if this section's attributes INCLUDE that
      type's fields (all or nearly all of them), the section embeds that type as
      its base. When one matches:
        - set base_ref to the OpenAPI $ref of that memory type
          (e.g. "$ref: '<source spec file>#/components/schemas/<TypeName>'");
        - set inherited_fields to the section attributes that are that type's
          fields;
        - the type's OWN properties are the remaining attributes.
      Judge the overlap sensibly (a field may be spelled slightly differently);
      you decide the match, it is not a literal string test. If no memory type's
      fields are covered by this section, leave base_ref empty and inherited_fields
      []. The fetched definitions (RAG) are secondary evidence when the memory
      lacks fields.

Base every judgement on the section content, the reference memory, and the fetched
definitions — never invent a field or a base type not evidenced in them.
"""

_STEP2_USER = """\
Section ID   : {section_id}
Section Title: {section_title}

Section content:
{section_content}

Reference memory so far ('type -> {{spec, fields}}'):
{ref_memory}

Phase 1 explanation of this section's external references:
{phase1_explanation}

Fetched definitions of the referenced types (RAG results):
{rag_results}

Sibling sections in this same service (id, title, and the kind already decided for
each — titles only, for context on the neighbourhood):
{sibling_summary}
  Use them to place this section relative to its neighbours: e.g. if a sibling is
  the "Type <Name>" definition that owns a schema, then a "Mapping"/"Resource"
  sibling only references that schema and is not its home. Judge THIS section by
  its own content; the siblings are context, not a substitute for it.

Produce the memory updates and the extractor guidance for this section.
"""

section_analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", _STEP2_SYSTEM),
    ("human",  _STEP2_USER),
])
