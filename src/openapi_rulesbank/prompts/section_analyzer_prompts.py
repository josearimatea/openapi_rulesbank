# src/prompts/section_analyzer_prompts.py

"""
Prompt and output schema for the Section Analyzer — a small LLM call the
Extractor Node runs BEFORE the main extraction.

The main extraction is one big call with many rules to follow at once, and the
model reliably misses two subtle, recurring judgements:
  1. whether a section DEFINES a data type (its schema_property rules) or only
     MAPS an operation/notification (HTTP rules only, no schema_property);
  2. whether a defined type embeds an imported common type (e.g. NotificationHeader
     from TS 28.623), whose fields must become one `allOf $ref` — not repeated as
     the type's own properties.

Resolving these in a small, focused call first — with the imported type's
composition retrieved from the 3GPP corpus — turns them into a few objective
instructions the Extractor can then follow, instead of re-deriving them inside
the crowded main prompt.

Classes:
    SectionAnalysis — structured verdict the analyzer returns.

Constants:
    section_analyzer_prompt — ChatPromptTemplate for the analyzer call.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class ExternalSpecRef(BaseModel):
    """One type declared as imported from another spec: name and source spec."""

    type_name: str = Field(description="The imported type's name, e.g. 'NotificationHeader'.")
    source_spec: str = Field(description="The spec it comes from, e.g. 'TS 28.623'.")


class SectionImports(BaseModel):
    """Step 1 output: external-spec references this section carries."""

    declared_refs: list[ExternalSpecRef] = Field(
        default_factory=list,
        description=(
            "Every type this section DECLARES as imported from another spec — the "
            "rows of a 'Data types imported' table whose Reference column names an "
            "external spec (e.g. NotificationHeader→TS 28.623, ErrorResponse→TS "
            "28.623). These are recorded so later sections can resolve types they "
            "use. Empty when the section declares no imports."
        )
    )
    embedded_base_type: str = Field(
        default="",
        description=(
            "If THIS section defines a data type that embeds an imported common type "
            "as its base (its attribute table repeats that type's fields), the base "
            "type's name (e.g. 'NotificationHeader'). Empty when the section defines "
            "no type or the type embeds no imported base."
        )
    )
    base_type_query: str = Field(
        default="",
        description=(
            "Only when embedded_base_type is set: a retrieval query to FETCH that "
            "type's own definition from the corpus of other specs, so its fields can "
            "be told apart from the section's own. Write it to match the definition "
            "block itself — the type name plus the shape of an OpenAPI schema "
            "definition (its properties and required list) and its source spec — not "
            "one-line mentions of the type elsewhere. Empty when embedded_base_type "
            "is empty."
        )
    )


_STEP1_SYSTEM = """\
You are triaging one 3GPP specification section before OpenAPI rule extraction.
Three small jobs, from the section's own text:

1. DECLARED IMPORTS — list every type the section declares as imported from another
   specification. 3GPP specs put these in a "Data types imported" table whose
   Reference column names an external spec (e.g. "NotificationHeader — TS 28.623",
   "ErrorResponse — TS 28.623"). Return each as {{type_name, source_spec}}. These
   get remembered so later sections can resolve a type they use. If the section has
   no such table, return an empty list.

2. EMBEDDED BASE — if this section DEFINES a data type (a "Definition of type
   <Name>" table) that embeds an imported common type as its base — recognisable
   because its attribute table repeats that imported type's fields (a notification
   type typically embeds NotificationHeader) — return that base type's name.
   Otherwise leave it empty.

3. BASE TYPE QUERY — only if you set an embedded base: write a retrieval query that
   will fetch THAT type's own definition from the other spec. Aim it at the
   definition block (the type name with its `type: object` / `properties:` /
   `required:` shape) and include the source spec, so it matches where the type is
   DEFINED, not the many places it is merely listed or referenced. Keep it general:
   describe an OpenAPI schema definition of the named type — do not hard-code field
   names. Leave empty when there is no embedded base.

USE THE REFERENCE MEMORY: the known imported types listed below were collected
from earlier sections (each shown as 'type → source spec'). A section that defines a type often
does NOT restate that it imports a common base — it just lists all the fields.
When the attributes this section lists include the fields of a type in the memory
(e.g. it lists the header fields of a known 'NotificationHeader'), treat that
memory type as the embedded base even though the section's own text does not name
it. That is exactly what the memory is for.

Judge from the section text plus the reference memory. Do not invent a type or a
spec that appears in neither.
"""

_STEP1_USER = """\
Known imported types so far (reference memory — {{type: source spec}}):
{ref_memory}

Section Title: {section_title}

Section content:
{section_content}

Return the declared imports, the embedded base type, and (if any) its fetch query.
"""

section_import_ref_prompt = ChatPromptTemplate.from_messages([
    ("system", _STEP1_SYSTEM),
    ("human",  _STEP1_USER),
])


class SectionAnalysis(BaseModel):
    """Up-front verdict about one section, guiding the Extractor's main call."""

    section_kind: str = Field(
        description=(
            "What this section fundamentally is, from its own content: "
            "'type_definition' (defines a data type's attributes — a table titled "
            "'Definition of type <Name>'), 'enumeration' (defines an enum's values), "
            "'mapping' (maps an IS operation/notification to its HTTP method, URI and "
            "SS parameters — text says parameters are 'mapped to SS equivalents'), "
            "'resource' (a resource URI and its HTTP methods), or 'other'."
        )
    )
    emits_schema_properties: bool = Field(
        description=(
            "True only when section_kind is 'type_definition' or 'enumeration' — "
            "the section is the HOME of a schema, so it is the source of "
            "schema_property rules. False for 'mapping'/'resource'/'other': those "
            "yield HTTP-level rules (path_operation, callback, request_body, "
            "response) but MUST NOT emit schema_property — the attributes they list "
            "belong to the type-definition section, not here."
        )
    )
    schema_name: str = Field(
        default="",
        description=(
            "For a type_definition/enumeration, the canonical schema name exactly as "
            "written in the defining heading ('Type <Name>' / 'Enumeration <Name>'), "
            "with its exact casing. Empty otherwise."
        )
    )
    base_ref: str = Field(
        default="",
        description=(
            "When a type_definition embeds an imported common type as its base "
            "(e.g. it lists that type in the section's 'Data types imported' table "
            "and its own table repeats that type's fields), the OpenAPI $ref to it, "
            "e.g. \"$ref: 'TS28623_ComDefs.yaml#/components/schemas/NotificationHeader'\". "
            "Empty when the type has no imported base."
        )
    )
    inherited_fields: list[str] = Field(
        default_factory=list,
        description=(
            "The attribute names the type INHERITS from base_ref (taken from the "
            "imported type's composition provided in the corpus context), which must "
            "therefore NOT be emitted as this type's own properties. Empty when "
            "base_ref is empty."
        )
    )
    guidance: str = Field(
        description=(
            "One or two imperative sentences telling the Extractor how to build this "
            "section's rules, reflecting the fields above. E.g. 'This is a mapping "
            "section: emit the callback/request_body/response, no schema_property.' "
            "or 'Type NotifyThresholdCrossing embeds NotificationHeader: emit one "
            "allOf $ref to it and only these own properties: <list>.'"
        )
    )


_SYSTEM = """\
You are an expert in 3GPP specifications and OpenAPI. You perform a quick triage
of ONE specification section before another agent extracts OpenAPI rules from it.
You do not extract rules — you make two judgements the extractor tends to get
wrong, and hand it clear instructions.

1. CLASSIFY the section by its own content (not its title):
   - type_definition — a table "Definition of type <Name>" listing attributes and
     their data types. It is the HOME of that schema.
   - enumeration — defines the allowed values of an enum type.
   - mapping — maps an IS operation or notification to its SS (HTTP) form; its text
     says parameters are "mapped to SS equivalents" and its table is a "Mapping of
     ... to SS equivalents". It lists parameter names but is NOT a type definition.
   - resource — a resource URI and the HTTP methods on it.
   - other — anything else (intro, overview, void).
   Set emits_schema_properties True ONLY for type_definition/enumeration.

2. RESOLVE IMPORTED BASE TYPES (composition/inheritance):
   Some defined types embed a common type imported from another spec — listed in
   the section's (or service's) "Data types imported" table with a Reference to an
   external spec such as "TS 28.623". When THIS type's attribute table repeats the
   fields of such an imported type, those fields are INHERITED, not own.
   The imported type's composition is given below under "Imported type context".
   In that case:
     - set base_ref to the OpenAPI $ref of the imported type;
     - set inherited_fields to exactly the attribute names that come from it
       (the ones appearing in BOTH this type's table and the imported type);
     - the type's OWN properties are the remaining attributes.
   If the type embeds no imported base, leave base_ref empty and inherited_fields [].

Base every judgement on the section content and the imported-type context provided
— never invent field names or a base type that is not evidenced there.
"""

_USER = """\
Section ID   : {section_id}
Section Title: {section_title}

Section content:
{section_content}

Imported type context (definitions of common types this section may import,
retrieved from the 3GPP corpus — use ONLY to decide which fields are inherited):
{imported_type_context}

Produce a SectionAnalysis for this section.
"""

section_analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])
