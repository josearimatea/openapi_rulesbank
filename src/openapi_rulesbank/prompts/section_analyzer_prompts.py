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
        default="",
        description=(
            "The spec that DEFINES this type, as written here, e.g. 'TS 11.111'. "
            "Take it from the row or sentence that names the type and the spec "
            "together. A spec number appearing elsewhere in the section is not the "
            "origin of this type — that one belongs in context_references. Leave "
            "EMPTY ('') only for a CHECK 4 pending import — the type is imported "
            "but this section does not name its spec; a later section supplies it."
        )
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


class ContextReference(BaseModel):
    """A spec this section points at for guidance, without importing a type from it.

    Distinct from SpecReference on purpose. "Further details on creating a resource
    with HTTP PUT are provided in TS 32.158" is worth fetching — that spec settles
    which status code a creation returns — but it defines no type here, and
    recording it as the origin of a type that happens to sit nearby invents a fact
    the document never stated.
    """

    source_spec: str = Field(
        description="The spec the section points at, as written here, e.g. 'TS 32.158'."
    )
    subject: str = Field(
        description=(
            "What the section says is found there, in its own words — e.g. "
            "'details on creating a resource with HTTP PUT', 'the design rules for "
            "RESTful solution sets'. This is the retrieval query's subject."
        )
    )
    why: str = Field(
        default="",
        description="One short line on what this section needs from that spec."
    )


class TypeFilenamePresence(BaseModel):
    """Whether THIS section itself writes out an imported type's YAML filename.

    A type is imported from another spec but its `$ref` needs the YAML file that
    spec publishes it in, which the normative text usually does not write at all —
    not in the section that USES the type, and not in the "data types imported"
    clause either (that one names the SPEC, e.g. "TS 28.623", never the file). Phase 1
    records, per referenced type, whether THIS section happens to write the filename
    out in full; when it does not, the filename is looked up elsewhere (the user's
    auxiliary context, the document's annex, the corpus) before Phase 2 decides.
    """

    type_name: str = Field(description="The referenced type's name.")
    filename_in_section: bool = Field(
        description=(
            "True when THIS section's own text writes the external filename out in "
            "full for this type — a reference of the form "
            "'<file>.yaml#/components/schemas/<ThisType>' appears here. False when the "
            "section only names/uses the type and no such filename is written here."
        )
    )
    filename: str = Field(
        default="",
        description=(
            "The YAML filename copied EXACTLY from this section's text when "
            "filename_in_section is True (e.g. 'TS28623_ComDefs.yaml'); empty when it "
            "is False. Never guess or derive it from the spec number."
        )
    )


class SectionImports(BaseModel):
    """Phase 1 output: the other-spec references this section carries."""

    references: list[SpecReference] = Field(
        default_factory=list,
        description=(
            "Every reference this section makes to a type DEFINED in another spec — "
            "the text linking the two, in one row or one sentence. One entry per "
            "referenced type. Empty when the section imports no type."
        )
    )
    context_references: list[ContextReference] = Field(
        default_factory=list,
        description=(
            "Every OTHER spec this section points at: for procedure, design rules, "
            "or further detail, without importing a type from it. These are fetched "
            "too — the answer often settles how a rule must be written — but they "
            "never become type origins. Empty when the section points at no such "
            "spec."
        )
    )
    filename_presence: list[TypeFilenamePresence] = Field(
        default_factory=list,
        description=(
            "One entry per type in `references`, saying whether THIS section's text "
            "writes that type's external YAML filename in full (and which). Usually "
            "False: the normative text rarely writes the file; it is then looked up "
            "outside the section text before Phase 2 decides. Empty when the section "
            "imports no type."
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

PUT EACH REFERENCE IN ONE OF TWO LISTS. Both are looked up; they differ in what
they mean.

FOR EACH TYPE THIS SECTION USES, RUN THESE THREE CHECKS IN ORDER AND STOP AT THE
FIRST THAT ANSWERS:

  LOOK IN BOTH PLACES: THE TABLES AND THE PROSE. A section states where its
  types come from in either — a table with a "Reference" column, a sentence, or
  both. Read whichever this section has.

  CHECK 1 — IS IT LOCAL? Does the reference for it point at a CLAUSE OF THIS
  DOCUMENT (a clause number like 11.111.2.3, or a heading here that defines it:
  "Type <Name>", "Enumeration <Name>")?
      in a table:  "<TypeName>   11.111.2.3   <description>"     -> LOCAL
      in prose  :  "<TypeName> is defined in clause 11.111.2.3"  -> LOCAL
  If yes: LOCAL. Put it in NEITHER list and move on, however many spec numbers
  appear nearby.

  CHECK 2 — IS IT IMPORTED? Does the reference for it point at ANOTHER
  SPECIFICATION?
      in a table:  "<TypeName>   TS 11.111   <description>"      -> import
      in prose  :  "the <TypeName> type defined in TS 11.111"    -> import
  If yes: `references`, with that spec.

  THE DECISION IS WHERE THE REFERENCE POINTS — into this document, or out of it.
  Nothing else. A document may list its local and its imported types in two
  separate tables, in one table, or only in prose; the pointer decides either way.

  CHECK 3 — DOES THE REFERENCE MEMORY ALREADY ATTRIBUTE IT TO ANOTHER SPEC? The
  memory above was written from a clause that DID pair them — typically the
  clause that declares the imported types, which most documents place after the
  clauses that use them. A section consuming such a type rarely restates the
  spec, and the import does not stop being one for that reason.
  If yes: `references`, with the spec the memory gives.

  CHECK 4 — IMPORTED, ORIGIN NOT YET KNOWN (a deferred import). Reached only when
  1–3 all failed. Does the type sit in the TYPE POSITION of a RESPONSE or
  REQUEST-BODY table — the "Data type" / "response body" column — as an
  UpperCamelCase name, with no local clause beside it, no spec named here, and
  nothing in the memory for it yet?
      in a table:  "| ErrorResponse | 4xx/5xx | Returned in case of an error |"
  Then it is imported from a spec THIS section does not name: the clause that
  pairs it with its spec ("data types imported") comes in a LATER section. Put it
  in `references` with source_spec = "" (empty — origin unknown), so it is
  recorded as a PENDING import; a later section that names the spec fills the
  origin in. This check is ONLY for a type in that table position — never for a
  primitive (string/integer), never for a type a local clause defines (CHECK 1
  caught those), and never for a name in ordinary prose.

  None of the four: the type is not an import. Leave it out of `references`.

WHY CHECK 3 EXISTS: without it, every clause that merely USES an imported type
reports nothing, no lookup runs for that type, and the retrieval that carries its
YAML filename never happens. The origin is still never invented — check 3 reads
an origin some earlier clause declared in text.

`context_references` — a spec the section points at for anything else:
procedure, design rules, further detail. It names no type:
    "Further details on <procedure> are provided in TS 22.222, clause 5.1"
    "The rules for <X> are defined in TS 22.222"

DO NOT PAIR A TYPE WITH A SPEC THE TEXT DID NOT PAIR. If a clause lists
"<TypeName>" in a table and a later sentence says "further details are provided
in TS 22.222", TS 22.222 is NOT the origin of <TypeName>. Put TS 22.222 in
`context_references` and leave <TypeName> out of `references`. Note this is
exactly what the three checks already decide: TS 22.222 answers none of them.

UNSURE ABOUT A SPEC THE SECTION POINTS AT? USE `context_references`. The lookup
still happens, and no false origin enters the type memory.

Document layouts differ: some declare imports in a table, some in prose, some not
at all. Look for the link between type and spec, not for a table.

`filename_presence` — for EACH type you put in `references`, say whether THIS
section's text writes that type's external YAML filename out in full. A filename is
written out only where a reference appears complete, as
    <file>.yaml#/components/schemas/<ThisType>
(e.g. "TS28623_ComDefs.yaml#/components/schemas/NotificationHeader"). If that
appears here for the type, set filename_in_section=True and copy the filename
exactly. If the section only NAMES or USES the type — a table row, a data-type
cell, a sentence, or a "data types imported" row that pairs it with a spec — with no
such full reference, set filename_in_section=False and leave filename empty. A
"data types imported" table names the SPEC (e.g. "TS 28.623"), not the file; the
filename is looked up outside the section text. One entry per type in
`references`; do not guess a filename and never derive it from the spec number.

Judge from the section text plus the memory. Do not invent a type or a spec that
appears in neither. Return empty lists when the section references nothing.
"""

_STEP1_USER = """\
Scope contract — which Management Service these rules are being built for, and
what belongs to it (a capability marked "(not in this YAML)" is the service's but
not this YAML's). A type this section references for another service's sake is
not one to look up here:
{contract_summary}

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
    source_file: str = Field(
        default="",
        description=(
            "The YAML filename that spec publishes this type in, e.g. "
            "'TS28623_ComDefs.yaml'. Take it from the RETRIEVED TEXT: the fetched "
            "chunks often carry a reference another spec wrote in full "
            "(\"$ref: 'TS28623_ComDefs.yaml#/components/schemas/NotificationHeader'\"), "
            "and the filename is the part before the '#'. Copy it exactly. "
            "Leave EMPTY when no retrieved text shows a filename — never derive one "
            "from the spec number and never guess. An empty value makes the "
            "Extractor report the gap; a wrong one points the YAML at a document "
            "that does not exist."
        )
    )
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


class CrossReferenceFact(BaseModel):
    """A fact about how things relate, learned from a referenced specification."""

    subject: str = Field(
        description=(
            "What the fact is about, named as this document names it — a type, an "
            "attribute, a resource, a URI variable."
        )
    )
    fact: str = Field(
        description=(
            "The relationship, in one sentence, as the retrieved text states it — "
            "not an inference from it. One of: who creates or carries the subject; "
            "which object or schema it belongs to; what larger thing it is part "
            "of; what it requires or implies. Prefer the ORIGIN over the "
            "destination: 'carried by the X object, which a consumer creates on "
            "the producer' is actionable, while 'notifications are sent to this "
            "address' is not."
        )
    )
    source: str = Field(
        default="",
        description="Where it was stated, e.g. 'TS 28.622 clause 4.3.22'."
    )


class SectionAnalysis(BaseModel):
    """Phase 2 output: memory updates + guidance for the Extractor."""

    callback_anchor: str = Field(
        default="",
        description=(
            "The anchor this section's callback rules hang off, decided HERE and "
            "used verbatim by the Extractor: exactly \"paths.<path>.<method>\", "
            "the path key and lowercase HTTP method of the operation through which "
            "the consumer supplies its notification address — e.g. "
            "\"paths./{resource}={id}.put\". No placeholder, no <angle brackets>, "
            "no trailing '.callbacks'. "
            "Empty when this section describes no notification delivered to a "
            "consumer-supplied address, or when the cascade below reaches its last "
            "step without an answer."
        )
    )
    callback_anchor_source: str = Field(
        default="",
        description=(
            "Which step of the cascade produced callback_anchor, so the decision "
            "is auditable: 'auxiliary' | 'spec' | 'openapi' | 'convention' | "
            "'unresolved'. Empty when callback_anchor is empty because the section "
            "has no such notification."
        )
    )

    cross_refs: list[CrossReferenceFact] = Field(
        default_factory=list,
        description=(
            "Facts the retrieved definitions state about HOW THINGS RELATE, which "
            "the fields of a schema cannot hold: which operation creates the object "
            "that carries an attribute, which resource hosts an endpoint, which "
            "clause owns a type. At most 2, and only what ANOTHER section would "
            "need in order to map a rule correctly — a fact used only here belongs "
            "in the guidance instead. Empty when the retrieved text adds no such "
            "fact. These accumulate across sections, so a clause that explains a "
            "relationship informs the later clause that has to express it."
        )
    )

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

    defer_external_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Names of EXTERNAL types whose $ref this section CANNOT write yet, so the "
            "Extractor must NOT emit a rule for them here. List a type when ALL of: "
            "(1) it is imported from another spec (in references/memory), (2) NO "
            "filename is known — Phase 1's filename-presence says not in this section "
            "AND the reference memory shows none (the memory already carries any "
            "filename the lookup found — helper context, annex, corpus), and (3) this "
            "section would "
            "otherwise emit a $ref to it (a response body, a property, an allOf base). "
            "Do NOT list a type whose filename is known (write it now) or a local one. "
            "Empty when nothing is deferred. This is the signal the Extractor, "
            "Reflector and Validator read to know the rule was withheld on purpose, "
            "not lost."
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
    maps_operation: str = Field(
        default="",
        description=(
            "For a section that maps ONE operation to HTTP — a resource's method "
            "clause, or an operation-mapping clause — the IS operation name it "
            "maps, e.g. 'createMOI'. Empty for any other kind of section, and "
            "empty when the section covers several operations rather than "
            "defining one. Recorded so a LATER section can find the operation it "
            "needs to refer to without re-reading this clause."
        )
    )
    maps_path: str = Field(
        default="",
        description=(
            "The path key that operation is on, as a path key — the resource part "
            "only, e.g. '/{className}={id}'. A clause title may abbreviate the "
            "base URL with a leading '...' (as in 'Resource \".../{className}={id}\"'); "
            "that prefix is the elided base URL, not part of the path key, so drop "
            "it and start the value at '/'. Never abbreviated with '...'. "
            "Empty when maps_operation is empty."
        )
    )
    maps_method: str = Field(
        default="",
        description=(
            "The HTTP method THIS document maps that operation to, lowercase: "
            "get | put | post | delete | patch. Take it from what the clause "
            "states, never from what the operation's name suggests — a create "
            "operation is not always mapped to post. Empty when maps_operation "
            "is empty."
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
references, and the fetched definitions of those references (RAG). Four jobs, all
of them required — a field you leave empty is one the next section cannot read:

A. UPDATE THE REFERENCE MEMORY. For each external type this section references,
   record what it is: its name, its source spec, THE YAML FILENAME THAT SPEC
   PUBLISHES IT IN, its OWN property names, and a short note.
   THE SPEC AND THE FILENAME ARE TWO SEPARATE QUESTIONS WITH TWO SEPARATE
   SOURCES. Answer them one at a time, in this order:

   Q1 — WHICH SPEC DEFINES IT? Answer ONLY from: a row or sentence of THIS
   SECTION naming the type and the spec together, or the reference memory above.
   THE FETCHED TEXT NEVER ANSWERS Q1 — a spec number sitting near a type in
   retrieved text does not make it that type's origin, and recording one that way
   puts a false origin in the memory every later section then trusts.
   No answer to Q1 — TWO cases, tell them apart:
     - Phase 1 did NOT reference the type at all: it is not an import. Write no
       entry, and do not go on to Q2.
     - Phase 1 referenced it with an EMPTY source_spec (a CHECK 4 pending import —
       a type in a response/request-body table with its origin declared only in a
       later section): it IS an import whose spec is not known yet. Write an entry
       with source_spec = "" (leave the spec empty — never invent one) and go on
       to Q2: the filename may already be known from the memory (looked up by the
       type name), and only if it is not does Q3 defer it. A later section that
       names the spec updates this same entry.

   Q2 — WHICH YAML FILE PUBLISHES IT? Ask this ONLY for a type that answered Q1.
   One specification publishes several files, so its number never yields the
   filename. Record a filename ONLY where some text you were given writes it out
   in full, as
       <filename>.yaml#/components/schemas/<TypeName>
   with <TypeName> being EXACTLY this type — that is a transcription. Otherwise
   leave it empty: never derive it from the spec number and never guess.

   AN EMPTY FILENAME DOES NOT MAKE THE TYPE LOCAL. Q1 already settled that it
   comes from another specification; Q2 only fills in where. Leaving Q2 empty
   keeps the type external with its origin unresolved, which the extractor
   reports as a gap — it never turns it into a type defined in this document.

   Q2b — WHEN SEVERAL FILES ARE POSSIBLE, PICK ONE BY CONTEXT. A deterministic
   lookup runs before you and reads, from the document's annex and the corpus,
   every YAML file a type is tied to. When it finds exactly one, the filename is
   already set and you do nothing. When it finds SEVERAL — the reference memory shows
   the type as "[EXTERNAL — filename AMBIGUOUS, choose by context among: fileA.yaml,
   fileB.yaml, …]" — the same type name is published by more than one spec (the
   clearest case is a generic 'resources' type that each NRM publishes under its
   own file). The code does NOT choose; you do. Pick the ONE file from that list
   whose spec matches the NRM/domain THIS clause is about (its title, its subject,
   the relationship facts), and write it as the filename. If the clause does not
   make the choice clear, leave the filename empty and say so in the note — a
   reported gap beats guessing the wrong file. Choose ONLY from the listed options;
   never type a file that is not among them, and never derive one from a number.

   Q3 — MUST THE EXTRACTOR DEFER THIS TYPE'S $ref? Add the type's name to
   `defer_external_refs` when ALL of these hold:
     - it is EXTERNAL — either it answered Q1 with a spec, OR it is a CHECK 4
       pending import (external with source_spec still empty); both are external,
       AND
     - its filename is unknown to you here: Phase 1's filename-presence list says
       "filename NOT in this section" for it, Q2 found none, AND the reference
       memory shows none for it. The memory already carries any filename the
       lookup found by the TYPE NAME (helper context, annex, corpus) — so a pending
       import whose spec is still unknown may well have its filename there; if the
       memory shows "[<file>]" for the type, it is NOT deferred, AND
     - THIS section would otherwise write a $ref to it (a response body, a property,
       an allOf base).
   That name tells the Extractor NOT to emit the rule now: an external $ref with no
   filename would be written as an internal one, asserting the type is local, which
   contradicts Q1. The rule waits, parked. Do NOT defer a type whose filename you DO
   have — from this section or from the memory (write it in full
   now) or a local type (internal $ref, never deferred). Leave `defer_external_refs`
   empty when nothing is deferred.
   Read the property names out of the fetched definition — they are the
   members the type declares, NOT the schema keywords that structure them (allOf,
   type, properties, required, items, $ref are structure, never property names).
   The fetched text may be split across several chunks and interleaved with other
   types; gather this type's own members from wherever they appear. This is generic
   — any imported type from any spec (imports and allOf composition are pervasive
   in these specs). Return empty when the section references no external types.

B. KEEP WHAT THE FETCHED TEXT SAYS ABOUT HOW THINGS RELATE (cross_refs). Job A
   records a type's SHAPE — its name, spec and members. A referenced clause often
   states something else entirely, which no list of fields can hold, and which
   decides how a LATER section maps its rules. This triage is the only place
   those sentences are ever read, so what you do not record here is lost.

   Read the fetched text for these, and record the ones it states. Each is worth
   recording because of what a later section does with it:

     WHO CREATES OR CARRIES IT — "a consumer creates an X instance", "X is sent in
       the request body of Y". A later section mapping a callback needs the
       operation to anchor it on; mapping a $ref needs the operation that owns the
       body. Record the ORIGIN — which operation or object carries the thing — not
       only where the value ends up: "notifications are sent to this address" says
       nothing about which operation received it.

     WHERE IT BELONGS — "X is an attribute of the Y object", "X is name-contained
       by Z". Tells a later section which schema owns a property, and which
       clause is that schema's home.

     WHAT IT IS PART OF — "X is one of the URI components defined by the common
       URI structure", "X is a patch format for this method". Tells a later
       section that a value belongs to the base URL rather than the path, or which
       media types an operation accepts.

     WHAT IT REQUIRES OR IMPLIES — "to receive notifications a consumer must first
       subscribe", "this is returned only when the resource was created". Tells a
       later section that a rule it is about to emit depends on an operation
       elsewhere.

   Write what the text states, not an inference from it, and name the subject as
   THIS document names it. Prefer the sentence a later section could act on. At
   most 2 per section; empty when the fetched text adds no such fact.

C. DECIDE THE CALLBACK ANCHOR (callback_anchor, callback_anchor_source).

   FIRST, does this section describe a notification delivered to an address the
   CONSUMER supplies — a runtime address the spec names as the consumer's, rather
   than a fixed resource of this service? If not, leave both fields empty and skip
   to D. If yes, the notification is a callback, and its rules hang off the
   operation through which the consumer supplied that address. Work the cascade
   below IN ORDER and STOP at the first step that answers. Do not weigh the steps
   against each other: an earlier step always wins.

     1. AUXILIARY CONTEXT / USER INSTRUCTIONS. Do they state the anchor, or the
        operation to anchor on, for this service? Use it.        -> 'auxiliary'
     2. THIS DOCUMENT, STATED OUTRIGHT. Does a clause NAME the operation that
        carries the address field — this section's own text, a relationship fact,
        or a sibling entry naming the address field itself? Only when it is
        stated; do not work it out by joining facts. Then read that operation off
        the sibling list: an entry reading "maps <operation>: <method> <path>"
        has already been resolved by the triage of that clause, so take its
        <path> and <method> as they stand and answer "paths.<path>.<method>".
        Do not re-derive them from the clause's title, and do not substitute the
        method an operation of that name would usually get.           -> 'spec'
     3. OUR CONVENTION. Nothing states it outright, so apply the rule we adopt
        for these specifications: the consumer supplies its address on an object
        it CREATES on the producer, and such objects are created by this
        service's generic create operation. Scan the sibling list for the entry
        whose <operation> creates a managed object instance, and answer
        "paths.<path>.<method>" from that entry's recorded <path> and <method>.
        Apply this WITHOUT establishing that the create operation's request body
        carries the address field, and WITHOUT a clause joining the subscription
        object to that operation. Supplying the join the clauses leave implicit
        is exactly what this convention is for, so its absence is not a reason to
        fall through to step 5 — this is a modelling decision we have made, not
        an assumption you are making. 'convention' records that the answer is
        ours rather than the document's.                         -> 'convention'
     4. OPENAPI REFERENCE. Only if the convention does not apply to this section
        at all: use the anchor the reference shows for this construct.
                                                                    -> 'openapi'
     5. Nothing above answers -> leave callback_anchor empty and set
        callback_anchor_source to 'unresolved'.

   Emit the anchor as a literal "paths.<path>.<method>": the path key as the YAML
   would key it and the method in lowercase. A clause title may abbreviate the
   base URL with a leading "..." — that prefix is the elided base URL, not part of
   the path key, so the anchor starts at '/'. Never a placeholder, and never an
   invented path or method — step 5 exists for that. An anchor reached by step 3
   is not an invention: the convention is a decision already made, and applying it
   is following an instruction, not supposing a fact.

D. GUIDE THE EXTRACTOR:
   0. RECORD WHAT THIS SECTION MAPS. When it maps ONE operation to HTTP — a
      resource's method clause, an operation-mapping clause — fill maps_operation
      with the IS operation name, maps_path with the path key it is on, and
      maps_method with the method THIS document gives it, read off the clause
      rather than assumed from the operation's name. Leave all three empty for
      any other section, and when the section covers several operations at once.
      This is for the sections that come after: one of them will have to name an
      operation defined here, and this is how it learns of it without reading
      this clause.
   1. CLASSIFY the section by its content: type_definition, enumeration, mapping,
      resource, or other. emits_schema_properties is True ONLY for
      type_definition/enumeration. Keep `guidance` CONSISTENT with this flag —
      when emits_schema_properties is False, the guidance must not tell the
      Extractor to emit a schema or its properties (a mapping section maps
      parameters to a request body that references types defined elsewhere; it
      does not define those types here).
      DECIDE 'resource' vs 'mapping' BY THIS TEST, in this order:
        - If the section's title is an HTTP method (POST, GET, PUT, PATCH, DELETE)
          OR its body defines one operation via the operation tables — URI query
          parameters, request data type, and RESPONSE codes (204/2xx and 4xx/5xx) —
          then it is 'resource'. A resource section OWNS that operation and its
          responses; they are defined HERE and nowhere else. This is true even
          though the tables "map" IS elements to HTTP: a per-operation table with
          response codes is a resource, never a mapping.
        - 'mapping' is ONLY a "Mapping of <IS operation/notification> input/output
          parameters to SS equivalents" table: rows of parameter-name ↔ SS-type,
          with NO response-code rows. It references types defined elsewhere and
          owns no responses.
      When unsure between the two, the presence of response-code rows (204, 4xx,
      5xx) decides it as 'resource'.
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
Scope contract — which Management Service these rules are being built for, and
what belongs to it: its capabilities (a "(not in this YAML)" one is the service's
but not this YAML's), the terms that signal its content, and the other services
to keep out. Every decision below is about THIS service — above all the callback
anchor, whose path must be a resource of this service and not one you recall from
another:
{contract_summary}

Section ID   : {section_id}
Section Title: {section_title}

Auxiliary context — supporting documents and any instruction the user gave for
this run. It is the FIRST step of the callback-anchor cascade and outranks every
other source: where it states how something maps for this service, that is the
decision:
{helper_context}

Section content:
{section_content}

Reference memory so far ('type -> {{spec, fields}}'):
{ref_memory}

Phase 1 explanation of this section's external references:
{phase1_explanation}

Phase 1's per-type filename presence — whether THIS section's text writes each
imported type's external filename in full, or only uses the type:
{filename_presence}

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


# ---------------------------------------------------------------------------
# Phase 3 — decide which of this section's rules to SKIP (already owned elsewhere)
# ---------------------------------------------------------------------------

class SectionScope(BaseModel):
    """Phase 3 output: the rules this section's TEXT supports, and which to skip."""

    candidate_rules: list[str] = Field(
        default_factory=list,
        description=(
            "The rules this section's OWN CONTENT supports — one short line each, "
            "anchored strictly in the section text (a row, a table, a stated code or "
            "media type). This is the positive scope: the Extractor should emit these "
            "and nothing beyond them. Do NOT list a rule the text does not state, "
            "even if a related section or the retrieved context has it — a mapping "
            "table that lists no response codes yields NO response candidates. "
            "EXCLUDE INHERITED FIELDS: when the pre-analysis verdict lists inherited "
            "fields (this type is an allOf of a base), those fields come from the base "
            "via the single allOf $ref and are NOT this section's to emit — never make "
            "one of them a candidate, even though the 3GPP attribute table lists them "
            "alongside the own attributes. Candidate the allOf base itself and the "
            "type's OWN (non-inherited) attributes only. List each item separately; "
            "never merge concatenated items (e.g. a '4xx/5xx' cell is TWO candidates, "
            "one 4XX and one 5XX)."
        )
    )
    skip: list[str] = Field(
        default_factory=list,
        description=(
            "The subset of candidate_rules that another section PROVABLY already "
            "owns. A candidate belongs here ONLY IF a line LITERALLY PRESENT in "
            "'Rules already created' covers the same rule_type and same single "
            "target (same path+method, same schema+property, or same "
            "path+method+one specific status code). If no such line exists — including "
            "when 'Rules already created' is empty — the candidate MUST NOT be "
            "skipped. Never skip on a hunch or because the verdict says 'mapping'; "
            "presence in 'Rules already created' is the only evidence. Same wording "
            "as the candidate it refers to. A concatenated '4XX/5XX' is never one "
            "match, check each code. Empty when nothing provably overlaps."
        )
    )
    reasoning: str = Field(
        default="",
        description="One line on what the section's text supports and what overlaps."
    )


_STEP3_SYSTEM = """\
You scope ONE section about to be extracted, in two steps, from its OWN content.

1. CANDIDATE RULES — list the rules this section's TEXT actually supports, each
   anchored in a specific line, row, table cell, stated status code or media type
   of the section content. This is the positive scope handed to the Extractor: it
   should emit these and nothing beyond them. Anchor strictly in the text — if the
   section does not state something, it is NOT a candidate, no matter what a related
   section or the retrieved context contains (a notification-parameter mapping
   table that lists no response codes produces NO response candidates).
   INHERITED FIELDS ARE NOT CANDIDATES: if the pre-analysis verdict lists inherited
   fields (this type is an allOf of a base), never candidate any of them — they come
   from the base via the single allOf $ref, not from this section, even though the
   attribute table lists them next to the own attributes. Candidate the allOf base
   and the OWN (non-inherited) attributes only. Listing an inherited field here would
   send the Extractor to emit a rule the Validator only discards again.
   List each item on its own; NEVER merge concatenated items — a "4xx/5xx" cell is
   two candidates (4XX and 5XX), a "PUT/PATCH" is two, etc.

2. SKIP — decide, on your own judgement, which candidates another section has
   ALREADY created, so this section does not duplicate them. This is YOUR call from
   the evidence, not an instruction to copy: weigh the section content, the sibling
   sections listed below, and the rules already created. A candidate goes in skip
   when the evidence shows another section already emitted that exact rule
   (same rule_type and same single target: same path+method, or same
   schema+property, or same path+method+one specific status code).
   THE SINGLE HARD LIMIT — read it literally:
     A response code, path, or property this section's OWN operation table defines
     is OWNED BY THIS SECTION. You may skip it ONLY when a line in 'Rules already
     created' literally shows another section already emitted that same target.
     If 'Rules already created' does not contain that target — or is empty — the
     candidate is NOT a duplicate: it STAYS, you MUST NOT skip it.
   Do NOT infer a skip from the pre-analysis kind. A section labelled 'mapping'
   still owns every response code and parameter its own table lists; the label is
   context, never a reason to skip. A "4XX/5XX" is two separate targets — decide
   4XX and 5XX independently. When the evidence is not clear, leave the candidate
   OUT of skip: a wrong skip silently deletes a real rule, while a missed skip is
   caught later by the Validator.

Never invent a candidate the text does not support; never skip a whole section.
"""

_STEP3_USER = """\
Scope contract — which Management Service these rules are being built for, and
what belongs to it. A candidate has to be this service's: one whose target names
another service's resource, or a capability marked "(not in this YAML)", does not
belong in the list however clearly the text states it:
{contract_summary}

Section ID   : {section_id}
Section Title: {section_title}

Section content (your PRIMARY evidence — candidates and skips are judged from here):
{section_content}

Pre-analysis of this section (CONTEXT ONLY — an earlier pass's opinion, not an
order; you may disagree with it based on the content above):
{section_verdict}

Sibling sections in this same service (id, title, and their decided kind) — use
this to tell whether ANOTHER section is the real owner of a candidate:
{sibling_summary}

Rules already created in other sections of this service (the ONLY evidence that a
candidate is already owned; if a target is absent here it is NOT owned):
{existing_rules}

List the candidate rules, then the skip subset your own judgement supports (empty
if nothing is provably owned elsewhere).
"""

section_scope_prompt = ChatPromptTemplate.from_messages([
    ("system", _STEP3_SYSTEM),
    ("human",  _STEP3_USER),
])
