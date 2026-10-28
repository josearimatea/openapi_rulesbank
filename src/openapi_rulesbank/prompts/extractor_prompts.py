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
from openapi_rulesbank.prompts.rule_type_definitions import rule_type_definitions_for_template

# The one definition of each rule type, the same in every stage (see
# prompts/rule_type_definitions.py).
_RULE_TYPE_DEFINITIONS = rule_type_definitions_for_template()
from pydantic import BaseModel, Field
from openapi_rulesbank.schemas.rules import RawRule


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
    deferred_rules: list[RawRule] = Field(
        default_factory=list,
        description=(
            "Rules this section DOES state but whose $ref points at a type that is "
            "HELD (origin not stated yet, or external with no filename — see the "
            "Section analysis and the reference memory). Write each completely, with "
            "the $ref as \"$ref: '#/components/schemas/<Type>'\" and the type in "
            "`references` (file empty); the code completes the target when a "
            "section states where the type is defined. Empty when nothing is held."
        ),
    )
    missing_context: list[str] = Field(
        default_factory=list,
        description=(
            "Facts you needed to map a rule correctly and could NOT find in the "
            "section or in any context given to you — at most 2, empty when there "
            "are none. Report a fact only when its absence changed what you "
            "emitted: you fell back to a shape you know is not the right one, or "
            "left a required part of openapi_object unresolved. Each entry names "
            "the one fact and why it blocked the mapping, in a sentence — not a "
            "rule you chose not to extract, not something already in the context, "
            "and never a request for a whole topic. "
            "Example: 'Which operation's request body carries the consumer's "
            "notification address — needed to anchor the callback on it, so the "
            "notification was emitted as a path instead.'"
        ),
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your task is to extract concrete OpenAPI rules from a section of a 3GPP document.
Each rule must have a rule_type, a clear rule_text, and a precise openapi_mapping.

SOURCE OF RULES — "Section content" is the ONE and ONLY place a rule may come
from; everything else only helps you phrase what is already there:
  - Extract a rule ONLY when its subject appears in the "Section content" of the
    user message. Every rule must trace to a specific line, row, or statement in
    THAT text — if you cannot point to where in the section content it is stated,
    do not emit it.
  - The other inputs are REFERENCE, never a source. Do NOT turn anything found only
    in the "Stage 2 context", the "Sibling stage 3 context", the OpenAPI reference,
    or the auxiliary context into a rule. They exist solely to correct or complete
    a rule the section content already supports — e.g. the canonical operation
    name, the real resource path, the exact schema name/casing. If a property,
    path, or code is in one of those contexts but NOT in the section content, it is
    another section's rule — leave it out.
  - If the "Section content" defines nothing concrete of its own (it is empty, or
    just a heading / a "###" marker / a cross-reference like "See clause X"), return
    an EMPTY list. Never populate it from the contexts.
  - Do NOT infer or hallucinate rules not explicitly stated or clearly implied by
    the section text.
  - If the section content contains a table, extract one rule per relevant row —
    EXCEPT rows the pre-analysis marks as inherited (see below): those are covered
    by the single allOf $ref, so skip them, do not also emit them as properties.
  - If the section content defines multiple attributes, extract one rule per
    attribute, again SKIPPING any the pre-analysis lists as inherited.
  - Ignore introductory or background text with no concrete mappable content.

SERVICE SCOPE — the rules must belong to the target service's OpenAPI YAML:
  - The SERVICE CONTRACT in the user message (when present) defines the scope of
    the specific OpenAPI YAML these rules are being built for.
  - Extract ONLY rules that belong to this service — i.e. rules aligned with its
    purpose, capabilities, key_concepts, or inclusion_keywords.
  - Do NOT extract a rule that belongs to another Management Service, even if the
    section mentions it (see the contract's exclusion_keywords / out_of_scope_signals).
    Such content is out of scope for this YAML — skip it.
  - When the contract is absent, extract every concrete rule the section supports.

OPERATION NAMING — use the canonical IS operation name for source_name:
  - The "Related spec context" in the user message is retrieved from the source
    3GPP document itself. For a stage 3 (HTTP mapping) section it surfaces the
    related stage 2 definition, which carries the canonical IS operation name
    (e.g. "getMOIAttributes" for an HTTP GET on a managed object resource).
  - For path_operation, response, and request_body rules, set source_name to that
    IS operation name — the one the spec defines for this method and resource.
  - NEVER use the bare HTTP method (e.g. "GET") as source_name, and NEVER invent
    an operation name (e.g. "retrieveMOI") that is not in the section or the
    related spec context.

USING THE TWO SPEC CONTEXTS — the section is extracted alone, so facts it depends
on live in other sections, supplied as "Stage 2 context" and "Sibling stage 3
context". Neither is a rule source; use them to get THIS section's rules right:

  - RESOURCE PATH — never build a path from an operation or notification name.
    A section that maps an operation/notification (e.g. a table row
    "<notificationName>  POST  /<resource>") gives the real URI in its own
    "Resource URI" column; when the row points elsewhere, the sibling stage 3
    context holds the resource section that defines that URI. Use that URI as the
    path. "paths./<notificationName>" (a path named after the notification) is
    always wrong — the path is "/<resource>". (And when the URI is the consumer's
    address, it is a callback, not a path — see RULE TYPE DEFINITIONS.)

  - NAMES: THREE KINDS, THREE PLACES, NEVER CONVERTED INTO EACH OTHER.
    A NAME IS TAKEN FROM ITS DEFINING HEADING. The heading is the source — the
    sibling clause list in the stage 3 context shows those headings even when the
    defining clause is not this section. Find the heading that names the thing and
    copy its spelling. The three forms below tell you WHICH heading to look in;
    they are a check on what you copied, never a way to produce a name. A spelling
    can have the right form and still be the wrong name: a data-type column may
    write a notification in flawless lowerCamelCase and still drop a segment the
    heading carries. If a candidate matches no heading, do not use it however
    well-formed it looks — report it in missing_context instead.
    The spec writes the same concept differently depending on what it is naming.
    Each kind is COPIED from the place listed below — never produced by re-casing
    another kind, and never taken from a data-type column:

      TYPE (a schema) — UpperCamelCase, acronyms normalised to an initial capital.
        From the heading "<clause> Type <Name>" / "Enumeration <Name>".
        Use in: openapi_object "components/schemas/<Name>", and in every $ref.
        e.g. "NotifyMoiCreation", "AlarmId", "PerfMetricDirection".

      NOTIFICATION — lowerCamelCase, acronyms kept as the spec capitalises them.
        From the heading "<clause> Notification <name>".
        Use in: a callback's source_name and its "callbacks.<name>" segment.
        e.g. "notifyMOICreation", "notifyNewAlarm".

      PROPERTY (a field) — lowerCamelCase, acronyms normalised.
        From the attribute/parameter table of the type that owns it.
        Use in: openapi_field "properties.<name>", "parameters[...name=<name>]",
        and the entries of "required".
        e.g. "notificationId", "alarmId", "objectInstance".

    The notification and the type it carries are DIFFERENT NAMES for different
    things, and differ by more than the first letter: notification
    "notifyMOICreation" carries type "NotifyMoiCreation" — "NotifyMOICreation"
    (re-cased notification) and "notifyMoiCreation" (re-cased type) are both
    wrong. Likewise property "notificationId" ↔ type "NotificationId".
    Where a data-type column disagrees with a defining heading, the HEADING wins:
    columns are hand-written prose and are where the spec's typos live.
    Do not coin a schema name from a notification or property mentioned in prose.

  - WHO OWNS A SCHEMA — schema_property rules come ONLY from a section that
    DEFINES a data type, never from one that MAPS parameters. Tell them apart by
    the section's own table/text, not its title:
      * A DEFINITION section states the type's fields and their JSON types — its
        table reads "Definition of type <Name>" with columns like "Attribute name
        / Data type". THIS is where schema_property rules come from.
      * A MAPPING section translates an IS operation or notification to its SS
        (HTTP) form — its text says the parameters are "mapped to SS equivalents"
        and its table reads "Mapping of IS ... to SS equivalents". Even though it
        lists parameter names, it is NOT a type definition.
    From a MAPPING section, extract the HTTP-level rules it establishes
    (path_operation, callback, request_body, response) but DO NOT emit any
    schema_property — those attributes are defined, with their real JSON types, in
    the matching "Definition of type <Name>" section, their single home. A mapping
    section titled "Notification <name>" still counts as a mapping section: emit
    its callback/request_body, not its parameters as schema properties. Emitting
    them in both places produces the same property twice with conflicting types
    (e.g. "string" in the mapping vs {{"type":"string","format":"date-time"}} in
    the definition).

  - OWN vs INHERITED PROPERTIES — when the pre-analysis (see "Pre-analysis of this
    section" in the user message) reports an imported base, it gives you TWO things:
    the base's $ref and the exact list of INHERITED field names. Then:
      * emit exactly ONE schema_property rule with openapi_field "allOf" and the
        base's $ref as its value;
      * emit a properties.<name> rule for every OTHER attribute — the type's own;
      * do NOT emit any properties.<name>, and do NOT list in "required", the
        inherited field names from that list. They already come from the base via
        the allOf $ref; repeating them here duplicates the header (once by $ref,
        once inline) and is wrong.
    This overrides "one rule per attribute" above: an inherited attribute is not an
    own attribute. When the pre-analysis reports no base, treat all attributes as
    own.

  - DO NOT REPEAT A SIBLING'S RULE — if a rule clearly belongs to a sibling stage 3
    section (its path, its schema, its properties), leave it to that section. Emit
    only the rules this section is the home of, so one operation/property is not
    extracted twice under slightly different mappings.

SCHEMA_PROPERTY IS GATED BY THE PRE-ANALYSIS — read this before the rule types:
  Emit `schema_property` rules ONLY when the "Pre-analysis of this section" marks
  this section a schema home (a type definition or an enumeration). If the
  pre-analysis says it is NOT a schema home (a mapping or a resource section),
  emit NO schema_property at all — only its HTTP-level rules (path_operation,
  callback, request_body, response). A "Mapping of ... to SS equivalents" table is
  NOT a type definition, however many attribute-name + type rows it lists: those
  attributes are defined in their own "Type <Name>" section, which is their home.
  This gate overrides "extract one rule per attribute/row" below — that rule
  applies to attribute ROWS only inside a schema-home section.

  SECOND GATE — inherited fields vs allOf: when the pre-analysis reports an
  embedded base ($ref) and a list of inherited fields, those fields come from the
  base via the single allOf $ref. NEVER also emit them as properties.<name> or
  list them in "required" — a field is EITHER inherited (via allOf) OR an own
  property, never both. Emit properties only for attributes NOT in the inherited
  list. Emitting an inherited field inline duplicates the base and is wrong.

WHERE EVERY FIELD'S VALUE COMES FROM — one question, asked for each field below:
IS THE VALUE SPECIFIED IN THE SPEC?

  YES — the section, or a sibling clause or message example in the spec context,
        states it. Use THAT value. It overrides any default.
  NO  — what kind of value is it?
        An OPENAPI DETAIL — a media type, a JSON Schema form, anything the OpenAPI
          reference above shows a value for. Fill it from the reference, emit the
          rule, AND report it in missing_context, naming the field and the value.
        AN IDENTIFIER OF THIS SERVICE — a path key, an operation name, a schema
          name, a field name. Take it from a source, in this order:
            1. this section, or a clause of this document in the context above —
               a section stating "Resource URI: {{X}}" has declared its path key,
               and that is the value, not a gap;
            2. the auxiliary context / the user's instructions for this run;
            3. the OpenAPI reference, where it shows a standard value for that
               field.
          Never suppose one and never invent one. Forbidden above all is carrying
          one over from where it does not apply: another Management Service in
          this document, a service you have seen before, or a name you coined
          because it reads plausibly. A path such as "/notificationSink" or
          "/eventListener" written into a service whose clauses never mention it
          is that failure — the rule looks complete and points at a resource that
          does not exist here. Nor is a non-identifier a safer answer: an empty
          path key, or the notification's own name used as one, is not.
          When none of the three sources gives it, emit what the section does
          support and report the missing identifier in missing_context.

  Both halves of the OPENAPI DETAIL case are required. Emit the rule: silence in a
  clause is never a reason to withhold one, to leave a field generic, or to write
  a placeholder — a 3GPP clause says what the interface does, not how OpenAPI
  writes it down, so this case is routine. And report it: you cannot see the
  clauses you were not given, so the value may be stated in one of them; the
  Reflector is what goes and checks. A field filled by default and left
  unreported is a value nobody will ever verify.

    request_body openapi_value — the media type. The clause names one (the patch
      formats, say) -> use it. It names only the body's DATA TYPE, as most do ->
      the media type from the reference's request-body examples, and report.

    schema_property openapi_value — the JSON Schema type. The clause gives the
      attribute's own type, so map it: a named type it defines or imports -> the
      $ref to that schema; a plain type -> the JSON Schema form the reference
      shows for it, e.g. a date-time string as
      {{"type": "string", "format": "date-time"}}.

    path_operation / path_parameter / query_parameter / response / request_body
    openapi_object — the path key. The clause gives a Resource URI, which is the
      base URL plus the resource; a path key holds the resource part alone, so
      use that part (see rule 1 below). Whether a URI is a path at all is settled
      by RULE TYPE DEFINITIONS (callback, path_operation). The anchor a callback
      needs, "paths.<path>.<method>", is not a value the OpenAPI reference can
      supply: it is settled by a clause of this document, or failing that by the
      callback-anchor convention below.

MODELLING CONVENTIONS — APPLY THESE. They settle the joins a 3GPP clause leaves
implicit, and they are how the published YAMLs express them. Where one applies,
use it and emit the rule: do not report the join as missing, do not leave a
placeholder, do not fall back to a shape you know is wrong.

  - CALLBACK ANCHOR. You do not work this out. The pre-analysis sees the sibling
    clauses that you cannot, and where this section's notification goes to a
    consumer-supplied address it hands you the anchor already resolved, under
    "CALLBACK ANCHOR" in the section analysis. Use that string verbatim. When it
    says the anchor is unresolved, report it — never invent one.
  - The "3GPP CONVENTION" parts of RULE TYPE DEFINITIONS are conventions of this
    same kind (the base URL in servers, base-URL variables are no path
    parameters, request body media type, named types as $ref, allOf for a base
    type): apply them the same way.

  PRECEDENCE, highest first: what a clause of this document states; the auxiliary
  context or the user's instructions; these conventions. A convention is the
  answer only where the two above are silent, and it is overridden the moment
  either speaks.

RULE TYPES — for each rule, assign exactly one rule_type and fill openapi_mapping accordingly:

1. path_operation  — one rule per HTTP method on a resource path
   rule_type      : "path_operation"
   source_name    : the IS operation name (e.g. "createMOI", "modifyMOIAttributes")
   openapi_object : the RESOURCE part of the path template, prefixed with "paths."
                    A 3GPP "Resource URI" line gives the whole address — the base
                    URL every MnS of the document shares, then the resource:
                      {{MnSRoot}}/<ServiceName>/{{MnSVersion}}/{{URI-LDN-first-part}}/{{className}}={{id}}
                    The leading part is the base URL: its variables are the ones
                    the URI-variables table defines by pointing at TS 32.158
                    (which specifies the URI structure common to every MnS), plus
                    the service name itself. In OpenAPI that part lives under
                    `servers`, never in a path key. Drop it and keep the resource,
                    starting at "/":
                      Right: "paths./{{className}}={{id}}"
                      Wrong: "paths.{{MnSRoot}}/ProvMnS/{{MnSVersion}}/{{URI-LDN-first-part}}/{{className}}={{id}}"
                      Wrong: "paths./{{MnSVersion}}/{{URI-LDN-first-part}}/{{className}}={{id}}"
                      Wrong: "paths./.../{{className}}={{id}}"   (never abbreviate)
                      Wrong: "paths.<resource>"                (never a placeholder)
                    Write the SAME key every time for the same resource: one
                    resource is one path, and rules spelling it differently become
                    separate paths in the output.
   openapi_field  : exactly one HTTP method in lowercase: get | put | post | delete | patch
                    NEVER combine two methods (e.g. never "put, patch") — create two rules.
   openapi_value  : the HTTP method in uppercase: GET | PUT | POST | DELETE | PATCH

2. schema_property — one rule per attribute, field, or schema-level constraint in a data model
   rule_type      : "schema_property"
   source_name    : the attribute, field, or constraint name (e.g. "nrCellDuId", "scopeType")
   openapi_object : exactly "components/schemas/<SchemaName>" — one level, never a
                    nested path (no ".../properties/<x>/..." and never a keyword such
                    as additionalProperties as the schema name).
                    <SchemaName> MUST be the name the spec gives the data type in the
                    HEADING of the section that defines it. These headings read
                    "<clause> Type <Name>" or "<clause> Enumeration <Name>", and
                    <SchemaName> is ONLY the <Name> part — copied with its exact
                    spelling and casing. The leading clause number and the label word
                    "Type" / "Enumeration" are NOT part of the name: strip them.
                    e.g. heading "12.3.1.2.4.6.4 Enumeration PerfMetricDirection"
                         → SchemaName "PerfMetricDirection" (NOT "Enumeration
                         PerfMetricDirection", NOT "12.3.1.2.4.6.4 ...").
                    Do NOT use the name of a property, of a notification, or of a table
                    column as the schema name, and never append a suffix the spec does
                    not use (no "Request", "RequestBody", "Response").
   openapi_field  : EITHER a named property: "properties.<propertyName>" e.g. "properties.nrCellDuId"
                    OR a schema-level keyword that constrains the whole schema:
                    "enum" | "required" | "oneOf" | "anyOf" | "allOf" | "items" |
                    "additionalProperties"
                    WHEN A PROPERTY IS AN ARRAY OR AN OPEN MAP, address the element
                    schema by NESTING ONE keyword under the property — never emit a
                    bare "properties.<name>" with value "array", which drops the
                    element type:
                    - array        : "properties.<name>.items"
                    - open map     : "properties.<name>.additionalProperties"
                    This is the only nesting allowed. A deeper path
                    ("properties.<name>.description", "...type") is WRONG — that
                    detail is the value of the property's schema, carried in
                    openapi_value, not part of the address.
   openapi_value  : depends on openapi_field:
                    - for "properties.<name>": the JSON Schema type ("string" |
                      "integer" | "boolean" | "object"), or, for a named type, a
                      $ref in the form the REFERENCE CASES below give it (internal
                      for a local type, external with the file for an imported one,
                      held when not settled)
                    - for "properties.<name>.items" (array element) and
                      "properties.<name>.additionalProperties" (map value): the
                      element/value schema. WHAT DISTINGUISHES THE CASES IS THIS
                      VALUE — a $ref when the element is a named type (form per the
                      REFERENCE CASES), or a plain type ("string" | "integer" | ...)
                      when it is not. Prefer a $ref whenever the element is a named
                      schema, rather than restating it inline.
                    - for "enum"      : the allowed values, e.g. "[add, remove, replace]"
                    - for "required"  : the required property name(s), e.g. "[id]"
                    - for "oneOf"/"anyOf"/"allOf": the list of referenced schemas,
                      each $ref in its REFERENCE CASES form, e.g. two local ones:
                      "[$ref: '#/components/schemas/A', $ref: '#/components/schemas/B']"
                    - for "items"     : the item type, or a $ref (REFERENCE CASES form)
                    - for "additionalProperties": the value type or "true"/"false"

3. path_parameter  — one rule per path variable in the RESOURCE part of the path
   rule_type      : "path_parameter"
   source_name    : the path parameter name (e.g. "id", "className")
                    ONLY a variable that appears in the path key you wrote for
                    openapi_object. A "URI variables" table lists the base-URL
                    variables too, and those yield NO rule — the table itself
                    tells them apart: a base variable is defined by a pointer to
                    TS 32.158, which specifies the URI structure common to every
                    MnS, while a resource variable is described right there in
                    terms of what it identifies.
   openapi_object : the path key that carries this variable. Use the path key
                    ALONE (e.g. "paths./{{className}}={{id}}") when the parameter
                    is shared by every method of the path — the path item is that
                    shared level. Append ".<method>" (e.g.
                    "paths./alarms/{{alarmId}}.patch") when the parameter belongs
                    to a single operation, which is how the section presents it.
                    Resource part only — never the base URL, never abbreviated
                    with "...".
   openapi_field  : "parameters[in=path,name=<paramName>]"
   openapi_value  : the parameter schema type ("string" | "integer" | "boolean" |
                    "array"), or, for a named type, a $ref in its REFERENCE CASES
                    form (internal / external with the file / held)

4. query_parameter — one rule per query string parameter
   rule_type      : "query_parameter"
   source_name    : the query parameter name (e.g. "scope", "filter", "attributes")
   openapi_object : the path key AND the HTTP method that accepts this parameter,
                    e.g. "paths./{{className}}={{id}}.get" — resource part of the
                    path (never the base URL, never abbreviated with "...") followed
                    by ".<method>". A query parameter belongs to ONE operation: the
                    table that lists it sits under a specific HTTP method, and the
                    text names it ("The HTTP GET operation supports..."). Write the
                    method from that text. NEVER write the path alone — without the
                    method the generator cannot tell which operation the parameter
                    belongs to, and spreads it across every method of the path.
                    (Contrast path_parameter above: a path variable may sit at the
                    path level, WITHOUT a method, when every method shares it — a
                    query parameter never does.)
   openapi_field  : "parameters[in=query,name=<paramName>]"
   openapi_value  : the parameter schema type ("string" | "integer" | "boolean" |
                    "array"), or, for a named type, a $ref in its REFERENCE CASES
                    form (internal / external with the file / held)

5. response        — one rule per HTTP response code or range for an operation
   rule_type      : "response"
   source_name    : the IS operation name (e.g. "getMOIAttributes", "createMOI")
   openapi_object : the COMPLETE path and method, e.g.
                    "paths./{{className}}={{id}}.put.responses"
                    NEVER abbreviate with "..." — always write the full URI template.
   openapi_field  : use exactly the field as defined in the section content:
                    - Specific code: "200" | "201" | "204" | "400" | "404" | "500" | etc.
                    - Range (when section defines a range explicitly): "1XX" | "2XX" |
                      "3XX" | "4XX" | "5XX"
                    - Catch-all for undeclared codes (when section uses "default"
                      or defines a fallback for all other codes): "default"
                    Do NOT invent a field not present in the section. Follow what
                    the section says exactly.
                    TWO HARD RULES FOR RANGES — no exceptions:
                    1. ALWAYS UPPERCASE the X. The section may write "4xx"/"5xx";
                       you MUST emit "4XX"/"5XX". Lowercase "4xx" is INVALID.
                    2. ONE code per rule, NEVER combined. A cell that reads
                       "4xx/5xx" or "4xx or 5xx" is TWO SEPARATE response rules —
                       one with openapi_field "4XX" and one with "5XX". Never emit a
                       single rule for both, and never let the rule's description say
                       "4xx or 5xx" — each rule names ONLY its own single range.
   openapi_value  : "$ref: '#/components/responses/<Name>'", or a $ref to the body's
                    named type in its REFERENCE CASES form (internal / external with
                    the file / held), or a brief schema description

6. request_body    — one rule per operation that accepts a request body
   rule_type      : "request_body"
   source_name    : the IS operation name (e.g. "createMOI", "modifyMOIAttributes")
   openapi_object : the COMPLETE path and method, e.g.
                    "paths./{{className}}={{id}}.put.requestBody"
                    NEVER abbreviate with "..." — always write the full URI template.
   openapi_field  : "content"
   openapi_value  : the media type: "application/json" |
                    "application/merge-patch+json" | "application/yang-data+json" |
                    etc. If several are named, create one rule per media type.

7. security_scheme — one rule per security or authentication requirement
   rule_type      : "security_scheme"
   source_name    : the security scheme name (e.g. "OAuth2", "BearerAuth")
   openapi_object : "components/securitySchemes/<SchemeName>"
   openapi_field  : "type"
   openapi_value  : "oauth2" | "http" | "apiKey" | "openIdConnect"

8. callback        — one rule per notification the PRODUCER sends to the consumer
   rule_type      : "callback"
   source_name    : the notification's own name, copied from wherever the spec
                    writes it — this section, a mapping table in the spec
                    context, or the reference memory. NEVER coin one: a name you
                    assembled yourself ("notifyEvent", "notifyChange") is wrong
                    even when it reads plausibly, because nothing downstream can
                    match it to the schema the notification carries.
                    Use the NOTIFICATION name here (see "NAMES: THREE KINDS"
                    above): lowerCamelCase from the heading "<clause> Notification
                    <name>", acronyms as written. NOT the type it carries — that
                    UpperCamelCase name belongs in the request_body's $ref.
                    ONE RULE PER NOTIFICATION. Where a table lists several
                    notifications on the same resource, that is one callback rule
                    each, under its own name — never a single rule standing for
                    all of them.
                    And the same notification EXACTLY ONCE. A data-type column may
                    spell it a third way, disagreeing with both headings (a
                    dropped segment, different casing); it is still that one
                    notification. Emitting it twice, once per spelling, is always
                    wrong: take the "Notification <name>" heading, emit one rule,
                    and report the disagreement in missing_context.
                    If no name is written anywhere you were given, do not invent
                    one: report it in missing_context.
   openapi_object : the full anchor, "paths.<path>.<method>.callbacks.<notificationName>"
                    where <path>.<method> is the operation in which the consumer
                    supplies its address (see ANCHOR IT ON THE OPERATION below).
   openapi_field  : the HTTP method the producer uses to deliver it, lowercase
                    (normally "post")
   openapi_value  : the runtime expression naming the consumer address field, as
                    written in the spec, e.g.
                    "{{request.body#/notificationRecipientAddress}}" or
                    "{{request.body#/consumerReference}}"

CALLBACK OR PATH — whether a notification is a callback or a path_operation is
  decided by the callback and path_operation definitions in RULE TYPE
  DEFINITIONS, nothing else. Being a notification decides nothing, and neither
  does where the resource is hosted. When the section itself does not give the
  delivery URI, read it where the spec gives it: the heading of the resource
  clause this section belongs to, or the sibling clauses.

  ANCHOR IT ON THE OPERATION WHERE THE CONSUMER SUPPLIES THAT ADDRESS. The
  section describing the delivery almost never names that operation — it is
  elsewhere, and often in another specification. Work down these, in order, and
  stop at the first that applies:

    1. This spec names a subscription operation (a "subscriptions" resource, a
       subscribe operation) -> anchor on it.
    2. A clause you were given states which operation's REQUEST BODY carries the
       address field -> anchor on that operation.
    3. The relationship facts above say the address belongs to an object the
       consumer CREATES on the producer -> creating an object is this service's
       own create operation on the resource that holds it: anchor on that
       operation of that resource. (A generic provisioning service creates any
       object through its single resource, so that resource's create method is
       the anchor.)
    4. None of the above -> emit what you can and report the missing anchor in
       missing_context. Do NOT write a placeholder such as
       "paths.<subscription-path>" or "paths.<owning-path>": an unresolved anchor
       is worse than a reported gap.

  Read the anchor operation, the address field name and the delivery method from
  THIS spec's text; never assume the values seen in other specifications.

  The relationship block above IS this spec's text — a clause's own words, carried
  here because the clause that states them is not the clause that needs them. So
  read it: it is where steps 2 and 3 find their answer, and the section that
  describes a delivery almost never names the operation itself. Use what it
  states; assume nothing it does not.

  The callback's payload and responses are separate rules whose openapi_object
  continues the same anchor, so the structure matches the target YAML:
    request_body : object "paths.<path>.<method>.callbacks.<name>.<cbMethod>.requestBody",
                   field "content", value the media type
    response     : object "paths.<path>.<method>.callbacks.<name>.<cbMethod>.responses",
                   field the status code, value the schema $ref or description

REFERENCES — structure every schema reference so the YAML builder needs no parsing:

  Whenever a rule's openapi_value is, or contains, a reference to another schema
  (a "$ref"), ALSO fill the mapping's `references` list. Every 3GPP $ref has the
  shape `[file]#/components/schemas/<schema>`, so each entry is {{file, schema_name}}.
  Which of the two a type takes is decided by the REFERENCE CASES below — never by
  habit (a HELD rule in `deferred_rules` also leaves file empty: there it means "not
  settled yet", and the code fills it in):
    - INTERNAL reference (a schema defined in THIS same YAML): file is "" (empty).
      e.g. value "$ref: '#/components/schemas/PerfMetricValue'"
           → references: [{{"file": "", "schema_name": "PerfMetricValue"}}]
    - EXTERNAL reference (a schema from another spec): file is that spec's YAML
      filename exactly as the spec writes it (e.g. "TS28623_ComDefs.yaml").
      e.g. value "$ref: 'TS28623_ComDefs.yaml#/components/schemas/NotificationHeader'"
           → references: [{{"file": "TS28623_ComDefs.yaml", "schema_name": "NotificationHeader"}}]
  Give ONE entry per referenced schema: one for a property/response/allOf that
  points to a single schema, several for a oneOf/anyOf/allOf over multiple schemas
  (list every referenced schema; ignore inline non-ref members like "type: integer").
  Leave `references` empty ([]) when the value has no $ref (a plain type such as
  "string"/"integer", an enum, an HTTP method, a media type).

RULES ABOUT WHAT NOT TO EXTRACT:

  - Do NOT extract rules about the absence of a construct.
    Example: "no query parameters are supported" is NOT a valid rule — absence cannot
    be mapped in OpenAPI.
  - Do NOT reproduce any rule listed as ALREADY VALIDATED or DISCARDED in the
    CORRECTION TASK — those already exist or were rejected on purpose.
  - Do NOT abbreviate URI paths with "..." in any openapi_object field —
    always write the full URI template.

ONE RULE PER ITEM — never combine into a single rule:

  - HTTP methods: create one path_operation rule per method.
    Wrong: one rule for "PUT and PATCH". Right: two rules, one for PUT, one for PATCH.
  - HTTP status codes: create one response rule per code or per range, and write
    every range in UPPERCASE ("4XX", not "4xx").
    Wrong: one rule with openapi_field "4xx/5xx", or "4xx or 5xx", or lowercase "4xx".
    Right: for a section that lists "4xx" and "5xx", emit TWO rules — one openapi_field
    "4XX" and one "5XX" — plus one rule per specific code (200, 201, 400, ...).
    A concatenated cell is NEVER one rule; split it, always, every time.
  - Media types in request_body: create one request_body rule per media type.
    Wrong: one rule for "application/json and application/merge-patch+json".
    Right: two rules, one per media type.

MISSING CONTEXT — report what the spec did not give you, in `missing_context`:

  A section is extracted alone, so a fact it depends on may live in a clause no
  context reached. Always emit the best rule you can — then report what was
  missing. One exception: WHERE A $ref TYPE IS DEFINED is never guessed and never
  reported here — a rule whose $ref type is not settled goes to `deferred_rules`
  (REFERENCE CASES), and a later section settles it. The Reflector reads these, checks whether the spec really is silent,
  and retrieves the clause when it is not, so the next attempt maps it right.
  Reporting is how a wrong mapping gets found: a fact you keep to yourself is one
  nobody can check.

  Report every fact the spec did not supply and you had to work around:
    - you fell back to a shape you know is not the right one (e.g. you emitted a
      path_operation for a notification whose URI is the consumer's address,
      because you could not tell which operation to anchor the callback on);
    - you could not resolve a required part of openapi_object and had to leave it
      generic;
    - you filled a field with a default value from the OpenAPI reference because
      no clause you were given states it (a media type, a JSON Schema form).
      Report it even though the rule came out complete — you cannot see the
      clauses you were not given, and one of them may state it.
  Name the fact and what you did about it, in a sentence.

  Do NOT report: a rule you decided not to extract; a fact the section or the
  contexts above DO state; or a whole topic ("more about callbacks"). At most 2
  entries — the ones that most affect the rules. Empty list only when the spec
  gave you every value you used.

OpenAPI reference (use ONLY to understand valid OpenAPI constructs and field names):
{openapi_reference_overview}
"""

_USER = """\
{contract_summary}

Section ID   : {section_id}
Section Title: {section_title}

Extraction focus (defined by the Planner):
{extraction_focus}

Pre-analysis of this section (a focused triage done BEFORE this extraction —
follow it: it already resolved this section's kind, whether it is a schema home,
any imported base type / inherited fields, and — when the section's notification
goes to a consumer-supplied address — the CALLBACK ANCHOR its callback rules
hang off. These are decisions, not suggestions: apply them as written. Where it
gives an anchor, use that exact string and resolve none of your own):
{section_analysis}

>>>>> BEGIN SECTION CONTENT — THE ONLY TEXT YOU MAY EXTRACT RULES FROM <<<<<
Every rule you return MUST trace to a specific line or row BETWEEN these markers.
Extract ONLY from here. If this is empty or just a heading/marker, return an empty
list.
{section_content}
>>>>> END SECTION CONTENT — DO NOT EXTRACT ANY RULE FROM ANYTHING BELOW THIS LINE <<<<<

Everything below is REFERENCE ONLY. You may READ it to phrase a rule already found
above (a canonical name, the real path, the exact casing), but you must NEVER turn
anything below into a rule of this section. If a code, property, body or path
appears below but NOT in the section content above, it is another section's rule —
never emit it here.

How things relate, from clauses other sections resolved. NEVER a rule source —
you emit only what THIS section states. Use these to fill a target your own text
leaves open, which is what they were kept for. Each kind decides something
different:

  who creates or carries a thing -> the operation to anchor a callback on, or the
    operation that owns a request body. "A consumer creates an X instance" means
    the create operation on the resource holding X is that operation.
  where a thing belongs -> the schema a property goes under, or the section that
    owns a type you should reference rather than redefine.
  what a thing is part of -> whether a URI variable belongs to the base URL (so
    it is neither in the path key nor a path_parameter) or to the resource, and
    which media types an operation accepts.
  what a thing requires -> that a rule here depends on an operation defined
    elsewhere; reference it, do not restate it.

Apply the fact rather than falling back to a generic shape or a placeholder. If
one contradicts this section's own text, the section wins:
{cross_ref_context}

Stage 2 context (abstract definitions of THIS service — reference for the canonical
IS operation name only; NEVER a rule source):
{stage2_context}

Sibling stage 3 context (other sections of THIS service — reference for the real
resource path, exact schema name/casing, and where properties are defined; NEVER a
rule source; never re-extract a rule that belongs to those sibling sections):
{stage3_context}

Candidate rules from a prior analysis of THIS section's content (a HINT, not a
ceiling):
{candidate_rules}
  Use these to stay anchored in the section's own content and to NOT turn the
  reference contexts above into rules the section does not state. They are not a
  closed list: if the section content clearly supports a rule not listed here,
  extract it; if the analysis lists none, extract only what the text genuinely
  states — do not pull rules from the retrieved context.

Rules to SKIP — of the above, already owned by another section (do NOT re-emit
these specific rules; they belong elsewhere):
{skip_rules}
  This list is a precise, per-rule list — NOT a reason to suppress the section. A
  skip entry covers only the rule it names: if it names this operation's path or
  request_body but NOT a given response code, that response is still produced (each
  status code is its own rule). Every candidate not skipped is produced — in `rules`,
  or in `deferred_rules` when its $ref points at a HELD type (REFERENCE CASES below).
  Holding a rule is producing it, not skipping it.

REFERENCE CASES — every $ref to a named type, in any rule type. Where the type is
defined decides the form: INTERNAL for a type of this document, EXTERNAL (with the
YAML file) for a type of another spec, HELD when that is not settled yet. The
reference memory below records what earlier sections settled — read which case each
type is in:

  1. "<Type> -> <spec> [<filename>]" -> EXTERNAL, filename before the '#':
     "$ref: '<filename>.yaml#/components/schemas/<Type>'". Complete.

  2. LOCAL — the memory says "LOCAL", or THIS section's text shows the type is
     defined in this document (a clause number of this document beside it, or this
     section defines it): "$ref: '#/components/schemas/<Type>'", `file` empty.

  3. NOT SETTLED — the memory says "ORIGIN NOT STATED" or "filename not known yet",
     the Section analysis lists the type under HOLD, or the type is not in the
     memory and nothing in this section says where it is defined. Do NOT write it in
     `rules`: the internal form would assert it is local, which nothing states. Put
     the rule in `deferred_rules` instead, written completely, with
     "$ref: '#/components/schemas/<Type>'" and the type in `references`; the code
     completes the target when a section states where the type is defined.

     Types held in this section: {held_types}

     This applies wherever OpenAPI expects a schema — a $ref can stand in any of
     those places, so check every rule you write, not only the obvious ones. EVERY
     rule whose value refers to one of the held types goes to `deferred_rules`,
     never to `rules`, whatever its position, for example: a response body or a
     request body; a parameter or a response header; a schema property; an array
     element (items); a map value (additionalProperties); a member of
     allOf / oneOf / anyOf; a schema that is itself just a $ref.

A type is never taken as local merely because it is missing from the memory —
being unlisted means its origin has not been stated yet, not that it is local.
{ref_memory}

Auxiliary context — supporting 3GPP documents and any instructions the user
supplied for this run. Anything here about how to map something OVERRIDES the
modelling conventions above: it is the decision for this service, and the
conventions are only the default where it is silent. Empty when none was given:
{helper_context}

{correction_task}
"""

extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM + "\n\n" + _RULE_TYPE_DEFINITIONS),
    ("human",  _USER),
])
