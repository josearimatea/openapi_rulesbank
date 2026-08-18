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
    "notifyThresholdCrossing  POST  /notificationSink") gives the real URI in its
    own "Resource URI" column; when the row points elsewhere, the sibling stage 3
    context holds the resource section that defines that URI. Use that URI as the
    path. "paths./notifyThresholdCrossing" (a path named after the notification)
    is always wrong — the path is "/notificationSink". (And when the URI is the
    consumer's address, it is a callback, not a path — see below.)

  - SCHEMA NAME AND CASING — a data type has ONE canonical name: the one in the
    heading "<clause> Type <Name>" / "Enumeration <Name>", found in this section or
    in the sibling stage 3 context. Always use that exact spelling and casing
    (e.g. "NotifyThresholdCrossing", never "notifyThresholdCrossing"). Do not coin
    a schema name from a notification or property mentioned in prose.

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

RULE TYPES — for each rule, assign exactly one rule_type and fill openapi_mapping accordingly:

1. path_operation  — one rule per HTTP method on a resource path
   rule_type      : "path_operation"
   source_name    : the IS operation name (e.g. "createMOI", "modifyMOIAttributes")
   openapi_object : the COMPLETE path template as written in the spec, prefixed with "paths."
                    NEVER abbreviate the path with "..." — always write the full URI template.
                    Wrong: "paths./.../{{className}}={{id}}"
                    Right: "paths./{{MnSVersion}}/{{className}}={{id}}"
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
   openapi_value  : depends on openapi_field:
                    - for "properties.<name>": the JSON Schema type or reference:
                      "string" | "integer" | "boolean" | "array" | "object" |
                      "$ref: '#/components/schemas/<Name>'" |
                      "$ref: '<OtherFile>.yaml#/components/schemas/<Name>'"
                    - for "enum"      : the allowed values, e.g. "[add, remove, replace]"
                    - for "required"  : the required property name(s), e.g. "[id]"
                    - for "oneOf"/"anyOf"/"allOf": the list of referenced schemas, e.g.
                      "[$ref: '#/components/schemas/A', $ref: '#/components/schemas/B']"
                    - for "items"     : the item type or "$ref: '#/components/schemas/<Name>'"
                    - for "additionalProperties": the value type or "true"/"false"

3. path_parameter  — one rule per path variable in a URI template
   rule_type      : "path_parameter"
   source_name    : the path parameter name (e.g. "MnSVersion", "id", "className")
   openapi_object : the COMPLETE path template, e.g. "paths./{{MnSVersion}}/{{className}}={{id}}"
                    NEVER abbreviate with "..." — always write the full URI template.
   openapi_field  : "parameters[in=path,name=<paramName>]"
   openapi_value  : the parameter schema type or reference:
                    "string" | "integer" | "boolean" | "array" |
                    "$ref: '#/components/schemas/<Name>'"

4. query_parameter — one rule per query string parameter
   rule_type      : "query_parameter"
   source_name    : the query parameter name (e.g. "scope", "filter", "attributes")
   openapi_object : the COMPLETE path template, e.g. "paths./{{MnSVersion}}/{{className}}={{id}}"
                    NEVER abbreviate with "..." — always write the full URI template.
   openapi_field  : "parameters[in=query,name=<paramName>]"
   openapi_value  : the parameter schema type or reference:
                    "string" | "integer" | "boolean" | "array" |
                    "$ref: '#/components/schemas/<Name>'"

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
   openapi_value  : "$ref: '#/components/responses/<Name>'" or
                    "$ref: '#/components/schemas/<Name>'" or
                    "$ref: '<OtherFile>.yaml#/components/schemas/<Name>'" or
                    a brief schema description

6. request_body    — one rule per operation that accepts a request body
   rule_type      : "request_body"
   source_name    : the IS operation name (e.g. "createMOI", "modifyMOIAttributes")
   openapi_object : the COMPLETE path and method, e.g.
                    "paths./{{className}}={{id}}.put.requestBody"
                    NEVER abbreviate with "..." — always write the full URI template.
   openapi_field  : "content"
   openapi_value  : the media type as defined in the section:
                    "application/json" | "application/merge-patch+json" |
                    "application/yang-data+json" | etc.
                    If multiple media types are defined, create one rule per media type.

7. security_scheme — one rule per security or authentication requirement
   rule_type      : "security_scheme"
   source_name    : the security scheme name (e.g. "OAuth2", "BearerAuth")
   openapi_object : "components/securitySchemes/<SchemeName>"
   openapi_field  : "type"
   openapi_value  : "oauth2" | "http" | "apiKey" | "openIdConnect"

8. callback        — one rule per notification the PRODUCER sends to the consumer
   rule_type      : "callback"
   source_name    : the notification name (e.g. "notifyMOICreation")
   openapi_object : the full anchor, "paths.<path>.<method>.callbacks.<notificationName>"
                    where <path>.<method> is the operation in which the consumer
                    supplies its address (see NOTIFICATIONS ARE CALLBACKS below).
   openapi_field  : the HTTP method the producer uses to deliver it, lowercase
                    (normally "post")
   openapi_value  : the runtime expression naming the consumer address field, as
                    written in the spec, e.g.
                    "{{request.body#/notificationRecipientAddress}}" or
                    "{{request.body#/consumerReference}}"

NOTIFICATIONS ARE CALLBACKS, NEVER PATHS:

  A `path` is an endpoint THIS service exposes and the consumer calls.
  A `callback` is an endpoint the CONSUMER exposes and this service calls back.

  Decide by asking who owns the URI. Spec tables map notifications to an HTTP
  method and a "Resource URI" that is the address the CONSUMER supplied — a
  placeholder standing for that address rather than a resource of this service.
  Whenever the URI is the consumer's address, emit a `callback` rule, never a
  `path_operation`, and never invent a path named after the notification.
  (Illustrative only — the actual names differ per spec: a URI written as
  "{{notificationTarget}}" or "{{consumerReference}}" is a consumer address, so
  "paths./notifyXyz" or "paths./{{notificationTarget}}" would be wrong.)

  This applies to EVERY section describing resources on the consumer side, not
  only to notification tables. A section may present a "resources and methods
  overview" table identical in shape to the producer's one while its heading or
  text places those resources on the consumer (e.g. "... on the MnS consumer").
  Resources hosted by the consumer are never paths of this service's YAML:
  express what the producer sends there as a `callback`, and emit no
  `path_operation` for them.

  Anchor the callback on the operation where the consumer supplies that address —
  the subscription operation when the service has one, otherwise the operation
  whose request body carries the address field. Read the anchor operation, the
  address field name and the delivery method from THIS spec's text; never assume
  the values seen in other specifications.

  The callback's payload and responses are separate rules whose openapi_object
  continues the same anchor, so the structure matches the target YAML:
    request_body : object "paths.<path>.<method>.callbacks.<name>.<cbMethod>.requestBody",
                   field "content", value the media type
    response     : object "paths.<path>.<method>.callbacks.<name>.<cbMethod>.responses",
                   field the status code, value the schema $ref or description

REFERENCES — structure every schema reference so the YAML builder needs no parsing:

  Whenever a rule's openapi_value is, or contains, a reference to another schema
  (a "$ref"), ALSO fill the mapping's `references` list. Every 3GPP $ref has the
  shape `[file]#/components/schemas/<schema>`, so each entry is {{file, schema_name}}:
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
  - HTTP status codes: create one response rule per code or per range.
    Wrong: one rule for "4xx/5xx responses".
    Right: one rule per specific code (200, 201, 400, etc.) or one rule per
    explicit range (4XX, 5XX), exactly as defined in the section.
  - Media types in request_body: create one request_body rule per media type.
    Wrong: one rule for "application/json and application/merge-patch+json".
    Right: two rules, one per media type.

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
and any imported base type / inherited fields):
{section_analysis}

Section content (the ONLY source from which rules must be extracted — every rule
you return must trace to a line or row BELOW; if this content is empty or is just
a heading/marker, return an empty list, do NOT draw rules from the contexts):
{section_content}

Stage 2 context (abstract definitions of THIS service, retrieved from the source
3GPP document — use ONLY for the canonical IS operation name, NOT as a rule source):
{stage2_context}

Sibling stage 3 context (other HTTP-solution-set sections of THIS SAME service,
retrieved from the source document — use to resolve the real resource path, the
exact schema name and casing, and which properties are defined where. NOT a rule
source, and do NOT re-extract rules that clearly belong to those sibling sections):
{stage3_context}

Rules to SKIP — already owned by another section (decided by the pre-analysis;
do NOT emit these specific rules, they belong elsewhere):
{skip_rules}
  This is the ONLY reason to skip a rule. It is a precise, per-rule list — NOT a
  reason to suppress the section. Emit every OTHER rule this section defines. If a
  skip entry names this operation's path or request_body but NOT its responses,
  you MUST still emit the responses (204, 4XX, 5XX): they are this section's to
  give and are not in the skip list.

Auxiliary context from supporting 3GPP documents (if relevant):
{helper_context}

{correction_task}
"""

extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])
