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

SOURCE OF RULES:
  - Extract rules ONLY from the "Section content" in the user message.
  - Do NOT extract rules from the OpenAPI reference below — it is reference material only.
  - Do NOT infer or hallucinate rules not explicitly stated or clearly implied by the text.
  - If a section contains a table, extract one rule per relevant row.
  - If a section defines multiple attributes, extract one rule per attribute.
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
                    HEADING of the section that defines it — headings of the form
                    "<clause> Type <Name>" or "<clause> Enumeration <Name>" — copied
                    with the heading's exact spelling and casing.
                    Do NOT use the name of a property, of a notification, or of a table
                    column as the schema name, and never append a suffix the spec does
                    not use (no "Request", "RequestBody", "Response").
                    Example of the mistake to avoid: in a section titled
                    "Enumeration CmNotificationTypes" the schema is CmNotificationTypes,
                    not the property that happens to carry it.
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

Section content (the ONLY source from which rules must be extracted):
{section_content}

Related spec context (retrieved from the source 3GPP document — use for the
canonical IS operation name and related definitions, NOT as a rule source):
{spec_context}

Auxiliary context from supporting 3GPP documents (if relevant):
{helper_context}

{correction_task}
"""

extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])
