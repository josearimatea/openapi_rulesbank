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
Each rule must have a rule_type, a clear rule_text, and a precise openapi_mapping.

SOURCE OF RULES:
  - Extract rules ONLY from the "Section content" in the user message.
  - Do NOT extract rules from the OpenAPI reference below — it is reference material only.
  - Do NOT infer or hallucinate rules not explicitly stated or clearly implied by the text.
  - If a section contains a table, extract one rule per relevant row.
  - If a section defines multiple attributes, extract one rule per attribute.
  - Ignore introductory or background text with no concrete mappable content.

RULE TYPES — for each rule, assign exactly one rule_type and fill openapi_mapping accordingly:

1. path_operation  — one rule per HTTP method on a resource path
   rule_type      : "path_operation"
   source_name    : the IS operation name (e.g. "createMOI", "modifyMOIAttributes")
   openapi_object : the path template as written in the spec, prefixed with "paths."
                    e.g. "paths./ProvMnS/{{MnSVersion}}/{{className}}/{{id}}"
   openapi_field  : exactly one HTTP method in lowercase: get | put | post | delete | patch
                    NEVER combine two methods (e.g. never "put, patch") — create two rules.
   openapi_value  : the HTTP method in uppercase: GET | PUT | POST | DELETE | PATCH

2. schema_property — one rule per attribute or field in a data model
   rule_type      : "schema_property"
   source_name    : the NRM attribute name (e.g. "nrCellDuId", "cellLocalId")
   openapi_object : the schema path, e.g. "components/schemas/NrCellDu"
   openapi_field  : "properties.<propertyName>"  e.g. "properties.nrCellDuId"
   openapi_value  : the JSON Schema type: "string" | "integer" | "boolean" | "array" |
                    "object" | "$ref: '#/components/schemas/<Name>'"

3. path_parameter  — one rule per path variable in a URI template
   rule_type      : "path_parameter"
   source_name    : the path parameter name (e.g. "MnSVersion", "id")
   openapi_object : the path template, e.g. "paths./ProvMnS/{{MnSVersion}}/{{id}}"
   openapi_field  : "parameters[in=path,name=<paramName>]"
   openapi_value  : the parameter schema type: "string" | "integer" | etc.

4. query_parameter — one rule per query string parameter
   rule_type      : "query_parameter"
   source_name    : the query parameter name (e.g. "scope", "filter")
   openapi_object : the path template, e.g. "paths./ProvMnS/{{MnSVersion}}/{{id}}"
   openapi_field  : "parameters[in=query,name=<paramName>]"
   openapi_value  : the parameter schema type: "string" | "integer" | etc.

5. response        — one rule per HTTP response code for an operation
   rule_type      : "response"
   source_name    : the IS operation name (e.g. "getMOIAttributes", "createMOI")
   openapi_object : "paths.<path>.<method>.responses"
   openapi_field  : the HTTP status code as string: "200" | "201" | "204" | "400" | "404" | "500"
   openapi_value  : "$ref: '#/components/responses/<Name>'" or a brief schema description

6. request_body    — one rule per operation that accepts a request body
   rule_type      : "request_body"
   source_name    : the IS operation name (e.g. "createMOI", "modifyMOIAttributes")
   openapi_object : "paths.<path>.<method>.requestBody"
   openapi_field  : "content"
   openapi_value  : the media type: "application/json" | "application/merge-patch+json" | etc.

7. security_scheme — one rule per security/authentication requirement
   rule_type      : "security_scheme"
   source_name    : the security scheme name (e.g. "OAuth2", "BearerAuth")
   openapi_object : "components/securitySchemes/<SchemeName>"
   openapi_field  : "type"
   openapi_value  : "oauth2" | "http" | "apiKey" | "openIdConnect"

RULES ABOUT WHAT NOT TO EXTRACT:

  - Do NOT extract rules about the absence of a construct.
    Example: "no query parameters are supported" is NOT a valid rule — absence cannot
    be mapped in OpenAPI.
  - Do NOT combine multiple HTTP status codes in a single rule.
    Wrong: one rule for "4xx/5xx responses".
    Right: one rule per code (400, 404, 500, etc.) or one wildcard rule (4XX, 5XX).
  - Do NOT duplicate rules already validated in previous iterations.

OpenAPI reference (use ONLY to understand valid OpenAPI constructs and field names):
{openapi_reference_overview}
"""

_USER = """\
Section ID   : {section_id}
Section Title: {section_title}

Extraction focus (defined by the Planner):
{extraction_focus}

Section content (the ONLY source from which rules must be extracted):
{section_content}

Auxiliary context from supporting 3GPP documents (if relevant):
{helper_context}

{correction_task}
"""

extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])
