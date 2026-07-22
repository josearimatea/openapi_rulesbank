# src/utils/rules_check.py

"""
Deterministic rule_type-specific checks for openapi_mapping fields.

Used by the Validator Node as Stage 1b — after Pydantic structural validation
(Stage 1a) and before LLM semantic validation (Stage 2).

HOW IT WORKS:
    Each rule extracted by the Extractor has a rule_type that determines what
    valid values openapi_object, openapi_field, and openapi_value must have.
    Pydantic (Stage 1a) only verifies that these are non-empty strings — it does
    not know that a 'path_operation' rule must have a single HTTP method as its
    openapi_field, or that a 'response' rule must have an HTTP status code.

    check_mapping_for_type() encodes those constraints as deterministic string
    checks. If any check fails, the Validator adds the rule to validation_errors
    with a precise error message, triggering loop-back feedback to the Extractor.

VALIDATION RULES BY rule_type:

    path_operation:
        openapi_object  must start with "paths."
        openapi_field   must be exactly one of: get | put | post | delete | patch
                        (never two methods combined — each method = one rule)
        openapi_value   must be the uppercase HTTP method: GET|PUT|POST|DELETE|PATCH

    schema_property:
        openapi_object  must contain "components/schemas"
        openapi_field   must start with "properties.<name>" OR be a schema-level
                        keyword: enum | required | oneOf | anyOf | allOf | items |
                        additionalProperties

    path_parameter:
        openapi_field   must contain "in=path"

    query_parameter:
        openapi_field   must contain "in=query"

    response:
        openapi_field   must be a 3-digit status code ("200", "404", etc.), an
                        OpenAPI wildcard ("1XX"–"5XX"), or the catch-all "default"
                        (the OpenAPI keyword for the response covering all undeclared
                        status codes). Rejects combined forms like "4XX/5XX", "4xx",
                        or "n/a".

    request_body:
        openapi_field   must be "content"
        openapi_value   must contain "/" (media type format, e.g. "application/json")

    security_scheme:
        openapi_field   must be "type"
        openapi_value   must be one of: oauth2 | http | apiKey | openIdConnect

Functions:
    check_mapping_for_type(rule_type, mapping) → list[str]
"""

import re

_VALID_HTTP_METHODS   = {"get", "put", "post", "delete", "patch"}
# OpenAPI 3.0.1 Security Scheme types (the "type" field of a securityScheme).
_VALID_SECURITY_TYPES = {"oauth2", "http", "apiKey", "openIdConnect"}
# Schema-level OpenAPI / JSON Schema keywords that constrain a whole schema rather
# than a single named property. A schema_property rule may target one of these in
# addition to a "properties.<name>" path.
_VALID_SCHEMA_KEYWORDS = {
    "enum", "required", "oneOf", "anyOf", "allOf", "items", "additionalProperties",
}
# Accepts 3-digit codes ("200", "404"), OpenAPI wildcards ("1XX"–"5XX"), and the
# OpenAPI "default" keyword (the response for all undeclared status codes).
# Rejects combined forms ("4XX/5XX"), lowercase wildcards ("4xx"), and "n/a".
_VALID_RESPONSE_FIELD = re.compile(r"^(\d{3}|[1-5]XX|default)$")


def check_mapping_for_type(rule_type: str, mapping: dict) -> list[str]:
    """
    Checks that openapi_object, openapi_field, and openapi_value are consistent
    with the given rule_type.

    Args:
        rule_type : the rule_type string from the extracted rule
        mapping   : dict with keys openapi_object, openapi_field, openapi_value

    Returns:
        list[str] — error messages describing what is wrong.
                    Empty list means all checks passed for this rule_type.
                    Unknown rule_types produce no errors (no-op).
    """
    errors: list[str] = []
    obj   = mapping.get("openapi_object", "")
    field = mapping.get("openapi_field",  "")
    value = mapping.get("openapi_value",  "")

    if rule_type == "path_operation":
        if not obj.startswith("paths."):
            errors.append(
                f"openapi_object must start with 'paths.', got '{obj}'."
            )
        if field not in _VALID_HTTP_METHODS:
            errors.append(
                f"openapi_field must be a single HTTP method "
                f"({'/'.join(sorted(_VALID_HTTP_METHODS))}), got '{field}'. "
                "Create one rule per method — never combine two methods in one rule."
            )
        if value.upper() not in {m.upper() for m in _VALID_HTTP_METHODS}:
            errors.append(
                f"openapi_value must be the uppercase HTTP method "
                f"(GET/PUT/POST/DELETE/PATCH), got '{value}'."
            )

    elif rule_type == "schema_property":
        if "components/schemas" not in obj:
            errors.append(
                f"openapi_object must reference 'components/schemas/<Name>', got '{obj}'."
            )
        # The field may be a named property ("properties.<name>") or a schema-level
        # keyword that applies to the whole schema (enum, required, oneOf, anyOf,
        # allOf, items, additionalProperties).
        if not (field.startswith("properties.") or field in _VALID_SCHEMA_KEYWORDS):
            errors.append(
                f"openapi_field must be 'properties.<name>' or one of "
                f"{sorted(_VALID_SCHEMA_KEYWORDS)}, got '{field}'."
            )

    elif rule_type in ("path_parameter", "query_parameter"):
        param_in = "path" if rule_type == "path_parameter" else "query"
        if f"in={param_in}" not in field:
            errors.append(
                f"openapi_field must contain 'in={param_in}', got '{field}'."
            )

    elif rule_type == "response":
        if not _VALID_RESPONSE_FIELD.match(field):
            errors.append(
                f"openapi_field must be an HTTP status code string ('200', '404', etc.), "
                f"an OpenAPI wildcard ('1XX'–'5XX'), or 'default', got '{field}'."
            )

    elif rule_type == "request_body":
        if field != "content":
            errors.append(
                f"openapi_field must be 'content', got '{field}'."
            )
        if "/" not in value:
            errors.append(
                f"openapi_value must be a media type (e.g. 'application/json'), "
                f"got '{value}'."
            )

    elif rule_type == "security_scheme":
        if field != "type":
            errors.append(
                f"openapi_field must be 'type', got '{field}'."
            )
        if value not in _VALID_SECURITY_TYPES:
            errors.append(
                f"openapi_value must be one of {sorted(_VALID_SECURITY_TYPES)}, "
                f"got '{value}'."
            )

    elif rule_type == "callback":
        # A callback hangs off the operation where the consumer supplies its
        # address: paths.<path>.<method>.callbacks.<notificationName>
        if not obj.startswith("paths.") or ".callbacks." not in obj:
            errors.append(
                "openapi_object must anchor the callback on an operation, as "
                "'paths.<path>.<method>.callbacks.<name>', got "
                f"'{obj}'. A notification is never a path of this service."
            )
        if field not in _VALID_HTTP_METHODS:
            errors.append(
                f"openapi_field must be the delivery HTTP method "
                f"({'/'.join(sorted(_VALID_HTTP_METHODS))}), got '{field}'."
            )
        # The value is the runtime expression pointing at the consumer's address.
        if not (value.startswith("{") and value.endswith("}")):
            errors.append(
                "openapi_value must be the runtime expression for the consumer "
                "address as written in the spec (e.g. '{request.body#/<field>}'), "
                f"got '{value}'."
            )

    return errors
