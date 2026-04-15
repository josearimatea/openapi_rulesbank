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
        openapi_field   must start with "properties."

    path_parameter:
        openapi_field   must contain "in=path"

    query_parameter:
        openapi_field   must contain "in=query"

    response:
        openapi_field   must be a 3-digit status code ("200", "404", etc.) or an
                        OpenAPI wildcard ("1XX"–"5XX"). Rejects combined forms like
                        "4XX/5XX", "4xx", or "n/a".

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
_VALID_SECURITY_TYPES = {"oauth2", "http", "apiKey", "openIdConnect"}
# Accepts 3-digit codes ("200", "404") and OpenAPI wildcards ("1XX"–"5XX").
# Rejects combined forms ("4XX/5XX"), lowercase wildcards ("4xx"), and "n/a".
_VALID_RESPONSE_FIELD = re.compile(r"^(\d{3}|[1-5]XX)$")


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
        if not field.startswith("properties."):
            errors.append(
                f"openapi_field must start with 'properties.<name>', got '{field}'."
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
                f"openapi_field must be an HTTP status code string ('200', '404', etc.) "
                f"or an OpenAPI wildcard ('1XX'–'5XX'), got '{field}'."
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

    return errors
