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
                        (openapi_object not checked for a method: a path param
                        may sit at the path level or inside one operation)

    query_parameter:
        openapi_field   must contain "in=query"
        openapi_object  must end in ".<method>" (paths.<path>.<method>) — a
                        query param belongs to one operation; "/get" is rejected,
                        only the canonical ".get" is accepted

    callback:
        openapi_object  must be "paths.<path>.<method>.callbacks.<name>"
        openapi_field   must be the delivery HTTP method (get|put|post|delete|patch)
        openapi_value   must be the runtime expression in braces (e.g. "{...}")

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
# Rejects combined forms ("4XX/5XX") and "n/a". A lowercase wildcard ("4xx") is
# not rejected but normalized to uppercase first (see below) — only the case of
# the "XX" is wrong, the code is valid OpenAPI once upper-cased.
_VALID_RESPONSE_FIELD = re.compile(r"^(\d{3}|[1-5]XX|default)$")
# Matches a bare wildcard in either case ("4xx", "4XX") so the field can be
# normalized to the canonical uppercase form before validation. Deliberately does
# NOT match combined forms like "4xx/5xx" — those are a real fusion error the
# Extractor must split, so they still fail _VALID_RESPONSE_FIELD.
_WILDCARD_RESPONSE_FIELD = re.compile(r"^[1-5]xx$", re.IGNORECASE)


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
        # Must name a schema directly: "components/schemas/<Name>". A nested path
        # (".../properties/x/additionalProperties") means the schema was misidentified —
        # the constrained element belongs in openapi_field, not in the object.
        if not obj.startswith("components/schemas/"):
            errors.append(
                f"openapi_object must reference 'components/schemas/<Name>', got '{obj}'."
            )
        elif "/" in obj[len("components/schemas/"):]:
            errors.append(
                f"openapi_object must name the schema itself, not a path inside it: "
                f"got '{obj}'. Use 'components/schemas/<Name>' and put the property or "
                "keyword in openapi_field."
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
        # A query parameter belongs to ONE operation, so its openapi_object must
        # carry the method ("paths.<path>.<method>"); without it the generator
        # routes the parameter to the path level and spreads it over every method.
        # ANY method is accepted — which one is the text's call (a query param can
        # sit under GET, DELETE, etc.; StreamingDataMnS has one under DELETE). This
        # only checks the method is PRESENT, not which. The separator must be a
        # dot (".get", the canonical form) — "/get" is rejected, even though the
        # generator tolerates it, so the bank only ever emits the canonical shape.
        # A path_parameter is NOT checked for a method: it may sit at the path
        # level (shared by every method, object stops at the path) OR inside one
        # operation (object ends in the method). Both are valid OpenAPI, so its
        # object is accepted either way — only the query param must carry a method.
        if rule_type == "query_parameter":
            methods = "|".join(sorted(_VALID_HTTP_METHODS))
            if not re.search(rf"\.(?:{methods})$", obj):
                errors.append(
                    "openapi_object must end with the method that accepts this "
                    f"parameter, as 'paths.<path>.<method>', got '{obj}'."
                )

    elif rule_type == "response":
        # The object must carry the operation: a method before ".responses".
        # Without it the generator cannot find the operation and DROPS the rule
        # silently (unlike a parameter, which it routes to the path level). The
        # method is the operation's own ("paths./x.put.responses") or, for a
        # callback's response, its delivery method
        # ("paths./x.put.callbacks.<name>.post.responses") — ".+" spans either.
        methods = "|".join(sorted(_VALID_HTTP_METHODS))
        if not re.search(rf"^paths\..+\.(?:{methods})\.responses$", obj):
            errors.append(
                "openapi_object must be the full operation path ending in "
                f"'.responses', as 'paths.<path>.<method>.responses', got '{obj}'."
            )
        # Normalize a lowercase wildcard ("4xx" -> "4XX") in place before checking,
        # so a rule that is valid apart from its casing is fixed, not rejected.
        # Combined forms ("4xx/5xx") do not match here and still fail below.
        if _WILDCARD_RESPONSE_FIELD.match(field):
            field = field.upper()
            mapping["openapi_field"] = field
        if not _VALID_RESPONSE_FIELD.match(field):
            errors.append(
                f"openapi_field must be an HTTP status code string ('200', '404', etc.), "
                f"an OpenAPI wildcard ('1XX'–'5XX'), or 'default', got '{field}'."
            )

    elif rule_type == "request_body":
        # Same as response: the object must carry the operation (a method before
        # ".requestBody"), direct ("paths./x.put.requestBody") or a callback's
        # ("paths./x.put.callbacks.<name>.post.requestBody"). Without a method the
        # generator cannot find the operation and drops the rule silently.
        methods = "|".join(sorted(_VALID_HTTP_METHODS))
        if not re.search(rf"^paths\..+\.(?:{methods})\.requestBody$", obj):
            errors.append(
                "openapi_object must be the full operation path ending in "
                f"'.requestBody', as 'paths.<path>.<method>.requestBody', got '{obj}'."
            )
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
        # address: paths.<path>.<method>.callbacks.<notificationName>. Two
        # methods appear in that structure and they are not the same: the one
        # BEFORE .callbacks. is the registering operation (put in ProvMnS, post
        # in FaultMnS/FileDataReporting — verified against the YAMLs, so it must
        # not be guessed), and openapi_field below is the delivery method (post).
        # Require the registering method to be present in the object, so the
        # generator never has to infer it from the delivery field.
        methods = "|".join(sorted(_VALID_HTTP_METHODS))
        if not re.search(rf"^paths\..+\.(?:{methods})\.callbacks\.[^.]+", obj):
            errors.append(
                "openapi_object must anchor the callback on an operation, as "
                "'paths.<path>.<method>.callbacks.<name>' (the method is the "
                f"registering operation, not the delivery method), got '{obj}'. "
                "A notification is never a path of this service."
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
