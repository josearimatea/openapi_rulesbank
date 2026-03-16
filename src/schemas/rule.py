# src/schemas/rule.py
# Pydantic model representing a single extracted rule.
#
# Fields (planned):
#   - section: str          — source section in the 3GPP document
#   - rule_text: str        — the extracted rule statement
#   - openapi_mapping: dict — mapping to OpenAPI schema constructs
#   - confidence: float     — confidence score assigned by Reflector
#   - flagged: bool         — whether rule requires priority validation
#
# TODO: implement Rule Pydantic model
