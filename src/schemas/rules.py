# src/schemas/rules.py

"""
Pydantic data models for the rules extracted by the pipeline.

The same rule evolves as it passes through nodes:

    Extractor  → RawRule
    Reflector  → ReflectedRule    (+ confidence, reasoning, flagged, rag_context)
    Validator  → ValidatedRule    (+ validation_notes)
                 ValidationError  (rules that failed, stored for loop-back)

MODEL HIERARCHY
───────────────
  RawRule
    ├── section_id      str   — section_id from parsed_sections
    ├── section_title   str   — section title, for traceability
    ├── rule_type       str   — path_operation | schema_property | path_parameter |
    │                           query_parameter | response | request_body | security_scheme
    ├── source_name     str   — name of the specific source element this rule covers:
    │                           path_operation  → IS operation name  (e.g. "createMOI")
    │                           schema_property → NRM attribute name (e.g. "nrCellDuId")
    │                           path_parameter  → path param name    (e.g. "MnSVersion")
    │                           query_parameter → query param name   (e.g. "scope")
    │                           response        → IS operation name  (e.g. "getMOIAttributes")
    │                           request_body    → IS operation name  (e.g. "createMOI")
    │                           security_scheme → scheme name        (e.g. "OAuth2")
    ├── rule_text       str   — full rule statement extracted from the 3GPP document
    └── openapi_mapping OpenAPIMapping
          ├── openapi_object  str   — OpenAPI object/schema (e.g. "NrCellDu", "paths./nrCellDu/{id}")
          ├── openapi_field   str   — field or keyword     (e.g. "type", "format", "get")
          └── openapi_value   str   — value or constraint  (e.g. "string", "int64", "200")

  ReflectedRule(RawRule)
    ├── reflection_confidence   float — 0.0–1.0; how grounded the rule is in the spec
    ├── reflection_reasoning    str   — CoT explanation from the Reflector
    ├── reflection_flagged      bool  — True if the rule should receive priority validation
    ├── reflection_rag_context  str   — Qdrant chunks used to ground the self-reflection
    ├── split_suggestion        str   — suggested split description; empty if none
    └── discard_suggestion      bool  — True if Reflector believes the rule should not exist

  ValidatedRule(ReflectedRule)
    ├── validation_notes   str   — optional corrections noted during semantic validation
    └── validation_passed  bool  — True if Validator approved; False if force-included by Builder

  ValidationError  (standalone, not in the inheritance chain)
    ├── error_type  str   — 'correction' | 'split' | 'discard'
    ├── stage       str   — 'structural' | 'semantic'
    ├── section_id  str   — section this error belongs to
    ├── rule        dict  — original rule dict that failed
    └── instruction str   — actionable command for the Extractor

  SectionFeedback  (standalone, not in the inheritance chain)
    ├── section_id    str        — section that needs additional rules extracted
    ├── missing_rules list[str]  — actionable instructions for rules to extract
    └── reasoning     str        — why these rules are missing (default: "")
"""

from pydantic import BaseModel, Field


class OpenAPIMapping(BaseModel):
    """
    Represents the mapping between a 3GPP NRM element and an OpenAPI construct.

    Example:
        A 3GPP attribute "nrCellDuId" maps to an OpenAPI schema property
        of type string under the NrCellDu schema object.
    """

    openapi_object: str = Field(
        description="The OpenAPI object or schema this rule applies to. "
                    "E.g. 'NrCellDu', 'paths./nrCellDu/{id}', 'components/schemas'."
    )
    openapi_field: str = Field(
        description="The specific OpenAPI field, property, or keyword. "
                    "E.g. 'type', 'format', 'required', 'operationId', '$ref'."
    )
    openapi_value: str = Field(
        description="The value or constraint that applies to the field. "
                    "E.g. 'string', 'int64', 'GET', '#/components/schemas/NrCellDu'."
    )


class RawRule(BaseModel):
    """
    A rule as extracted by the Extractor Node — not yet reflected or validated.

    Produced by: Extractor Node
    Consumed by: Reflector Node
    """

    section_id: str = Field(
        description="The section_id from parsed_sections where this rule was found."
    )
    section_title: str = Field(
        description="The section title, for traceability and reporting."
    )
    rule_type: str = Field(
        description=(
            "Category of OpenAPI construct this rule describes. "
            "Must be exactly one of: "
            "path_operation | schema_property | path_parameter | query_parameter | "
            "response | request_body | security_scheme"
        )
    )
    source_name: str = Field(
        description=(
            "Name of the specific source element this rule covers. "
            "Depends on rule_type:\n"
            "  path_operation  → IS operation name  (e.g. 'createMOI', 'modifyMOIAttributes')\n"
            "  schema_property → NRM attribute name (e.g. 'nrCellDuId', 'cellLocalId')\n"
            "  path_parameter  → path param name    (e.g. 'MnSVersion', 'id')\n"
            "  query_parameter → query param name   (e.g. 'scope', 'filter')\n"
            "  response        → IS operation name  (e.g. 'getMOIAttributes')\n"
            "  request_body    → IS operation name  (e.g. 'createMOI')\n"
            "  security_scheme → scheme name        (e.g. 'OAuth2', 'BearerAuth')"
        )
    )
    rule_text: str = Field(
        description="The full rule statement as extracted from the 3GPP document. "
                    "Should be a clear, actionable statement."
    )
    openapi_mapping: OpenAPIMapping = Field(
        description="The OpenAPI construct this rule maps to."
    )


class ReflectedRule(RawRule):
    """
    A rule after Chain-of-Thought self-reflection by the Reflector Node.
    Extends RawRule with confidence score, reasoning trace, and a flag.

    Produced by: Reflector Node
    Consumed by: Validator Node
    """

    reflection_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 and 1 assigned by the Reflector. "
                    "1.0 = fully grounded in the spec, 0.0 = uncertain or hallucinated."
    )
    reflection_reasoning: str = Field(
        description="Chain-of-Thought explanation produced by the Reflector: why the "
                    "rule was extracted and how confident the model is in its correctness."
    )
    reflection_flagged: bool = Field(
        description="Set by the Reflector. True if this rule should receive priority "
                    "validation, e.g. low confidence, ambiguous mapping, or conflicting signals."
    )
    reflection_rag_context: str = Field(
        default="",
        description="Relevant context retrieved from Qdrant (OpenAPI spec) "
                    "used by the Reflector to ground the self-reflection."
    )
    split_suggestion: str = Field(
        default="",
        description=(
            "Reflector suggestion that this rule should be split into multiple rules. "
            "Empty string if no split is needed. "
            "Example: 'Split into two rules: one for 400 and one for 500 response codes.'"
        )
    )
    discard_suggestion: bool = Field(
        default=False,
        description=(
            "True if the Reflector believes this rule should not exist — "
            "e.g. it maps an absence of a construct, which is not a valid OpenAPI rule."
        )
    )


class ValidatedRule(ReflectedRule):
    """
    A rule that passed both structural and semantic validation.

    Produced by: Validator Node
    Consumed by: Builder Node
    """

    validation_notes: str = Field(
        default="",
        description="Optional notes from the Validator, e.g. minor corrections "
                    "applied during semantic validation."
    )
    validation_passed: bool = Field(
        default=True,
        description="True if the rule was approved by the Validator Node. "
                    "False if it was force-included by the Builder at MAX_ITERATIONS "
                    "after failing validation in every loop-back iteration."
    )


class ValidationError(BaseModel):
    """
    Feedback from the Validator to the Extractor for a single rule that failed.

    error_type drives what the Extractor must do:
      'correction' — fix the rule's mapping or rule_text
      'split'      — break this rule into multiple more specific rules
      'discard'    — remove this rule entirely (not a valid OpenAPI construct)

    Produced by: Validator Node
    Consumed by: conditions.should_loop_or_build() and Extractor (on loop-back)
    """

    error_type: str = Field(
        description="'correction' | 'split' | 'discard'"
    )
    stage: str = Field(
        description="'structural' | 'semantic'"
    )
    section_id: str = Field(
        description="section_id this error belongs to. Always populated."
    )
    rule: dict = Field(
        description=(
            "The original rule dict that failed validation. "
            "Stored as dict so structurally invalid rules can also be captured."
        )
    )
    instruction: str = Field(
        description=(
            "Actionable instruction for the Extractor. Always written as a direct command. "
            "Examples:\n"
            "  correction: 'Fix openapi_field to use format parameters[in=query,name=scope]'\n"
            "  split: 'Split into three rules: one for 400, one for 404, one for 500'\n"
            "  discard: 'Discard — absence of query parameters is not a valid OpenAPI rule'"
        )
    )


class SectionFeedback(BaseModel):
    """
    Per-section completeness feedback produced by the Reflector (Fase 2).
    Instructs the Extractor to extract rules that are missing from a section.
    Not tied to any specific rule.

    Produced by: Reflector Node (Fase 2 — section-level completeness check)
    Consumed by: Validator Node (decides whether to forward to Extractor)
    Forwarded by: Validator Node → section_feedback state field
    """

    section_id: str = Field(
        description="The section that needs additional rules extracted."
    )
    missing_rules: list[str] = Field(
        description=(
            "Actionable instructions for rules to extract. "
            "Each entry is a direct command to the Extractor. "
            "Example: 'Extract rule for HTTP 201 Created response for PUT operation'"
        )
    )
    reasoning: str = Field(
        default="",
        description="Why these rules are missing based on the section content."
    )
