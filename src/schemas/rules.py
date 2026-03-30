# src/schemas/rules.py

"""
Pydantic data models for the rules extracted by the pipeline.

These models define the structure of a single OpenAPI rule at each stage
of the pipeline. The same rule evolves as it passes through nodes:
    Extractor  → produces RawRule
    Reflector  → enriches to ReflectedRule (adds confidence, reasoning, flagged)
    Validator  → promotes to ValidatedRule or ValidationError

All models inherit from RawRule so fields are consistent across stages.
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

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 and 1. "
                    "1.0 = fully grounded in the spec, 0.0 = uncertain or hallucinated."
    )
    reasoning: str = Field(
        description="Chain-of-Thought explanation of why the rule was extracted "
                    "and how confident the model is in its correctness."
    )
    flagged: bool = Field(
        description="True if this rule should receive priority validation, "
                    "e.g. low confidence, ambiguous mapping, or conflicting signals."
    )
    rag_context: str = Field(
        default="",
        description="Relevant context retrieved from Qdrant (3GPP + OpenAPI spec) "
                    "used to ground the self-reflection."
    )


class ValidatedRule(ReflectedRule):
    """
    A rule that passed both structural and semantic validation.

    Produced by: Validator Node
    Consumed by: Builder Node
    """

    validation_notes: str = Field(
        default="",
        description="Optional notes from the validator, e.g. minor corrections "
                    "applied during semantic validation."
    )


class ValidationError(BaseModel):
    """
    A rule that failed validation, with the failure reason attached.
    Stored in state['validation_errors'] for loop-back feedback to the Extractor.

    Produced by: Validator Node
    Consumed by: conditions.should_loop_or_build() and Extractor (on loop-back)
    """

    rule: RawRule = Field(
        description="The original rule that failed validation."
    )
    reason: str = Field(
        description="Description of why the rule failed validation. "
                    "Passed back to the Extractor as feedback on loop-back."
    )
    stage: str = Field(
        description="Which validation stage caught the error: 'structural' or 'semantic'."
    )
