# src/prompts/validator_prompts.py

"""
Prompt templates and output schemas for the Validator Node.

The Validator gives the final word to the Extractor. Structural validation
(Pydantic + mapping consistency) runs first in code without LLM. Semantic
validation runs in three ordered LLM stages:

  Stage 1 — Initial section analysis (1 call per section, before per-rule):
    Input: current_rules + already_validated_rules + previous_missing_rules.
    Filters previous_missing_rules: removes what is now covered, keeps what
    is still absent. Does NOT add new rules yet.
    Output: SectionAnalysisResult (updated missing_rules).

  Stage 2 — Per-rule verdict (1 call per rule):
    Input: the rule + already_validated_rules + current_missing_rules (from Stage 1,
    growing as each rule is processed).
    Assesses whether the rule is correct and whether new missing rules arise from
    assessing this specific rule.
    Output: ValidationVerdict (valid/error_type/instruction + new_missing_rules).

  Stage 3 — Reflector review (1 call per section, after all per-rule verdicts):
    Input: Reflector section_reflection + validated rules + missing_rules from Stages 1+2.
    Finalises missing_rules: confirms additions from the Reflector, removes anything
    now covered.
    Output: ReflectorReviewResult (final missing_rules).

Classes:
    SectionAnalysisResult         — Stage 1 output
    ValidationVerdict             — Stage 2 output (per rule)
    ReflectorReviewResult         — Stage 3 output

Constants:
    validator_section_init_prompt      — Stage 1 prompt
    validator_prompt                   — Stage 2 prompt (per rule)
    validator_reflector_review_prompt  — Stage 3 prompt
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage 1 — Initial section analysis
# ---------------------------------------------------------------------------

class SectionAnalysisResult(BaseModel):
    """
    Updated missing_rules produced before per-rule validation begins.
    Filters previous_missing_rules against what is already covered.
    """

    missing_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Rules from previous_missing_rules that are still genuinely absent. "
            "Remove any that are now covered by current_rules or already_validated_rules. "
            "Keep the rest unchanged. Do NOT add new rules at this stage. "
            "Leave empty if all previous missing rules are now covered."
        )
    )
    reasoning: str = Field(
        description="Which previous missing rules were resolved and which remain, and why."
    )


_SYSTEM_SECTION_INIT = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your task is to update the list of missing rules for a section before per-rule
validation begins.

You receive:
  - current_rules: rules extracted in this iteration
  - already_validated_rules: rules validated in previous iterations
  - previous_missing_rules: rules identified as missing in the previous iteration

For each rule in previous_missing_rules:
  - Check if a matching rule now exists in current_rules or already_validated_rules.
    A rule is "covered" if it has the same rule_type and openapi_field as the missing instruction.
  - If covered: remove it from the list.
  - If still absent: keep it.

Do NOT add new rules at this stage. Only filter what was previously identified.
Do NOT list rules about the absence of a construct (e.g. 'no query parameters').
"""

_USER_SECTION_INIT = """\
Section ID: {section_id}

Rules extracted in this iteration:
{current_rules}

Rules already validated in previous iterations:
{already_validated_rules}

Rules identified as missing in the previous iteration:
{previous_missing_rules}

Which of the previous missing rules are still genuinely absent?
"""

validator_section_init_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_SECTION_INIT),
    ("human",  _USER_SECTION_INIT),
])


# ---------------------------------------------------------------------------
# Stage 2 — Per-rule verdict
# ---------------------------------------------------------------------------

class ValidationVerdict(BaseModel):
    """Per-rule semantic verdict produced by the Validator Node."""

    valid: bool = Field(
        description=(
            "True if the rule is semantically correct as written: the rule_text is "
            "grounded in the section content, the openapi_mapping is accurate, and "
            "the openapi_value is consistent with the OpenAPI spec. "
            "False if the rule contains an incorrect mapping, hallucinated content, "
            "or an openapi_value that contradicts the spec."
        )
    )
    error_type: str = Field(
        default="correction",
        description=(
            "Only relevant when valid=False. "
            "'correction' — fix the rule's mapping or text. "
            "'split' — this rule is invalid and must be fully replaced by N atomic "
            "           rules listed in instruction. "
            "'discard' — remove this rule entirely, no replacement. "
            "Leave as 'correction' when valid=True."
        )
    )
    instruction: str = Field(
        default="",
        description=(
            "When valid=False: direct command to the Extractor for this rule. "
            "Empty when valid=True. "
            "For 'split': list all replacement rules the Extractor must produce. "
            "Examples:\n"
            "  correction: 'Fix openapi_field to parameters[in=query,name=scope]'\n"
            "  split: 'Replace with: (1) response openapi_field=4XX "
            "openapi_value=$ref:#/components/schemas/ErrorResponse; "
            "(2) response openapi_field=5XX same value'\n"
            "  discard: 'Discard — absence of parameters is not a valid OpenAPI rule'"
        )
    )
    new_missing_rules: list[str] = Field(
        default_factory=list,
        description=(
            "New missing rules discovered while assessing this specific rule. "
            "Use when this rule reveals a sibling rule that is absent from both "
            "already_validated_rules and current_missing_rules. "
            "Example: this rule covers 4XX and is valid, but 5XX is absent and "
            "not yet listed in current_missing_rules — add it here. "
            "Do NOT repeat entries already in current_missing_rules or already_validated_rules. "
            "Leave empty when no new gaps are found."
        )
    )


_SYSTEM_RULE = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

You give the final verdict on a single rule. The Reflector has done the analysis —
use it as input, but the decision is yours.

Assess whether the rule under review is correct as written:

  valid=True  → the rule is correct. instruction is empty.
  valid=False → the rule has a problem. Choose error_type:
    'correction' → fix this rule. Instruct the Extractor what to change exactly.
    'discard'    → remove this rule entirely. No replacement.
                   Use when the rule describes the absence of a construct
                   (e.g. 'no query parameters') — that cannot be mapped to OpenAPI.
    'split'      → this rule is invalid because it combines constructs that must be
                   separate. List ALL replacement rules in instruction.
                   This rule is discarded and fully replaced by the listed rules.

Additionally: if assessing this rule reveals a sibling rule that is absent from
both already_validated_rules and current_missing_rules, add it to new_missing_rules.
Example: this rule covers 4XX and is valid, but 5XX is absent and not listed anywhere
→ add '5XX response rule with ErrorResponse' to new_missing_rules.

VALIDATION RULES BY RULE_TYPE:
  path_operation  : openapi_field must be a single lowercase HTTP method (get/put/post/delete/patch).
                    openapi_value must be the uppercase HTTP method (GET/PUT/POST/DELETE/PATCH).
                    Split only when the rule combines multiple methods in one field (e.g. "put, patch").
                    Each split target must also be a path_operation — one per method.
                    Do NOT split a path_operation into request_body, response, or other rule_types.
                    If the Reflector suggests that kind of split, reject it and set valid=True.
  schema_property : openapi_field must be "properties.<name>". openapi_value must be a JSON Schema type.
  path_parameter  : openapi_field must be "parameters[in=path,name=<name>]".
  query_parameter : openapi_field must be "parameters[in=query,name=<name>]".
  response        : openapi_field must be a specific HTTP status code ("200", "400", etc.),
                    an OpenAPI wildcard ("1XX"–"5XX"), or "default".
                    "default" only when the section has no specific code (catch-all).
                    Each code or range mentioned explicitly must be its own rule.
                    A rule covering only one code (e.g. 4XX) is valid on its own even
                    if a sibling (5XX) is absent — do not reject it for that reason.
  request_body    : openapi_field must be "content". openapi_value must be a media type.
  security_scheme : openapi_field must be "type". openapi_value must be a valid OAuth2/http/apiKey value.

ALREADY-VALIDATED RULES:
  If the rule under review duplicates one already validated, set valid=False,
  error_type='discard', instruction='Duplicate of already-validated rule'.
"""

_USER_RULE = """\
Rule to validate:

  Section ID     : {section_id}
  Section title  : {section_title}
  Rule type      : {rule_type}
  Rule text      : {rule_text}
  OpenAPI object : {openapi_object}
  OpenAPI field  : {openapi_field}
  OpenAPI value  : {openapi_value}

Reflector assessment (analysis only — you decide):
  Confidence        : {reflection_confidence}
  Flagged           : {reflection_flagged}
  Reasoning         : {reflection_reasoning}
  Split suggestion  : {split_suggestion}
  Discard suggestion: {discard_suggestion}

Already-validated rules for this section:
{already_validated_rules}

Missing rules identified so far for this section (do NOT repeat these):
{current_missing_rules}

Section content (source of the rule):
{section_content}

Return your verdict on this rule and any new missing rules you identify.
"""

validator_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_RULE),
    ("human",  _USER_RULE),
])


# ---------------------------------------------------------------------------
# Stage 3 — Reflector review
# ---------------------------------------------------------------------------

class ReflectorReviewResult(BaseModel):
    """
    Final missing_rules after incorporating the Reflector's section analysis.
    """

    missing_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Final list of rules the Extractor must still produce. "
            "Start from current_missing_rules. "
            "For each rule in reflector_missing_rules: add it if not already covered "
            "by validated_rules_this_iteration, already_validated_rules, or current_missing_rules. "
            "Do not remove entries from current_missing_rules unless you have clear evidence "
            "they are now covered. "
            "Leave empty if everything is covered."
        )
    )
    reasoning: str = Field(
        description=(
            "Why these rules are still missing after reviewing the Reflector's analysis "
            "and what was added or excluded from the Reflector's suggestions."
        )
    )


_SYSTEM_REFLECTOR_REVIEW = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

You are performing the final step of section validation.

You have already:
  1. Filtered previous_missing_rules to keep only what is still absent (Stage 1)
  2. Validated each rule and accumulated new gaps (Stage 2)

Now review the Reflector's section analysis against the full context and produce
the definitive missing_rules list for this iteration.

For each entry in reflector_missing_rules:
  - If already covered by validated_rules_this_iteration or already_validated_rules → exclude.
  - If already in current_missing_rules → do not duplicate.
  - If genuinely still absent → add to the final list.

The final missing_rules = current_missing_rules + confirmed additions from reflector_missing_rules.
Do NOT remove from current_missing_rules unless you have clear evidence it is now covered.
Do NOT list rules about the absence of a construct.
"""

_USER_REFLECTOR_REVIEW = """\
Section ID: {section_id}

Rules validated in this iteration:
{validated_rules_this_iteration}

Rules already validated in previous iterations:
{already_validated_rules}

Missing rules accumulated in Stages 1 and 2:
{current_missing_rules}

Reflector's section analysis — rules it identified as missing:
{reflector_missing_rules}

Produce the final missing_rules list for this section.
"""

validator_reflector_review_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_REFLECTOR_REVIEW),
    ("human",  _USER_REFLECTOR_REVIEW),
])
