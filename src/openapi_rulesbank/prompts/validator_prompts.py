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
from openapi_rulesbank.prompts.rule_type_definitions import rule_type_definitions_for_template

# The one definition of each rule type, the same in every stage (see
# prompts/rule_type_definitions.py).
_RULE_TYPE_DEFINITIONS = rule_type_definitions_for_template()
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
  - ALSO remove it when the pre-analysis verdict shows it is an INHERITED field of
    this section's allOf base (already provided by the base $ref — NOT missing), or
    a rule the verdict shows belongs to ANOTHER section. Keeping such a rule only
    makes the Extractor re-emit what the per-rule check then discards.
  - ALSO remove it when its $ref type is listed in the verdict as HELD: the rule was
    held on purpose because where the type is defined is not settled in this
    section; it is kept and emitted by the section that settles it. Not missing here.
  - If still genuinely absent (not inherited / not owned elsewhere / not held): keep it.

Do NOT add new rules at this stage. Only filter what was previously identified.
Do NOT list rules about the absence of a construct (e.g. 'no query parameters').
"""

_USER_SECTION_INIT = """\
Section ID: {section_id}

Pre-analysis verdict for this section (kind, schema-home, allOf base, and INHERITED
fields the base already provides — an inherited field is NOT missing):
{section_verdict}

Rules extracted in this iteration:
{current_rules}

Rules already validated in previous iterations:
{already_validated_rules}

Rules identified as missing in the previous iteration:
{previous_missing_rules}

Scope contract — which Management Service these rules are being built for. A rule
for another service, or for a capability marked "(not in this YAML)", was never
this section's to produce: drop it from the missing list rather than carrying it
forward:
{contract_summary}

Auxiliary context — supporting documents and any instruction the user gave for
this run. A join it settles is not a missing rule:
{helper_context}

Which of the previous missing rules are still genuinely absent? (Drop inherited
fields, rules owned by another section, and anything the auxiliary context or a
modelling convention already settles — a callback anchored by convention on the
service's create operation is resolved, not absent.)
"""

validator_section_init_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_SECTION_INIT + "\n\n" + _RULE_TYPE_DEFINITIONS),
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
            "Leave as 'correction' when valid=True. "
            "This verdict covers the rule under review, which EXISTS — so a wrong "
            "rule is a 'correction' here, and the SAME rule must NOT also appear in "
            "new_missing_rules (a target is one classification: correction OR "
            "missing, never both). new_missing_rules is only for a DIFFERENT sibling "
            "that is absent everywhere."
        )
    )
    instruction: str = Field(
        default="",
        description=(
            "When valid=False: direct command to the Extractor for this rule. "
            "Empty when valid=True. "
            "A 'correction' MUST NAME THE VALUE TO WRITE, not the work to do. "
            "State the field and the literal value that belongs in it — the "
            "Extractor applies what you name and cannot look anything up. An "
            "instruction that says to 'apply the fact', 'resolve the gap', "
            "'use the convention' or 'consult the reference' names no value: the "
            "Extractor re-reads the same silent clause, re-emits the same value, "
            "and the section burns every attempt on a rule you never told it how "
            "to change. If you cannot name the value, the rule is not correctable "
            "— leave it valid=True and record what is missing in new_missing_rules "
            "instead. Where the Reflector's reasoning already names the right "
            "value, take it from there. "
            "For 'split': list all replacement rules the Extractor must produce. "
            "Examples:\n"
            "  correction: 'Fix openapi_field to parameters[in=query,name=scope]'\n"
            "  correction: 'Set openapi_value to application/merge-patch+json'\n"
            "  NOT a correction: 'Apply the declared media-type gap' (no value)\n"
            "  split: 'Replace with: (1) response openapi_field=4XX "
            "openapi_value=$ref:#/components/schemas/ErrorResponse; "
            "(2) response openapi_field=5XX same value'\n"
            "  discard: 'Discard — absence of parameters is not a valid OpenAPI rule'"
        )
    )
    new_missing_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Sibling rules — DIFFERENT targets from the rule under review — that this "
            "assessment reveals are genuinely ABSENT everywhere (not in "
            "already_validated_rules, not in current_missing_rules, not in "
            "sibling_rules). "
            "TWO RULES, READ LITERALLY:\n"
            "1. ONE classification per target. A single atomic target (one status "
            "code, one method, one property) is EITHER a correction/valid rule (it "
            "exists) OR missing (it does not) — NEVER both. The rule under review "
            "exists by definition, so if it is wrong it is a correction; NEVER put "
            "the rule under review itself in new_missing.\n"
            "2. A truly-absent counterpart IS added here. If, judging the 4XX rule, "
            "the sibling 5XX is mapped nowhere (not in sibling_rules, "
            "already_validated_rules, or current_missing_rules), add the 5XX rule to "
            "new_missing — it is a distinct target that is really missing. But if a "
            "5XX rule is already present as a sibling, add nothing. "
            "So: never double-classify the SAME rule; do add a distinct sibling that "
            "is genuinely absent. Leave empty when no new gaps are found."
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
    'split'      → ONLY when THIS rule's own openapi_field literally combines two or
                   more targets in one field (e.g. field "put, patch", or one field
                   naming both "4xx/5xx"). List ALL replacement rules; this rule is
                   discarded and fully replaced by them.
                   HARD BAN — atomic rules are never split: a rule whose openapi_field
                   already names a SINGLE target (one method, one status code such as
                   "4XX" or "5XX", one property) is atomic. Do NOT split it.
                   In particular, if a single "5XX" rule is under review and "4XX"
                   already exists (or vice-versa), that is NOT a split and NOT a
                   replacement: 4XX and 5XX are distinct, co-existing responses. The
                   rule under review stays valid=True. If instead the SIBLING code is
                   the one missing (e.g. 4XX exists, 5XX does not appear anywhere),
                   record the missing one in new_missing_rules — never by splitting
                   or replacing this rule.

Additionally: if assessing this rule reveals a sibling rule that is absent from
both already_validated_rules and current_missing_rules, add it to new_missing_rules.
Example: this rule covers 4XX and is valid, but 5XX is absent and not listed anywhere
→ add '5XX response rule with ErrorResponse' to new_missing_rules.

A JOIN THE CLAUSE LEAVES IMPLICIT IS SETTLED BY CONVENTION, NOT BY THE SECTION:

  The Extractor is given the modelling conventions the published YAMLs follow,
  for the connections a 3GPP clause never states: a callback anchors on the
  operation through which the consumer supplies its address (the create operation
  on the service's resource, where there is no subscription operation); variables
  of the common URI structure belong to the base URL and are not path parameters;
  a clause naming only a data type carries it in the reference's request-body
  media type; an attribute typed by a named type is a $ref to it.

  A rule following one of these is CORRECT, even though the section does not say
  so — do not reject it for lacking support in the text, and do not instruct the
  Extractor to report the join as missing. Reject it only when a clause, the
  auxiliary context or the user's instructions state something different, and
  then name what they state.

AN IDENTIFIER IS TAKEN FROM A SOURCE, NEVER SUPPOSED:

  A path key, an operation name, a schema name and a field name come from one of
  these, in this order:
    1. the section content, or a clause of this document in the context;
    2. the auxiliary context / the user's instructions for this run;
    3. the OpenAPI reference, when it shows a standard value for that field.

  A rule whose identifier comes from any of the three is correct. Reject one that
  comes from none: a path belonging to a DIFFERENT Management Service
  ("/notificationSink" or "/eventListener" in a service whose clauses never
  mention them), or a name coined because it reads plausibly. Neither supposing
  nor inventing is allowed — set valid=False, error_type='correction', and name
  what the section does support instead. Such a rule reads as complete and points
  at something that does not exist in this YAML, which is why it survives every
  other check.

  AND A NAME IS CHECKED AGAINST ITS DEFINING HEADING, NOT AGAINST ITS FORM.
  The sibling clause list above holds those headings — "<clause> Notification
  <name>", "<clause> Type <Name>". Look the name up there first: a rule's name is
  right when it matches the heading naming the same thing, and wrong when it does
  not, however well-formed it reads. A data-type column can write a notification
  in flawless lowerCamelCase and still drop a segment the heading carries, and
  those columns are where the spec's typos live. The forms below tell you WHICH
  heading to look in; they never make a name correct on their own.

  EACH KIND OF IDENTIFIER ALSO HAS ITS OWN FORM. The spec names the same concept
  differently by position, and one form is never produced by re-casing another:
    - TYPE (schema): UpperCamelCase, acronyms normalised — from the heading
      "Type <Name>"/"Enumeration <Name>". Belongs in "components/schemas/<Name>"
      and in every $ref. e.g. "NotifyMoiCreation", "AlarmId".
    - NOTIFICATION: lowerCamelCase, acronyms as written — from the heading
      "Notification <name>". Belongs in a callback's source_name and its
      "callbacks.<name>". e.g. "notifyMOICreation".
    - PROPERTY: lowerCamelCase, acronyms normalised — from the owning type's
      attribute table. Belongs in "properties.<name>", "parameters[...]" and
      "required". e.g. "notificationId".
  So notification "notifyMOICreation" carries type "NotifyMoiCreation": a rule
  using "NotifyMOICreation" as a schema, or "notifyMoiCreation" as a
  notification, has the right concept in the wrong form — correct the form, do
  not discard the rule. When a data-type column disagrees with a defining
  heading, the heading is right; the column is hand-written and carries the
  spec's typos. Where only the column exists and it is the sole source, accept
  it rather than rejecting the rule: a name in the wrong form still points at
  something real, which an absent rule does not.

A VALUE THE CLAUSE DOES NOT SPELL OUT IS NOT AN INVENTION:

  For every field, one question settles the value: IS IT SPECIFIED IN THE SPEC?
  The Extractor takes it from the spec where a clause states it, and from the
  OpenAPI reference where none does — a media type, a JSON Schema type, a path
  key, any of them. A 3GPP clause routinely specifies none of these.

  So do NOT reject a value for lacking a source in the section, and never
  instruct the Extractor to "use the confirmed value" or to stop "asserting"
  one: the next attempt reads the same silent clause, so it either repeats the
  value or writes a placeholder, and the section burns its attempts and reaches
  the Builder with nothing. Judge the value on whether it is RIGHT — a media type
  that is a media type, a JSON Schema type that fits the attribute, a path key
  that is the resource. Correct a wrong value by naming the right one.

  Where the spec DOES specify a value and the rule used another, that is a
  correction: name the value the spec gives.

VALIDATION RULES BY RULE_TYPE:
  path_operation  : openapi_field must be a single lowercase HTTP method (get/put/post/delete/patch).
                    openapi_value must be the uppercase HTTP method (GET/PUT/POST/DELETE/PATCH).
                    Split only when the rule combines multiple methods in one field (e.g. "put, patch").
                    Each split target must also be a path_operation — one per method.
                    Do NOT split a path_operation into request_body, response, or other rule_types.
                    If the Reflector suggests that kind of split, reject it and set valid=True.
  schema_property : openapi_object must be "components/schemas/<Name>" naming the schema
                    itself — never a nested path inside it. <Name> must be the data type
                    as the defining section's heading spells it ("Type <Name>" /
                    "Enumeration <Name>"), with that exact casing; a property name, a
                    notification name, or an invented suffix (Request, RequestBody,
                    Response) in its place is wrong — set valid=False,
                    error_type='correction' and give the name from the heading.
                    openapi_field must be "properties.<name>" OR a schema-level keyword
                    (enum | required | oneOf | anyOf | allOf | items | additionalProperties).
                    openapi_value must be a JSON Schema type, a $ref, or the keyword's
                    value (e.g. the list of enum values or required property names).
  path_parameter  : openapi_field must be "parameters[in=path,name=<name>]".
  query_parameter : openapi_field must be "parameters[in=query,name=<name>]".
  response        : openapi_field must be a specific HTTP status code ("200", "400", etc.),
                    an OpenAPI wildcard ("1XX"–"5XX"), or "default".
                    "default" only when the section has no specific code (catch-all).
                    Each code or range mentioned explicitly must be its own rule.
                    A rule covering only one code (e.g. 4XX) is valid on its own even
                    if a sibling (5XX) is absent — do not reject it for that reason.
                    NEVER split or replace one response code with another: a 5XX rule
                    is not a "split" of a 4XX rule and must not be replaced by 4XX (or
                    vice-versa). Each is atomic. If reviewing one reveals the sibling
                    is missing, put the sibling in new_missing_rules — keep this rule
                    valid=True. Reject any Reflector split suggestion between codes.
  request_body    : openapi_field must be "content". openapi_value must be a media type.
  security_scheme : openapi_field must be "type". openapi_value must be a valid OAuth2/http/apiKey value.
  callback        : source_name and <name> must be a notification name the spec
                    writes somewhere — the section, a mapping table in the
                    context, or the reference memory. A coined one
                    ("notifyEvent", "notifyChange") is wrong however plausible:
                    set valid=False, error_type='correction', and name the
                    notification the spec gives. Where a table lists several
                    notifications and one rule stands for all of them, that is
                    also a correction — one callback rule per notification.
                    THE SAME NOTIFICATION TWICE IS ALWAYS AN ERROR. The spec may
                    spell one notification two ways (a heading and a data-type
                    column that disagree), and the rules then carry it under both
                    — two callbacks, or two request_body/response sets, for one
                    notification. Reject the duplicate: set valid=False,
                    error_type='correction' on the one whose name is NOT from the
                    "Notification <name>" heading, and keep the other. Judge by
                    the concept, not the string: two names differing only by a
                    dropped segment or by acronym casing are one notification.
                    openapi_object must be "paths.<path>.<method>.callbacks.<name>" —
                    openapi_field is the delivery method (normally "post"); openapi_value
                    is the runtime expression for the consumer address as the spec writes
                    it. Callback vs path_operation is decided by the callback and
                    path_operation definitions in RULE TYPE DEFINITIONS — check BOTH
                    parts of the callback one before accepting a callback, and never
                    by the fact that it is a notification or by where the resource
                    is hosted. When the section does not give the delivery URI, read
                    it in the heading of the resource clause it belongs to and in the
                    sibling clauses. A callback that fails the definition is not
                    corrected into a path_operation that already exists — see
                    ALREADY-VALIDATED RULES.

RULES BUILT ON A DECLARED GAP — read what the gap actually says:

  declared_gaps lists facts the Extractor says it needed for THIS section and did
  not have. It emitted its rules anyway. Two very different things are reported
  here, and ONE QUESTION separates them:

    DID THE GAP LEAVE THE MAPPING UNDETERMINED, OR DOES IT MERELY RECORD THAT A
    DEFAULT WAS APPLIED?

  (a) UNDETERMINED — nothing supplied the value, so whatever the rule carries is
      a guess: "could not tell which operation to anchor the callback on, emitted
      as a path", "no notification name anywhere, so the callback could not be
      named". The rule looks well-formed and points at something that does not
      exist. Set valid=False, error_type='correction', and NAME THE VALUE that
      belongs there. If you cannot name it, do not issue an empty correction —
      leave the rule valid=True and put what is missing in new_missing_rules.

  (b) DEFAULT APPLIED — the clause is silent and the Extractor filled the field
      from the OpenAPI reference or a modelling convention, exactly as instructed,
      and reported it because reporting is required: "the clause names only the
      data type, so application/json was used per the request-body convention".
      THIS RULE IS CORRECT. The two blocks above say so — a value the clause does
      not spell out is not an invention, and a join left implicit is settled by
      convention. The gap is an audit note, not a defect. Set valid=True. Do NOT
      instruct the Extractor to "apply the fact", "resolve the gap" or "use the
      reference": the convention is where the value already came from, so the next
      attempt re-reads the same silent clause, re-emits the same value, and the
      section burns every attempt without a single thing changing.

  Judge by what the gap says the Extractor lacked, not by the fact that a gap
  exists: a section that declared one gap does not thereby have bad rules, and the
  gap touches only the rules whose value it left undetermined. A rule no gap
  touched is judged normally. When a gap describes a rule the Extractor could not
  emit at all — absent, not wrong — that belongs in new_missing_rules.

ALREADY-VALIDATED RULES:
  If the rule under review duplicates one already validated, set valid=False,
  error_type='discard', instruction='Duplicate of already-validated rule'.
  The same holds when the only correction you would give turns this rule INTO one
  already validated or present among this section's rules (e.g. "change rule_type
  to path_operation" where that path_operation already exists): the corrected rule
  would be a duplicate, so discard this one instead, naming the rule it would
  duplicate. Never send back a correction whose result already exists.

DUPLICATES ACROSS SECTIONS — discard ONLY a true duplicate; when unsure, KEEP:
  similar_rules_other_sections lists rules already accepted in OTHER sections that
  resemble this one. Several sections often describe the same operation (e.g. the
  operation mapping table, the resource structure, and the HTTP method subsection
  all describe one PUT), so the same rule can reach you more than once.

  Discard THIS rule only when an earlier one has the SAME rule_type AND the SAME
  target. For most types the target is openapi_object + openapi_field alone:
    - schema_property: same schema AND same property/keyword
      (components/schemas/X · properties.foo);
    - path_operation: same path AND same method;
    - parameter: same path+method AND same field.
  For these, a difference only in the VALUE's wording/format is NOT enough to
  keep the rule (e.g. a property typed "string" vs
  {{"type":"string","format":"date-time"}} for the SAME schema+property is still a
  duplicate), because their address holds at most one such rule.

  EXCEPTION — request_body and response: the target is
  openapi_object + openapi_field + openapi_VALUE. One operation may carry several
  request bodies, and one status code several response bodies, that share the
  field ('content' / the status code) and differ only in media type or schema
  (the ProvMnS PATCH defines 4 request-body media types, its GET 200 three
  response media types). So for these two types a DIFFERENT value is a DIFFERENT
  rule — KEEP it; discard only when object, field AND value all match.
  Only when a rule is a duplicate: set valid=False, error_type='discard',
  instruction='Duplicate of rule already validated in section <id>'.

  It is NOT a duplicate — KEEP it — when the target differs in ANY of these,
  however similar the rules look:
    - a DIFFERENT schema (enum of PerfNotificationTypes vs enum of
      PerfMetricDirection are two separate rules — same rule_type, different
      openapi_object);
    - a DIFFERENT property or keyword of the same schema;
    - a DIFFERENT status code (204 vs default vs 4XX are separate response rules);
    - a DIFFERENT method or path.
  When you are not certain two rules share the exact same object+field, KEEP this
  rule — dropping a distinct rule loses coverage, which is worse than a duplicate
  the deterministic dedup can still catch later.

A RULE THAT SHOULD HAVE BEEN HELD:
  Applies ONLY to a type listed under "held $refs" in THIS section's verdict. When
  this rule's value refers to one of those types — in any position: a body, a
  parameter, a property, an items element, a map value, an allOf/oneOf/anyOf
  member — it was emitted when it should have been held: its $ref form asserts
  where the type is defined, and nothing has stated that. Set valid=False,
  error_type='correction', and an instruction naming the type: "Its $ref points at
  <Type>, which is held in this section (origin not stated yet): move this rule to
  deferred_rules, written the same way." Do not discard it and do not change its
  $ref target yourself — the rule is right; only where it goes is wrong.
  NEVER give this instruction for a type the verdict does not list as held — in
  particular a type of a held rule completed in this section: that type is SETTLED
  here, and sending its rule to deferred_rules makes it wait for a section that
  will never come.

HELD RULES EMITTED IN THIS SECTION:
  When the pre-analysis verdict lists "held rules completed and emitted in this
  section" and this rule is one of them (same rule type, object and field), it was
  written by the EARLIER section named there, held because the origin of its $ref
  type was not stated in that section, and emitted here ONCE, by the code, because
  THIS section states where the type is defined. It is expected NOT to appear in
  this section's text. Do NOT discard it as "owned by another section" or as
  unsupported by this section's text, and do not correct its $ref target — the
  code set it from this section's settling of the type.
  - Correct everywhere else → valid=True.
  - Wrong in some other field → error_type='correction', and the instruction names
    that field and the exact value to write ("Set <field> to <value>"). Never
    instruct to move it to deferred_rules, to hold it, or to emit it again.

FILENAME BUILT BY CONVENTION:
  When the verdict lists "filenames built by convention" and this rule's $ref uses
  one of those files, the file was built by the naming convention because no source
  gave it. Do NOT fail the rule for that, and do not ask for another filename; say in
  validation_notes that the filename came from the convention fallback.
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
  Belongs-in-final score : {reflector_score}
  Flagged           : {reflection_flagged}
  Reasoning         : {reflection_reasoning}
  Split suggestion  : {split_suggestion}
  Discard suggestion: {discard_suggestion}

Previous correction of THIS rule — the instruction it got on an earlier attempt:
{previous_instruction}
That instruction did not produce a correct rule, or the rule would not be back here.
If the rule is still wrong, do NOT repeat it: that wording already failed. Write a
more explicit instruction that replaces it — name the exact field and the exact
value to write, and say what in the rule is still wrong after that attempt. If the
earlier instruction itself was mistaken (it asked for something the rule must not
do), say so and give the right one.

Already-validated rules for this section:
{already_validated_rules}

Other rules of THIS section right now (extracted this iteration but not yet
validated, plus ones validated earlier) — its siblings. A counterpart present here
is NOT missing; never add it to new_missing_rules:
{sibling_rules}

Similar rules already validated in OTHER sections (discard this rule if it merely
restates one of them — see DUPLICATES ACROSS SECTIONS):
{similar_rules_other_sections}

Missing rules identified so far for this section (do NOT repeat these):
{current_missing_rules}

Scope contract — which Management Service these rules are being built for, and
what belongs to it. This is what settles whether an identifier is the service's
own: a rule naming another service's resource, or a capability marked "(not in
this YAML)", is a correction however well-formed it looks:
{contract_summary}

Auxiliary context — supporting documents and any instruction the user gave for
this run. It OVERRIDES the modelling conventions: where it states how something
maps, a rule following IT is correct and one following the convention is the
correction:
{helper_context}

Facts the Extractor said it lacked when it produced this section, and that are
being fetched for the next attempt (empty when it declared none):
{declared_gaps}

Pre-analysis verdict for this section (decided before extraction):
{section_verdict}

The sibling CLAUSES of this service block, by title. A canonical name is written
in full only here: "<number> Notification <name>" gives a notification's name,
"<number> Type <Name>" a type's. The section under review may spell the same thing
differently in a data-type column — those columns are hand-written and are where
the spec's typos live. Judge a name against these headings, not against its shape
alone: a name can have the right form and still be the wrong name. When a rule's
name matches no heading while a heading names the same thing, that is a
correction, and the value to name is the heading's spelling. These titles are also
where an operation defined in another clause is recorded ("maps <op>: <method>
<path>"), so a rule pointing at such an operation can be checked without reading
that clause:
{sibling_summary}
  Enforce it — a rule that violates it is a discard (valid=False,
  error_type='discard'):
  - a schema_property in a section that is NOT a schema home → the attribute is
    defined in its own "Type <Name>" section, not in this mapping/resource one.
  - a properties.<name> (or a "required") for a name in the inherited-fields list →
    that field comes from the base via the single allOf $ref, so it must not also
    be emitted as an own property here.

Section content (source of the rule):
{section_content}

Return your verdict on this rule and any new missing rules you identify.
"""

validator_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_RULE + "\n\n" + _RULE_TYPE_DEFINITIONS),
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

For each entry in reflector_missing_rules AND current_missing_rules:
  - If already covered by validated_rules_this_iteration or already_validated_rules → exclude.
  - If the pre-analysis verdict shows it is an INHERITED field of this section's allOf
    base → exclude: it is provided by the base $ref, NOT missing. Also exclude any
    rule the verdict shows belongs to ANOTHER section. This overrides the "do not
    remove" caution below — an inherited or foreign rule must never survive as
    missing, or the Extractor loops re-emitting what gets discarded.
  - If its $ref type is listed in the verdict as HELD → exclude: the rule was held
    on purpose (where the type is defined is not settled here) and is emitted by the
    section that settles it. It must never survive as missing — also overrides the
    "do not remove" caution.
  - If it is one of the held rules the verdict lists as completed and emitted in
    this section → exclude: the code emitted it here, so it EXISTS. Asking for it
    again only makes the Extractor write a copy. Also overrides the caution.
  - If its target is already being CORRECTED this iteration (see the corrections
    list) → exclude: that rule EXISTS and the correction covers it. A target is one
    thing — correction OR missing, never both. Never keep as missing a target that
    is in the corrections list.
  - If already in current_missing_rules → do not duplicate.
  - If genuinely still absent (and not inherited / not owned elsewhere) → keep/add.

The final missing_rules = the still-genuinely-absent rules from current_missing_rules
plus confirmed additions from reflector_missing_rules.
Do NOT remove a genuinely-absent rule from current_missing_rules without clear evidence
it is now covered (inheritance / foreign-ownership counts as such evidence).
Do NOT list rules about the absence of a construct.
"""

_USER_REFLECTOR_REVIEW = """\
Section ID: {section_id}

Pre-analysis verdict for this section (kind, schema-home, allOf base, and INHERITED
fields the base already provides — an inherited field is NOT missing):
{section_verdict}

Rules validated in this iteration:
{validated_rules_this_iteration}

Rules already validated in previous iterations:
{already_validated_rules}

Missing rules accumulated in Stages 1 and 2:
{current_missing_rules}

Reflector's section analysis — rules it identified as missing:
{reflector_missing_rules}

Rules being CORRECTED this iteration (they exist, only need a fix — their targets
are NOT missing, exclude them from the final list):
{corrections}

Scope contract — which Management Service these rules are being built for. Keep
nothing in the final missing list that falls outside it: another service's rule,
or one for a capability marked "(not in this YAML)", is not missing here:
{contract_summary}

Auxiliary context — supporting documents and any instruction the user gave for
this run. A join it settles, like one a modelling convention settles, is resolved
and NOT a missing rule:
{helper_context}

Produce the final missing_rules list for this section.
"""

validator_reflector_review_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_REFLECTOR_REVIEW + "\n\n" + _RULE_TYPE_DEFINITIONS),
    ("human",  _USER_REFLECTOR_REVIEW),
])
