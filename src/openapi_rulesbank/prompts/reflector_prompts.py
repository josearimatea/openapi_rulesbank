# src/prompts/reflector_prompts.py

"""
Prompt templates and output schemas for the Reflector Node.

The Reflector applies Chain-of-Thought self-reflection to each RawRule produced
by the Extractor. For each rule it receives:
  - The rule itself (section_id, rule_text, openapi_mapping)
  - Relevant OpenAPI reference chunks retrieved via RAG

It returns a ReflectionResult with confidence score, reasoning trace, and flag.

Classes:
    ReflectionResult — structured LLM output for a single rule reflection

Constants:
    reflector_prompt — ChatPromptTemplate used to invoke the LLM per rule
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured output schema
# The LLM is asked to return JSON matching ReflectionResult via
# llm.with_structured_output(ReflectionResult).
# ---------------------------------------------------------------------------

class ReflectionResult(BaseModel):
    """Self-reflection assessment produced by the Reflector Node for one rule."""

    reflector_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How sure you are that this rule SHOULD BE IN THE FINAL RULE SET — NOT "
            "how faithfully it copies the section text. Separating rules that belong "
            "from rules that must go is your whole job. "
            "1.0 = the rule is correct AND belongs to THIS section: right rule_type, "
            "this section is its owner, it is not an inherited field, mapping valid. "
            "0.0 = the rule must NOT be in the final set — it is an inherited field "
            "(comes from a base type via an allOf $ref), it is owned by another "
            "section, it duplicates one already validated elsewhere, its mapping is "
            "invalid, or it is not mappable to OpenAPI. "
            "A rule can be copied word-for-word from the section and still score LOW "
            "when it does not belong in the final output. When this score is below "
            "0.5 you MUST set discard_suggestion True."
        )
    )
    reflection_reasoning: str = Field(
        description=(
            "Chain-of-Thought explanation of why this rule was extracted, "
            "whether its openapi_mapping is correct, and how the retrieved "
            "OpenAPI reference context supports or contradicts it."
        )
    )
    reflection_flagged: bool = Field(
        description=(
            "True if this rule should receive priority attention during validation. "
            "Flag when: reflector_score < 0.7, the openapi_mapping is ambiguous, "
            "the rule is not explicitly stated in the section, or the RAG context "
            "suggests a different mapping."
        )
    )
    split_suggestion: str = Field(
        default="",
        description=(
            "Suggest a split ONLY when THIS rule's own openapi_field literally packs "
            "two or more targets into one field. Empty string otherwise. "
            "Example: a single field 'put, patch' → split into one rule per method; a "
            "single field '4xx/5xx' → split into a 4XX rule and a 5XX rule. In every "
            "case each split target keeps the SAME rule_type. "
            "Do NOT suggest splitting a path_operation into request_body, response, "
            "or other rule_types — those are separate complementary rules, not a "
            "decomposition of path_operation. "
            "NEVER suggest a split for a rule that is already atomic — one field with "
            "a single method, a single status code (e.g. '4XX' alone, or '5XX' "
            "alone), or a single property. A lone 5XX rule is NOT a split of a 4XX "
            "rule; they coexist. If a sibling code is missing, that is a coverage gap "
            "for the Validator, not a split of this rule."
        )
    )
    discard_suggestion: bool = Field(
        default=False,
        description=(
            "True whenever this rule should NOT be in the final rule set — i.e. "
            "whenever reflector_score is low. Set it in ALL of these cases, not just "
            "unmappable ones:\n"
            "  - the field is INHERITED from a base type (present via an allOf $ref "
            "    to another schema) — it must not be re-emitted as an own property "
            "    or 'required' of the subtype from this section;\n"
            "  - the rule is OWNED BY ANOTHER SECTION (this section only references a "
            "    type/operation whose real definition lives elsewhere);\n"
            "  - the rule DUPLICATES one already validated in another section;\n"
            "  - the rule describes the ABSENCE of a construct (e.g. 'no query "
            "    parameters'), or otherwise cannot be mapped to OpenAPI.\n"
            "Use the pre-analysis verdict (inherited fields, schema-home flag) as "
            "evidence. When in doubt whether a field is inherited or owned elsewhere, "
            "flag it True — the Validator makes the final call."
        )
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your job is to decide, for one extracted rule, whether it BELONGS IN THE FINAL
RULE SET — and to mark clearly what is right, what must go, and what needs a split.
Judging fidelity to the text is not enough: a rule can quote the section perfectly
and still not belong here.

For each rule, reason through the following questions:
  1. Is the rule explicitly stated or clearly implied by the section content?
  2. Is the openapi_mapping (object, field, value) correct and complete?
  3. Does the OpenAPI reference context below confirm or contradict this rule?
  4. Does this rule BELONG in the final set for THIS section? It does NOT when it is
     an inherited field (present via an allOf $ref to a base type), when another
     section is the real owner of the type/operation, when it duplicates a rule
     already validated elsewhere, or when its mapping is invalid/unmappable. Set
     reflector_score to reflect this belonging (1.0 = clearly belongs, 0.0 = clearly
     must go), NOT how closely it echoes the text.
  5. Should this rule be split into multiple more specific rules?
     Split ONLY a rule whose own field packs several targets together — e.g. one
     field '4xx/5xx' becomes a 4XX rule and a 5XX rule, one field 'put, patch'
     becomes two. A rule already carrying a SINGLE target (a lone 4XX, a lone 5XX,
     one method, one property) is atomic: do NOT split it. A lone 5XX is never a
     split of 4XX — they are separate co-existing rules.
  6. Should this rule be discarded? Set discard_suggestion True whenever it does NOT
     belong (question 4): an inherited field re-emitted here, a rule owned by another
     section, a cross-section duplicate, the absence of a construct (e.g. 'no query
     parameters'), or anything not mappable to OpenAPI. A low reflector_score and
     discard_suggestion=True must agree.

Then produce:
  - A confidence score (0.0 to 1.0)
  - A reasoning trace explaining your assessment step by step
  - A flag indicating whether the rule needs priority validation
  - A split suggestion if the rule should be broken into multiple rules
  - A discard suggestion if the rule should not exist at all

Relevant OpenAPI Specification context (retrieved for this rule):
{openapi_reference_context}
"""

_USER = """\
Rule to assess:

  Section ID     : {section_id}
  Section title  : {section_title}
  Rule type      : {rule_type}
  Rule text      : {rule_text}
  OpenAPI object : {openapi_object}
  OpenAPI field  : {openapi_field}
  OpenAPI value  : {openapi_value}

Pre-analysis verdict for THIS section (decided before extraction):
{section_verdict}
  Enforce it — when the rule above violates it, set discard_suggestion True and
  explain in reflection_reasoning:
  - If it says this section is NOT a schema home and the rule above is a
    schema_property, the rule is WRONG here: this is a mapping/resource section,
    and the attribute belongs to its own "Type <Name>" definition section, not here.
  - If it lists inherited fields and the rule above is a properties.<name> (or a
    "required") for one of those inherited names, the rule is WRONG: that field
    comes from the base via the single allOf $ref, so it must not also be emitted
    as an own property here.

Other rules of THIS section (extracted in this batch, plus ones already validated
in earlier iterations of the section):
{sibling_rules}
  Use this only as context to judge the rule above as part of the whole — do NOT
  re-assess the siblings. In particular, a rule that has a natural counterpart
  (e.g. a 4XX response whose sibling is 5XX, or a PUT whose sibling is PATCH) is
  NOT incomplete when that counterpart already appears in the list above: the
  counterpart exists (extracted or already validated), so do not treat the rule as
  missing its sibling.

Section content (the source from which this rule was extracted):
{section_content}

Related 3GPP spec context (retrieved from the source document — use to check the
rule is grounded in the specification, e.g. the abstract stage 2 definition):
{spec_context}

Apply Chain-of-Thought reasoning and return your assessment.
"""

reflector_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _USER),
])


# ---------------------------------------------------------------------------
# Section-level completeness assessment (Fase 2)
# Invoked once per section after all individual rules have been reflected.
# ---------------------------------------------------------------------------

class ContextQuery(BaseModel):
    """One retrieval query answering one gap the Extractor declared."""

    gap: str = Field(
        description="The gap this query answers, copied from the declared gaps."
    )
    query: str = Field(
        description=(
            "Search query, in the vocabulary of the source being searched — the "
            "words the ANSWER would be written in, not the words the gap is "
            "phrased in. Name the concrete thing sought (a clause's subject, a "
            "type name, an OpenAPI construct); never repeat the section number "
            "or a phrase like 'is not stated in this section'."
        )
    )
    sources: list[str] = Field(
        description=(
            "The collections to search, one or two, from: "
            "'spec' — another clause of THIS 3GPP document; "
            "'corpus' — a DIFFERENT 3GPP specification; "
            "'openapi' — the OpenAPI specification itself. "
            "A gap usually has two halves — WHICH value this service uses, and "
            "HOW that value is written in OpenAPI — so pair 'openapi' with "
            "whichever of 'spec'/'corpus' holds the value. Never pair 'spec' "
            "with 'corpus': a type is either defined in this document or "
            "imported from another, never both. One source alone when the gap "
            "has only one half."
        )
    )


class SectionReflection(BaseModel):
    """
    Completeness assessment for a full section — produced once per section
    after all individual rules have been reflected (Fase 2).
    """

    context_queries: list[ContextQuery] = Field(
        default_factory=list,
        description=(
            "One query per declared gap WORTH looking up — never more than one "
            "per gap, and fewer entries than gaps when some cannot be answered "
            "by retrieval. Empty when none are, and always empty when no gaps "
            "were declared."
        ),
    )
    missing_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Rules genuinely missing from this section. "
            "Each entry is a direct instruction for the Extractor. "
            "Example: 'Extract rule for HTTP 201 Created response for PUT operation'. "
            "Do NOT include rules already present in current_rules or already_validated_rules."
        )
    )
    reasoning: str = Field(
        description=(
            "Why these rules are missing and what in the section content "
            "supports their existence."
        )
    )


_SECTION_SYSTEM = """\
You are an expert in 3GPP technical specifications and OpenAPI design.

Your task is to assess whether all OpenAPI rules have been extracted from a 3GPP section.

You will receive:
  - The full section content
  - Rules just extracted and reflected in this iteration (current_rules)
  - Rules already validated in previous iterations (already_validated_rules)
  - Rules identified as missing in the previous iteration (previous_missing_rules)

SYSTEMATIC COMPLETENESS CHECK:
For each rule_type below, check whether the section content implies it should exist.
If yes, verify that at least one rule of that type is present in current_rules OR
already_validated_rules. If not present, report it as missing.

  path_operation  → Did the section define an HTTP method (GET, PUT, POST, DELETE, PATCH)
                    on a resource path? If yes → must have one path_operation rule per method.

  request_body    → Did the section describe a request body, input data structure, or
                    mandatory payload? If yes → must have at least one request_body rule.

  response        → Did the section describe response codes or response data structures?
                    If yes → must have one rule per response code, wildcard range (4XX, 5XX),
                    or the "default" catch-all error response.
                    Check each code mentioned: 200, 201, 204, 4xx/5xx, default, etc.

  path_parameter  → Did the section describe path variables (e.g. {{id}}, {{version}}, {{className}})?
                    If yes → must have one path_parameter rule per variable.

  query_parameter → Did the section describe named query parameters (not their absence)?
                    If yes → must have one query_parameter rule per named parameter.

  schema_property → Did the section describe data model attributes, fields, enumerations,
                    required fields, or schema composition (oneOf/anyOf/allOf)?
                    If yes → must have one schema_property rule per attribute or constraint.

  security_scheme → Did the section describe authentication or security requirements?
                    If yes → must have at least one security_scheme rule.

Go through this checklist for every rule_type. Do not skip any.

A rule is MISSING if ALL of the following are true:
  1. The section content clearly implies it should exist as an OpenAPI construct
  2. It is NOT present in current_rules
  3. It is NOT present in already_validated_rules
  4. It is NOT already covered by the section's inherited base (see below)

INHERITED FIELDS ARE NOT MISSING — read the pre-analysis verdict below. When the
verdict says this section is a schema with an allOf base and lists inherited fields,
those fields are ALREADY PROVIDED by the base via the single allOf $ref. They are
NOT missing, even though the section's attribute table lists them (3GPP tables list
inherited and own attributes together). NEVER report an inherited field as a missing
schema_property or 'required'. Only the type's OWN attributes (those NOT in the
inherited list) can be missing. Reporting an inherited field as missing forces the
Extractor to re-emit a rule the Validator will only discard again — do not do it.

Do NOT suggest:
  - Rules already covered by current_rules or already_validated_rules
  - Any inherited field named in the pre-analysis verdict (covered by the allOf base)
  - Rules the pre-analysis verdict shows belong to ANOTHER section (this section only
    references a type/operation whose definition lives elsewhere)
  - Rules about the absence of a construct (e.g. 'no query parameters')
  - Vague or non-actionable rules
  - Rules that duplicate what is already extracted under a different phrasing

For previous_missing_rules: check whether each one was resolved by current_rules.
Only carry forward what is genuinely still absent.

Be specific and actionable. Each missing rule must be a direct instruction
to the Extractor describing exactly what to extract and how to map it.

TURNING DECLARED GAPS INTO QUERIES (context_queries):

The Extractor also reports, in missing_context, the facts it needed and could not
find. A DECLARED GAP IS NOT A MISSING RULE, and the two have opposite remedies:

  missing rule — a rule the Extractor should have emitted from what it already
                 had. It knows the rule is owed; it simply did not produce it.
                 Remedy: an instruction. List it in missing_rules.

  declared gap — a FACT it did not have. The rule it emitted is mapped wrong
                 BECAUSE of that, and telling it again to emit the rule changes
                 nothing: on the next attempt it will still lack the fact and
                 make the same mapping. Remedy: fetch the fact. Write a query.

Answering a gap with a missing_rules instruction is therefore the one response
that guarantees the next attempt is as stuck as this one. Judge every gap for a
query of its own.

A candidate rule never answers a gap. Candidates say WHICH rule belongs here; a
gap is the fact needed to MAP it. "PUT request body supports the Resource data
structure" is a candidate — it does not say the media type to write into
requestBody.content, which is precisely what that gap asked for.

The Extractor's word that something is missing does not settle it. Check each gap
against what is in front of you — the section content, the candidate rules, the
contexts and verdict above — and decide for yourself.

Emit NO query for a gap when:
  - the value is a standard OpenAPI one, and the Extractor is expected to take it
    from the OpenAPI reference it already has: the media type of a request body,
    the JSON Schema form of a declared type. A 3GPP clause routinely leaves these
    unsaid, so the Extractor may report one as a gap — it is not. Confirm the
    clause names no value of its own, and skip: the answer is the standard value,
    not something to retrieve;
  - the fact is a modelling choice the target YAML makes, written down in no
    specification;
  - the gap names a topic ("more about callbacks") rather than one fact, so no
    query could answer it;
  - the section content above already states it.

Write the query for what is left: a fact only this document or another 3GPP spec
holds — which operation owns a body, which resource a URI belongs to, what a
referenced type contains.

THEN WRITE THE QUERY AS THE ANSWER WOULD READ, not as the gap reads.

  Ask yourself: if a clause somewhere states this fact, what words does IT use?
  Write those. Never describe the absence — retrieval matches meaning, so a query
  built from the complaint ("this clause does not name the owning operation")
  lands on the very clauses that also fail to name it.

  AIM AWAY FROM THIS SECTION. The one clause guaranteed not to hold the answer is
  the one the Extractor just read — it is where the gap came from. A query built
  out of this section's own wording retrieves this section back, and the attempt
  learns nothing. So drop what identifies it: its clause number, its title, the
  phrasing of its tables. Keep only the thing being looked for, and add the words
  that would appear AROUND the answer elsewhere — a header, an example, a
  different table's column.

  Gap  : "which operation the callback anchors on is not stated here"
  Poor : "POST notificationTarget callback anchor owning operation"
         -> that is this section's own subject; it comes straight back.
  Good : "notificationRecipientAddress attribute in the subscription request body"
         -> names the FIELD the owning operation carries, so the clause defining
            that operation is what matches.

  ONE FACT PER QUERY. Two facts in one query retrieve whatever matches the mix,
  which is often neither. If two gaps need two facts, write two queries.

CHOOSING sources — every gap has two halves:

  WHICH value does this service use?   -> 'spec' or 'corpus'
      Any answer that would have to name something out of THIS document — a
      type, an operation, a path, a field, a media type it uses — can only come
      from the specifications.
      'spec'   when this document defines it, in a clause other than this
               section: a sibling resource or type clause, a mapping table, a
               message example.
      'corpus' when this document refers to it but defines it elsewhere, by spec
               number or by name (a type from TS 28.623, a definition in
               TS 32.158).
      These two are alternatives, never a pair: a type is either defined here or
      imported, not both.

  HOW is that written in OpenAPI?      -> 'openapi'
      The shape of the construct that the value goes into. A reliable test: the
      OpenAPI specification was written without any knowledge of 3GPP, so it can
      only answer what would make sense for ANY API. The moment the answer you
      want would contain a name from this document, 'openapi' is not holding it.

  So pair them: 'openapi' plus whichever of 'spec'/'corpus' holds the value. Use
  one source alone only when the gap plainly has no second half.

  Gap    : "the media type for the PUT request body is not stated, so the
            Resource request-body row could not be mapped"
  Query  : none — skip this gap.
      -> The clause names no media type of its own, and none is needed from the
         specifications: this is the standard value the Extractor takes from the
         OpenAPI reference it already has. Retrieval would return what it holds
         already. Skip, and it fills the rule on its own.

  Gap    : "which operation's request body carries the consumer's notification
            address — needed to anchor the callback on it"
  Query  : "notificationRecipientAddress attribute in the subscription request
            body"
  sources: ['spec', 'openapi']
      -> 'spec' names the operation, by looking for the ADDRESS FIELD rather
         than for the notification: the clause that owns the field is the one
         that owns the operation. 'openapi' gives the callback's shape and the
         runtime expression that keys it.

  Gap    : "the fields of NotificationHeader are not given here"
  Query  : "NotificationHeader type definition href notificationId eventTime
            systemDN"
  sources: ['corpus', 'openapi']
      -> 'corpus' has the imported type's fields; 'openapi' how a referenced
         schema is composed into this one.

When no gaps were declared, context_queries is empty.
"""

_SECTION_USER = """\
Section ID: {section_id}

Section content:
{section_content}

Pre-analysis verdict for this section (its kind, whether it is a schema home, its
allOf base, and the INHERITED fields that the base already provides — never report
those as missing):
{section_verdict}

Pre-scoped candidate rules for this section (the positive scope: what this section
is supposed to emit, anchored in its text). Judge "missing" against THIS scope, not
the raw attribute table — anything outside it is not this section's to produce:
{candidate_rules}

Rules another section already owns (Phase-3 skip) — NEVER list any of these as
missing here; their home is elsewhere:
{skip_rules}

Rules the Validator ALREADY DISCARDED in this section (rejected on purpose — NEVER
list any of these as missing; doing so loops them back only to be discarded again):
{discarded_rules}

Rules the Validator is CORRECTING in this section (they EXIST, they just need a fix
— so they are NOT missing). Never list any of these targets as a missing rule; the
correction already covers them:
{corrections}

Rules extracted and reflected in this iteration:
{current_rules}

Rules already validated in previous iterations (do NOT suggest these as missing):
{already_validated_rules}

Rules identified as missing in the previous iteration (check if now resolved):
{previous_missing_rules}

Gaps the Extractor declared for this section (its missing_context — empty when it
declared none). These are NOT missing rules: a missing rule is one it should have
emitted from what it had, and the fix is to instruct it to emit that rule. A gap
is a FACT it did not have, which is why the rule it did emit is mapped wrong — and
the fix is to fetch the fact. Repeating a gap back as a missing rule instruction
leaves the next attempt exactly as stuck as this one, because the Extractor
already knew what it owed; what it lacked was the fact. Each of these belongs in
context_queries, unless one of the skip conditions applies:
{declared_gaps}

Two separate answers, then. First: what is still genuinely missing from this
section? (Exclude inherited fields, rules owned by another section, skipped rules,
anything already discarded, and any target already being corrected.) Second: which
of the gaps above are worth a lookup, and what query would answer each?
"""

reflector_section_prompt = ChatPromptTemplate.from_messages([
    ("system", _SECTION_SYSTEM),
    ("human",  _SECTION_USER),
])
