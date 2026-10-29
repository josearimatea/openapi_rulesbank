# src/graph/state.py
# Defines the shared state that flows through the entire LangGraph pipeline.
#
# HOW IT WORKS:
#   The state is a single dictionary-like object created once at the start of
#   the pipeline and passed through every node. Each node receives the full
#   state, reads the fields it needs, and returns only the fields it updates.
#   LangGraph merges those updates back into the shared state before calling
#   the next node.
#
#   rule_bank_flow.py  → defines HOW nodes connect (graph structure)
#   state.py           → defines WHAT travels between them (shared memory)
#
# FIELD OWNERSHIP (which node writes each field):
#   Reader          → parsed_sections, helper_context, openapi_reference_context
#   ContractBuilder → service_contract
#   Planner   → extraction_plan
#   Extractor → raw_rules
#   Reflector → reflected_rules, section_reflection
#   Validator → validated_rules, validation_errors, validated_rules_by_section
#   Builder   → final_output_path
#   (flow control fields are managed by the graph itself)

import operator
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


def _merge_ref_memory(left: dict, right: dict) -> dict:
    """Reducer for other_spec_ref_memory (right wins on key clash).

    The Extractor's pre-analyzer collects references to types imported from other
    specs (e.g. {"NotificationHeader": "TS 28.623"}) section by section. Each call
    returns only the refs it saw; LangGraph merges them into the running memory
    instead of overwriting it, so a reference declared in an earlier section (a
    'General' clause) stays available to the later sections that use it.
    """
    return {**(left or {}), **(right or {})}


class RuleBankState(TypedDict):

    # -------------------------------------------------------------------------
    # INPUT — set by main.py before the graph starts running
    # -------------------------------------------------------------------------

    # Absolute path to the main 3GPP specification file (e.g., TS 28.312)
    main_doc_path: str

    # Absolute paths to auxiliary 3GPP specification files (e.g., TS 32.160).
    # Empty list if no auxiliary documents are provided.
    auxiliary_doc_paths: list[str]

    # Target OpenAPI YAML document the rules are built for, e.g. "ProvMnS".
    # Used as a label for the legacy YAML and downstream metadata; the document's
    # Annex A is not used as context. Defaults to settings.TARGET_YAML; main.py
    # --target overrides it.
    target_yaml: str

    # Optional path to a legacy (previous-release) OpenAPI YAML file, e.g. a
    # Rel-17 ProvMnS.yaml used as extra context when building Rel-18 rules.
    # Empty string if not provided.
    legacy_yaml_path: str

    # -------------------------------------------------------------------------
    # READER OUTPUT — produced by the Reader Node
    # -------------------------------------------------------------------------

    # List of relevant sections extracted from the main 3GPP document.
    # Each entry is a dict with keys: "section_id", "title", "content".
    # Only sections that match OpenAPI-relevant keywords are kept.
    parsed_sections: list[dict]

    # Sections removed by the reader-level filters (symbolic titles, keyword filter).
    # Each entry: {section_id, title, reason: "symbolic_title" | "keyword_filter"}.
    # Carried to the Builder to include in the output for traceability.
    excluded_sections_reader: list[dict]

    # Sections the Planner received but did not select for extraction.
    # Each entry: {section_id, title}.
    # Written by planner_node so the count is always accurate — even in tests
    # where the extraction_plan is manually overridden after the Planner runs.
    excluded_sections_planner: list[dict]

    # Concatenated summary of auxiliary 3GPP documents.
    # Used as background context by the Planner and Extractor nodes.
    helper_context: str

    # Text of the document's embedded OpenAPI annex, carried ONLY so the external
    # $ref filename cascade can look up the YAML file an imported type lives in.
    # Never fed to the Planner or extraction prompts. Empty when there is no annex.
    annex_text: str

    # Spec number of the document being processed, as the corpus records it
    # ("28532"). Corpus searches leave this spec out (its content comes from its own
    # run-time collection) when settings.EXCLUDE_TARGET_SPEC_FROM_CORPUS is on.
    # "" when the file name has none.
    target_spec_number: str

    # Content retrieved from the local snapshot of the OpenAPI/Swagger
    # specification (data/references/openapi_reference/).
    # Used by the Reflector node for RAG-grounded self-reflection.
    openapi_reference_context: str

    # Name of the temporary Qdrant collection holding the target spec's sections,
    # indexed by the ContractBuilder (tools.rag_tools.index_target_spec_sections) so
    # the Extractor and Reflector can retrieve related sections. Dropped by the
    # Builder at the end of the run. Empty string when spec RAG is unavailable.
    spec_collection_name: str

    # Name of the temporary Qdrant collection holding the rules validated so far.
    # Created by the ContractBuilder and grown by the Validator, which indexes each
    # rule as it passes. Lets the Validator spot a rule that duplicates one already
    # accepted in ANOTHER section — the per-section check never sees those.
    # Dropped by the Builder at the end of the run.
    rules_collection_name: str

    # -------------------------------------------------------------------------
    # CONTRACT BUILDER OUTPUT — produced by the ContractBuilder Node
    # -------------------------------------------------------------------------

    # Semantic scope contract for the target Management Service (ServiceContract
    # as dict). Built once from the spec text, between Reader and Planner.
    # Read by the Planner (to prioritize sections by scope) and the Extractor
    # (to classify each rule's contract_status). Empty dict before the
    # ContractBuilder runs; no other node writes to it afterwards.
    service_contract: dict

    # -------------------------------------------------------------------------
    # PLANNER OUTPUT — produced by the Planner Node
    # -------------------------------------------------------------------------

    # Strategy produced by the LLM after reasoning about the document structure.
    # Contains: section priority list, extraction granularity, focus areas.
    # Example: {"priority_sections": ["6.2", "7.1"], "granularity": "attribute"}
    extraction_plan: dict

    # -------------------------------------------------------------------------
    # EXTRACTOR OUTPUT — produced by the Extractor Node
    # -------------------------------------------------------------------------

    # Raw rules extracted from parsed_sections, guided by extraction_plan.
    # Each entry is a dict with keys: "section_id", "rule_text", "openapi_mapping".
    # Not yet validated — may contain errors or incomplete mappings.
    raw_rules: list[dict]

    # Facts the Extractor needed to map a rule correctly and found in neither the
    # section nor any context it was given — at most 2 per attempt, each a sentence
    # naming the fact and how its absence changed what was emitted (e.g. a
    # notification emitted as a path because the operation to anchor the callback
    # on was unknown). Written by the Extractor, overwritten each attempt.
    # Reported only for now: no node acts on it yet. It is the input for the
    # planned step where the Reflector weighs each gap and turns it into one
    # retrieval query, whose result feeds the section's next attempt.
    missing_context: list[str]

    # -------------------------------------------------------------------------
    # REFLECTOR OUTPUT — produced by the Reflector Node
    # -------------------------------------------------------------------------

    # Answers to the gaps above, keyed by section_id: the text retrieved by the
    # queries the Reflector wrote for them (one per gap it judged answerable,
    # each against the collection that holds the answer — this document, the
    # 3GPP corpus, or the OpenAPI specification). Written by the Reflector, read
    # by the section's NEXT extraction attempt. Overwritten each attempt, so a
    # gap that was answered does not keep being re-fetched.
    retrieved_context: dict[str, str]

    # Rules after Chain-of-Thought self-reflection.
    # Each entry extends raw_rules with: "confidence" (float), "reasoning" (str),
    # "flagged" (bool — True means priority validation is needed).
    reflected_rules: list[dict]

    # -------------------------------------------------------------------------
    # VALIDATOR OUTPUT — produced by the Validator Node
    # -------------------------------------------------------------------------

    # Rules that passed both structural (Pydantic) and semantic (LLM) validation.
    # Uses operator.add so LangGraph appends new validated rules on each iteration
    # instead of overwriting — preserving rules that passed in earlier loop iterations.
    validated_rules: Annotated[list[dict], operator.add]

    # Rules that failed validation, with failure reasons attached.
    # Overwritten each iteration — only the latest iteration's errors are here.
    # Each entry is a ValidationError dict: error_type, stage, section_id, rule, instruction.
    validation_errors: list[dict]

    # Validator summary keyed by section_id — used by Reflector (Fase 2) to avoid
    # suggesting as missing rules that were already validated in previous iterations.
    # Each value: list of compact rule dicts {section_id, rule_type, source_name, rule_text}.
    # Overwritten each iteration.
    validated_rules_by_section: dict[str, list[dict]]

    # Completeness analysis produced by the Reflector (Fase 2), once per section.
    # Each entry: {section_id, missing_rules: list[str], reasoning: str}.
    # Written by Reflector. Read by Validator (Stage 3) as analysis input only.
    # Reset to [] by Builder when advancing to the next section.
    section_reflection: list[dict]

    # Final missing-rules instruction produced by the Validator after its 3-stage process.
    # Each entry: {section_id, missing_rules: list[str]}.
    # Written by Validator. Read by Extractor (on loop-back) and conditions.py.
    # Reset to [] by Builder when advancing to the next section.
    validator_section_reflection: list[dict]

    # Running memory of types imported from OTHER specs, collected by the
    # Extractor's pre-analyzer as it reads each section in document order:
    # {imported_type_name: source_spec}, e.g. {"NotificationHeader": "TS 28.623"}.
    # A 'General' clause declares these before the 'Type' clauses that use them; the
    # merge reducer keeps earlier entries so a later section can resolve a type it
    # embeds (fetch its fields, emit an allOf $ref) without the reference being in
    # its own text. Accumulates across sections; never reset.
    other_spec_ref_memory: Annotated[dict, _merge_ref_memory]

    # Facts a referenced specification states about HOW THINGS RELATE, kept by the
    # Extractor's pre-analyzer alongside the type memory above: which operation
    # creates the object that carries an attribute, which resource hosts an
    # endpoint. A type's fields cannot hold these, and the clause that explains
    # the relationship is rarely the clause that has to express it — the URI
    # clause learns that a consumer creates a subscription, the method clause is
    # where that decides an anchor. Keyed by subject, merged across sections with
    # the same reducer, and read by every later section.
    cross_ref_facts: Annotated[dict, _merge_ref_memory]

    # The Extractor pre-analyzer's verdict for each section it processed, kept so
    # the Reflector and Validator can enforce it instead of re-judging the raw text
    # (a mapping table looks like a schema). Keyed by section_id:
    # {section_kind, emits_schema_properties, base_ref, inherited_fields}. Merged
    # across sections with the same dict reducer. The Extractor writes it; the
    # Reflector flags rules that violate it; the Validator rejects them and asks for
    # a fix. Each node reads ONLY its current section's entry into its prompt — the
    # map accumulates in state, the prompt context stays flat.
    section_analysis_by_id: Annotated[dict, _merge_ref_memory]

    # Held rules the final review discarded (see in_held_review), one entry per
    # rule with the Validator's reason. Kept so a held rule that ends up out of the
    # bank still leaves a trace of what it was and why it was dropped.
    held_review_discards: Annotated[list[dict], operator.add]

    # -------------------------------------------------------------------------
    # BUILDER OUTPUT — produced by the Builder Node
    # -------------------------------------------------------------------------

    # Absolute path to the final rules bank JSON file saved in data/outputs/rules_bank/.
    final_output_path: str

    # -------------------------------------------------------------------------
    # FLOW CONTROL — managed by the graph (conditions.py) and main.py
    # -------------------------------------------------------------------------

    # Maximum number of allowed retry iterations per section before forcing to Builder.
    max_iterations: int

    # Index of the section currently being processed in extraction_plan["sections_to_extract"].
    # Starts at 0. Advanced by the Builder after each section completes.
    current_section_idx: int

    # How many times the Extractor has run for the current section.
    # Incremented by Extractor. Reset to 0 by Builder when advancing to the next section.
    # Compared against max_iterations by conditions.py to decide whether to retry.
    section_iteration_count: int

    # -------------------------------------------------------------------------
    # TRACEABILITY — reasoning and message history
    # -------------------------------------------------------------------------

    # Full message history from all LLM calls across all nodes.
    # The add_messages annotation tells LangGraph to append new messages
    # instead of overwriting the list on each node update.
    messages: Annotated[list, add_messages]


def in_held_review(state: dict) -> bool:
    """True when the plan entry being processed is a final review of held rules.

    Rules held because the origin of their $ref type was not stated, and that no
    section settled, get one last pass at the end of the plan: the Builder appends
    one entry per section that held them, flagged "held_review". There the
    Extractor hands over the held rules instead of extracting the section again,
    and the judges skip the missing-rule checks — only the held rules are judged.
    """
    plan = (state.get("extraction_plan", {}) or {}).get("sections_to_extract", []) or []
    idx = state.get("current_section_idx", 0) or 0
    return idx < len(plan) and bool(plan[idx].get("held_review"))
