# src/prompts/contract_builder_prompts.py

"""
Prompt templates and output schemas for the ContractBuilder Node.

The ContractBuilder runs once per pipeline run, between Reader and Planner, and
makes two LLM calls:

  Call 1 — Locator: reads a lightweight table (section_id | title | first 200
    chars) and returns the section IDs that hold the service's purpose,
    capabilities, and key concepts. Output: SectionLocator (may be all empty).

  Call 2 — Contract: reads the full content of the located sections and builds
    the ServiceContract (from schemas.contract). Output: ServiceContract.

Classes:
    SectionLocator  — Call 1 output (structured); the ServiceContract used for
                      Call 2 lives in schemas/contract.py.

Constants:
    contract_locator_prompt   — Call 1 prompt (section summaries → section IDs)
    contract_extract_prompt   — Call 2 prompt (section content → ServiceContract)
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured output schema — Call 1 (locator)
# ---------------------------------------------------------------------------

class SectionLocator(BaseModel):
    """
    Output of the ContractBuilder locator call (LLM call 1).

    Identifies which section IDs contain the concepts needed to build the
    contract. Empty lists are valid — the ContractBuilder falls back to the
    first few sections when nothing is located.
    """

    purpose_section_ids: list[str] = Field(
        default_factory=list,
        description="Section IDs that contain the general description/overview of the service."
    )
    capabilities_section_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Section IDs that describe what the service supports — "
            "operations, notifications, states, subscriptions, or any other capability type."
        )
    )
    concepts_section_ids: list[str] = Field(
        default_factory=list,
        description="Section IDs that define domain terms central to this service."
    )
    other_services: list[str] = Field(
        default_factory=list,
        description=(
            "Names of the OTHER Management Services present in the document — every "
            "MnS that is NOT the target service. Use the short 'XxxMnS' abbreviation "
            "ONLY when the document actually defines one; many services have no such "
            "abbreviation (e.g. 'Heartbeat', 'Performance assurance') — for those use "
            "the service name exactly as the document presents it. Do not invent an "
            "abbreviation. These services are out of scope for the target contract."
        )
    )
    stage3_clause: str = Field(
        default="",
        description=(
            "The clause-number prefix of the TARGET service's stage 3 block — the "
            "'solution set' that maps the abstract operations to a concrete protocol "
            "(RESTful HTTP: paths, methods, schemas, status codes). Identify it by "
            "titles containing 'solution set', 'RESTful HTTP', or 'Mapping to HTTP...'. "
            "Return only the clause prefix, e.g. '12.1'. This is the block OpenAPI "
            "rules must be extracted from. Empty string if the document has no stage 3."
        )
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_LOCATOR_SYSTEM = """\
You are reading a 3GPP specification that defines SEVERAL Management Services
(MnS), one per top-level clause (e.g. "11.1 Generic provisioning management
service", "11.2 Generic fault supervision management service", "11.3 Performance
assurance"). Your task is to identify the sections that belong to the ONE target
service named by the user — and ONLY that service.

METHOD:
  1. Find the umbrella clause whose title names the target service (e.g. for
     "ProvMnS", the clause titled "...provisioning management service"). Its
     subsections (same clause number prefix, e.g. 11.1.x) hold that service's
     stage-2 definitions.
  2. Select sections that belong to the target service:
     - PURPOSE: the service's introduction/overview section.
     - CAPABILITIES: sections defining its operations, notifications, states, or
       subscriptions. Do not assume the type in advance.
     - KEY_CONCEPTS: sections defining domain terms/data structures for it.

STRICT SCOPE RULES:
  - A section belongs to the target ONLY if the spec places it under the target's
    clause. When a title names another service (fault, alarm, performance,
    heartbeat, streaming, file), it is NOT the target — do not select it.
  - Sub-section titles may be terse (e.g. "Operations and notifications" with an
    empty preview). Judge membership from the clause number in the title, not the
    preview.
  - Precision over coverage: if you are not certain a section is the target
    service's, leave it out. Return empty lists rather than guessing.

ALSO IDENTIFY THE OTHER SERVICES:
  The document defines several Management Services, one per top-level clause.
  In `other_services`, list every service that is NOT the target — read their
  umbrella-clause titles (e.g. "11.2 Generic fault supervision management
  service", "11.4 Heartbeat"). Use the short "XxxMnS" abbreviation only when the
  document defines one; otherwise use the name as written (e.g. "Heartbeat"). Do
  not invent abbreviations. These are recorded as out of scope for the target.

ALSO IDENTIFY THE TARGET'S STAGE 3 BLOCK:
  3GPP specs describe each service in two stages:
    - Stage 2: abstract definitions — operations, input/output parameters,
      notifications — with NO protocol binding (no HTTP method, URI, status code,
      media type, or JSON schema). Titles like "Operations and notifications",
      "Input parameters", "Output parameters".
    - Stage 3: the "solution set" that maps stage 2 onto a concrete protocol
      (RESTful HTTP): resource paths, HTTP methods, schemas, response codes.
      Titles contain "solution set", "RESTful HTTP", or "Mapping to HTTP...".
  OpenAPI rules can only come from stage 3. In `stage3_clause`, return the
  clause-number prefix of the TARGET service's stage 3 block (e.g. "12.1"), or
  "" if the document has no stage 3.
"""

_LOCATOR_USER = """\
Service being documented: {service_name}

Below is a summary of all sections, one per line, in the format:
    section_id | title | first 600 characters of content

The section_id is the FIRST column only (a short number like "5" or "42"),
NOT the clause number embedded in the title (like "11.1.1").

Identify which sections contain PURPOSE, CAPABILITIES, and KEY_CONCEPTS, and
return their section_id — the first-column value — for each category.

SECTION SUMMARIES:
{section_summaries}
"""

_CONTRACT_SYSTEM = """\
You are building a scope contract for ONE 3GPP Management Service (the target).
This contract will be used by downstream pipeline nodes to decide whether rules
extracted from the spec belong to this service or not.

PRINCIPLES:
  - Extract only what is explicitly present in the provided spec text. Do not
    invent anything not present in the text.
  - Precision over coverage: include a capability or concept ONLY when the text
    makes clear it belongs to the target service. When unsure, leave it out — the
    Planner completes the picture later. A small, correct contract is better than
    a large, contaminated one.
  - Every list must contain UNIQUE entries. Never repeat the same value.
  - If a field cannot be determined from the text, return an empty list or string.
"""

_CONTRACT_USER = """\
Service being documented: {service_name}

RELEVANT SPEC SECTIONS:
{relevant_content}

From the text above, extract the scope contract for this service ONLY.
Include something only when the text makes clear it belongs to this service.
Every list must have unique entries — never repeat a value.

For purpose: one sentence describing what THIS service exists to do.

For capabilities:
  - A capability is something the service OFFERS to its consumers — an action it
    performs or exposes: an invocable operation, an emitted notification, a
    subscription. List each one once.
  - Do NOT list conditions, states, or descriptive facts as capabilities. Text
    that merely DESCRIBES a situation (e.g. pre/post conditions, before/after
    states, triggering circumstances, counts, or the internal state model of an
    operation or notification) is NOT a capability — it describes how or when a
    capability behaves, not a capability itself. Skip it.
  - The type of each capability comes from how the spec labels it:
    if the spec says "the following operations", type = operation;
    if it says "the following notifications", type = notification; and so on.
    Use "state" only if the spec genuinely offers a state as a service feature —
    never for a before/after or trigger condition of another capability.
  - If the spec does not label the type explicitly, infer from context.
  - Do not include capabilities the text attributes to a different service.

For key_concepts: unique domain terms/data structures central to THIS service.
For inclusion_keywords: unique terms that are specific to THIS service.
For exclusion_keywords: terms the text attributes to another MnS — use format
  "term → OtherMnS" when the other service is known (e.g. "alarm → FaultMnS").
For out_of_scope_signals: short "term → OtherMnS" pairs, e.g. "threshold → PerfMnS".
"""


contract_locator_prompt = ChatPromptTemplate.from_messages([
    ("system", _LOCATOR_SYSTEM),
    ("human",  _LOCATOR_USER),
])

contract_extract_prompt = ChatPromptTemplate.from_messages([
    ("system", _CONTRACT_SYSTEM),
    ("human",  _CONTRACT_USER),
])
