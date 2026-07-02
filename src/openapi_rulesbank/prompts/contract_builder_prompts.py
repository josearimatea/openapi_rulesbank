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


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_LOCATOR_SYSTEM = """\
You are reading a 3GPP specification that defines a Management Service.
Your task is to identify which sections of the document contain the information
needed to understand what this service does and what it supports.

You are looking for sections that contain:
  - PURPOSE: general description or overview of what this service exists to do
  - CAPABILITIES: what this service supports — operations, notifications, states,
    subscriptions, or any other type of capability. Do not assume the type in advance.
  - KEY_CONCEPTS: domain terms and data structures central to this service

These may or may not exist as dedicated sections. If you cannot find a clear match
for a category, return an empty list for that key — do not guess.
"""

_LOCATOR_USER = """\
Service being documented: {service_name}

Below is a summary of all sections: section_id | title | first 200 characters of content.

Identify which section IDs contain PURPOSE, CAPABILITIES, and KEY_CONCEPTS.

SECTION SUMMARIES:
{section_summaries}
"""

_CONTRACT_SYSTEM = """\
You are building a scope contract for a 3GPP Management Service.
This contract will be used by downstream pipeline nodes to decide whether rules
extracted from the spec belong to this service or not.

Extract only what is explicitly present in the provided spec text.
If a field cannot be determined from the text, return an empty list or empty string.
Do not invent anything not present in the text.
"""

_CONTRACT_USER = """\
Service being documented: {service_name}

RELEVANT SPEC SECTIONS:
{relevant_content}

From the text above, extract the scope contract for this service.

For capabilities:
  - List every operation, notification, state, subscription, or other capability
    the spec declares this service supports.
  - The type of each capability comes from how the spec labels it:
    if the spec says "the following operations", type = operation;
    if it says "the following notifications", type = notification;
    if it says "the following states", type = state; and so on.
  - If the spec does not label the type explicitly, infer from context.

For inclusion_keywords: terms that appear repeatedly and are specific to this service.
For exclusion_keywords: terms that belong to another MnS — use format "term → OtherMnS"
  when the other service is known.
For out_of_scope_signals: short "term → OtherMnS" pairs, e.g. "alarm → FaultMnS".
"""


contract_locator_prompt = ChatPromptTemplate.from_messages([
    ("system", _LOCATOR_SYSTEM),
    ("human",  _LOCATOR_USER),
])

contract_extract_prompt = ChatPromptTemplate.from_messages([
    ("system", _CONTRACT_SYSTEM),
    ("human",  _CONTRACT_USER),
])
