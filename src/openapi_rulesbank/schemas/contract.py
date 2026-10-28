# src/schemas/contract.py

"""
Pydantic data models for the service scope contract.

The contract defines the semantic scope of the 3GPP Management Service being
processed. It is built once per pipeline run, from the spec text only, and
injected into the Planner and Extractor so both know what the service is and
which rules belong to it.

MODEL HIERARCHY
───────────────
  Capability          — one declared capability of the service
    ├── name          str   — capability name as declared in the spec
    ├── type          str   — operation | notification | state | subscription | other
    └── description   str   — one sentence taken from the spec text

  ServiceContract     — the full scope contract (Extractor/Planner input)
    ├── service_name         str        — name of the Management Service (e.g. "ProvMnS")
    ├── purpose              str        — one sentence: what the service exists to do
    ├── capabilities         list[Capability] — all capabilities declared in the spec
    ├── key_concepts         list[str]  — domain terms central to the service (e.g. MOI, IOC)
    ├── inclusion_keywords   list[str]  — terms that signal a rule belongs to this service
    ├── exclusion_keywords   list[str]  — terms that signal another service ("term → OtherMnS")
    └── out_of_scope_signals list[str]  — short "term → OtherMnS" pairs (e.g. "alarm → FaultMnS")

The SectionLocator model used by the ContractBuilder's first LLM call lives with
the other structured-output schemas in prompts/contract_builder_prompts.py.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Capability(BaseModel):
    """A single capability declared by the Management Service."""

    name: str = Field(
        description="Name of the capability as declared in the spec text."
    )
    type: Literal["operation", "notification", "state", "subscription", "other"] = Field(
        description=(
            "Type of capability as labeled by the spec. "
            "Use what the spec declares — if unlabeled, infer from context."
        )
    )
    description: str = Field(
        description="One sentence describing this capability, taken from the spec text."
    )
    bound_in_stage3: bool = Field(
        default=True,
        description=(
            "Whether the stage 3 solution set this contract is for actually maps "
            "this capability to a protocol — its mapping tables are the authority. "
            "A capability the abstract stage 2 declares but stage 3 does not map "
            "belongs to the service and NOT to the target YAML, so it is kept here "
            "with False rather than dropped: the description is still true, and "
            "every node reading the contract needs to know the difference between "
            "'this service has it' and 'this YAML contains it'. True when the "
            "stage 3 tables map it, or when you could not check."
        )
    )


class ServiceContract(BaseModel):
    """
    Scope contract for a 3GPP Management Service.

    Produced by: ContractBuilder Node (once per pipeline run)
    Consumed by: Planner Node (to prioritize sections by scope)
                 Extractor Node (to classify contract_status per rule)

    Built exclusively from the spec text — no YAML, no external knowledge.
    Every field defaults to empty so a partial or missing contract never blocks
    the pipeline.
    """

    service_name: str = Field(
        description="Name of the Management Service (e.g. ProvMnS, PerfMnS)."
    )
    purpose: str = Field(
        default="",
        description="One sentence describing what this service exists to do."
    )
    capabilities: list[Capability] = Field(
        default_factory=list,
        description=(
            "The capabilities you identified for this service in the sections you "
            "were given — not necessarily every one the service has. A capability "
            "you are unsure belongs here is left out: the list is read as a list of "
            "things that ARE this service's, so a wrong entry misleads every node "
            "downstream, while a missing one only leaves them with less."
        )
    )
    capabilities_complete: bool = Field(
        default=False,
        description=(
            "True only when the sections you were given plainly enumerate the "
            "service's capabilities and you listed all of them. False when they "
            "may hold more than you could identify — which is the usual case, and "
            "tells the reader to treat the list as partial rather than closed."
        )
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="Domain terms central to this service (e.g. MOI, IOC, LDN)."
    )
    inclusion_keywords: list[str] = Field(
        default_factory=list,
        description="Terms that signal a rule belongs to this service."
    )
    exclusion_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Terms that signal a rule belongs to another service. "
            "Format: 'term → OtherMnS' when the target service is known."
        )
    )
    out_of_scope_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Short signals mapping terms to other services. "
            "Format: 'alarm → FaultMnS', 'threshold → PerfMnS'."
        )
    )
    stage_scope: str = Field(
        default="",
        description=(
            "Explains where this service's OpenAPI rules come from: the stage 3 "
            "block (the RESTful HTTP solution set) versus stage 2 (abstract "
            "definitions, not a rule source). Guides the Planner/Extractor to "
            "extract only from stage 3 sections."
        )
    )
    stage3_clause: str = Field(
        default="",
        description=(
            "Clause number of the target service's RESTful HTTP solution set (the "
            "solution-set clause itself, not its parent), as found by the "
            "ContractBuilder's locator. The machine-readable "
            "counterpart of stage_scope: the Planner uses it to keep only stage 3 "
            "sections before reasoning. Empty when the document has no stage 3 block."
        )
    )


def format_contract_summary(service_contract: dict) -> str:
    """
    Renders a ServiceContract dict as a compact text block for LLM prompts.

    Shared by the Planner (to prioritize sections by scope) and the Extractor
    (to classify each rule). Returns "" for an empty/missing contract so callers
    can inject nothing when the ContractBuilder was skipped.
    """
    if not service_contract:
        return ""

    # A capability stage 3 does not bind belongs to the service but not to this
    # YAML, so it is marked rather than listed as if it were extractable.
    cap_names = [
        c.get("name", "") + ("" if c.get("bound_in_stage3", True) else " (not in this YAML)")
        for c in service_contract.get("capabilities", [])
    ]
    summary = (
        "SERVICE CONTRACT\n"
        f"Service: {service_contract.get('service_name', '')}\n"
        f"Purpose: {service_contract.get('purpose', '')}\n"
        # Say whether the list is closed. A reader that takes a partial list for a
        # complete one treats an absence as evidence the service lacks something.
        f"Capabilities ({'complete' if service_contract.get('capabilities_complete') else 'identified so far — there may be more'}): "
        f"{', '.join(cap_names)}\n"
        f"Key concepts: {', '.join(service_contract.get('key_concepts', []))}\n"
        f"Inclusion keywords: {', '.join(service_contract.get('inclusion_keywords', []))}\n"
        f"Exclusion keywords: {', '.join(service_contract.get('exclusion_keywords', []))}\n"
        f"Out-of-scope signals: {', '.join(service_contract.get('out_of_scope_signals', []))}"
    )
    stage_scope = service_contract.get("stage_scope", "")
    if stage_scope:
        summary += f"\nStage scope: {stage_scope}"
    return summary
