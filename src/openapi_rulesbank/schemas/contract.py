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


class OperationMapping(BaseModel):
    """
    One row of the stage 3 'Mapping of operations' table: an IS operation and its
    concrete HTTP binding. Fixed to the 3GPP RESTful solution-set shape; specs
    without a stage 3 REST mapping simply leave operation_map empty.
    """

    is_operation: str = Field(
        description="The IS operation name from the mapping table (e.g. 'getMOIAttributes')."
    )
    http_method: str = Field(
        description="The HTTP method it maps to (e.g. 'GET', 'PUT', 'PATCH')."
    )
    resource_uri: str = Field(
        default="",
        description="The resource URI template it maps to, as written in the table."
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
        description="All capabilities declared by the service in the spec."
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
    operation_map: list[OperationMapping] = Field(
        default_factory=list,
        description=(
            "Canonical mapping of each IS operation to its HTTP binding, taken from "
            "the stage 3 'Mapping of operations' table. Used by the Extractor to "
            "name rules with the correct IS operation (e.g. 'getMOIAttributes' for "
            "an HTTP GET) instead of inventing a name or using the HTTP method."
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

    cap_names = [c.get("name", "") for c in service_contract.get("capabilities", [])]
    summary = (
        "SERVICE CONTRACT\n"
        f"Service: {service_contract.get('service_name', '')}\n"
        f"Purpose: {service_contract.get('purpose', '')}\n"
        f"Capabilities: {', '.join(cap_names)}\n"
        f"Key concepts: {', '.join(service_contract.get('key_concepts', []))}\n"
        f"Inclusion keywords: {', '.join(service_contract.get('inclusion_keywords', []))}\n"
        f"Exclusion keywords: {', '.join(service_contract.get('exclusion_keywords', []))}\n"
        f"Out-of-scope signals: {', '.join(service_contract.get('out_of_scope_signals', []))}"
    )
    stage_scope = service_contract.get("stage_scope", "")
    if stage_scope:
        summary += f"\nStage scope: {stage_scope}"

    op_map = service_contract.get("operation_map", [])
    if op_map:
        lines = [
            f"  {m.get('is_operation', '')} → {m.get('http_method', '')} "
            f"{m.get('resource_uri', '')}".rstrip()
            for m in op_map
        ]
        summary += "\nOperation map (IS operation → HTTP method → URI):\n" + "\n".join(lines)
    return summary
