"""Data models for multi-agent collaboration in AppleCiDEr.

Specialists evaluate an astronomical source from their specific viewpoint
(triage/quality, photometric classification, anomaly/novelty, and host/catalogue context).
The Lead Scientist then debates, reconciles any conflicts, and synthesizes a DiscoveryReport.
"""
from __future__ import annotations
from typing import Dict, List, Literal, Optional, Any, ClassVar
from pydantic import BaseModel, Field
from ..core.report import Report, Verdict, Evidence

SpecialistRole = Literal["triage", "photometry", "anomaly", "context", "spectroscopy"]


class SpecialistAssessment(BaseModel):
    """An evaluation submitted by a specialist agent."""
    role: SpecialistRole = Field(description="The domain of the specialist")
    verdict_candidate: Verdict = Field(description="Candidate verdict from this specialist's perspective")
    status: Literal["ok", "unavailable", "error"] = "ok"
    confidence_kind: str = "heuristic; not a calibrated probability"
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the assessment [0, 1]")
    key_findings: List[str] = Field(default_factory=list, description="Core facts and measurements observed by this agent")
    flags: Dict[str, Any] = Field(default_factory=dict, description="Boolean or numeric diagnostic flags")
    concerns: List[str] = Field(default_factory=list, description="Skeptical checks or potential pitfalls identified")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Selected tool output numbers")


class DebateMessage(BaseModel):
    """A contribution in the multi-agent debate."""
    sender: str = Field(description="Name or role of the contributing agent")
    recipient: Optional[str] = Field(default="all", description="Target recipient or 'all'")
    content: str = Field(description="The scientific argument, challenge, or question")
    cited_evidence: List[Evidence] = Field(default_factory=list, description="Tool-measured numbers cited in the message")


class DiscoveryReport(Report):
    """Enriched report from the collaborative multi-agent investigation.

    Inherits from the core Report so it is 100% compatible with existing SkyPortal
    publish gates, CLI formatting, and serialization, while adding multi-agent debate traces.
    """
    ORIGIN: ClassVar[str] = "AppleCiDEr-collaboration"
    model: str = "flare"
    model_results: Dict[str, Any] = Field(default_factory=dict)
    data_status: Dict[str, str] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    score_kind: str = "heuristic ranking; not a probability of discovery"
    llm_review: Optional[str] = None
    llm_status: Literal["not_requested", "complete", "error"] = "not_requested"
    llm_error: Optional[str] = None

    def comment(self) -> str:
        return (f"[AppleCiDEr] {self.verdict} (priority {self.priority}; models: {self.model}). "
                f"{self.summary} Action: {self.suggested_action}")

    def annotation(self) -> dict:
        out = dict(verdict=self.verdict, priority=self.priority, models=self.model,
                   ranking_score=self.discovery_potential, score_kind=self.score_kind)
        for name, result in self.model_results.items():
            out[f"{name}_status"] = self.data_status.get(name, "unavailable")
            if result.get("predicted") is not None:
                out[f"{name}_class"] = result["predicted"]
        return out

    scientific_hypothesis: str = Field(
        default="",
        description="Physical explanation of the transient event (e.g. FBOT, Kilonova candidate, standard SN Ia, AGN flare)"
    )
    discovery_potential: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Heuristic follow-up ranking, NOT a calibrated discovery probability"
    )
    specialist_assessments: Dict[str, SpecialistAssessment] = Field(
        default_factory=dict,
        description="Individual assessments from each participating specialist"
    )
    debate_transcript: List[DebateMessage] = Field(
        default_factory=list,
        description="The debate and cross-examination history between agents"
    )
    conflicts_resolved: List[str] = Field(
        default_factory=list,
        description="Disagreements between agents that were reconciled by the Lead Scientist"
    )

    def detailed_summary(self) -> str:
        """Render a readable multi-agent synthesis."""
        lines = [
            f"=== MULTI-AGENT DISCOVERY REPORT: {self.obj_id} ===",
            f"Verdict: {self.verdict.upper()} (Priority {self.priority})",
            f"Follow-up ranking: {self.discovery_potential:.2f} ({self.score_kind})",
            f"Data status: {self.data_status}",
            f"Hypothesis: {self.scientific_hypothesis}",
            "",
            "--- Specialist Assessments ---"
        ]
        for role, assess in self.specialist_assessments.items():
            lines.append(f"[{role.upper()}] Candidate: {assess.verdict_candidate} (conf: {assess.confidence:.2f})")
            for f in assess.key_findings:
                lines.append(f"  • {f}")
            if assess.concerns:
                for c in assess.concerns:
                    lines.append(f"  ⚠ {c}")
        if self.conflicts_resolved:
            lines.append("")
            lines.append("--- Conflicts Resolved ---")
            for cr in self.conflicts_resolved:
                lines.append(f"  ⚖ {cr}")
        lines.append("")
        lines.append(f"Action: {self.suggested_action}")
        if self.llm_error:
            lines.extend(["", f"LLM review failed: {self.llm_error}"])
        if self.llm_review:
            lines.extend(["", "--- LLM review (advisory) ---", self.llm_review])
        return "\n".join(lines)
