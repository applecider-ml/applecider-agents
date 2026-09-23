"""The structured verdict the agent must return, and how it renders."""
from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field

Verdict = Literal["ordinary", "needs_spectrum", "anomaly_review", "likely_stellar_or_agn", "already_classified", "insufficient_data"]


class Evidence(BaseModel):
    tool: str = Field(description="tool the fact came from")
    fact: str = Field(description="one measured fact, with its number")


class Report(BaseModel):
    obj_id: str
    verdict: Verdict
    priority: int = Field(ge=0, le=3, description="0 ignore, 1 archive check, 2 spectrum if time allows, 3 spectrum tonight")
    flare_label: str
    prediction_set: List[str]
    credibility: float
    energy_percentile: float
    summary: str = Field(description="two or three sentences; every number must come from a tool result")
    evidence: List[Evidence]
    suggested_action: str
    caveats: List[str]

    def comment(self) -> str:
        """The SkyPortal comment: short, prefixed, numbers only from tools."""
        s = "|".join(self.prediction_set) or "{}"
        head = f"[FLARE] {self.verdict.replace('_', ' ')} (priority {self.priority}): {self.flare_label}, set {{{s}}}, credibility {self.credibility:.2f}, energy pct {self.energy_percentile:.1f}."
        return head + " " + self.summary + (f" Action: {self.suggested_action}" if self.suggested_action else "")
