"""Batch and stream scanner for the BOOM alert broker.

Retrieves candidates from the broker, adapts their alerts, runs the multi-agent
investigation, and generates a ranked Discovery Leaderboard.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from .client import BoomConnectorClient
from .adapter import adapt_boom_object
from ..collaboration import collaborate, DiscoveryReport


class ScoredCandidate(BaseModel):
    """A broker candidate evaluated by the multi-agent system."""
    obj_id: str
    verdict: str
    priority: int
    discovery_potential: float
    flare_label: str
    credibility: float
    scientific_hypothesis: str
    suggested_action: str
    report: DiscoveryReport


class DiscoveryLeaderboard(BaseModel):
    """Ranked leaderboard of transient discoveries from the broker."""
    timestamp: float = Field(default_factory=time.time)
    total_scanned: int = 0
    high_priority_count: int = 0
    candidates: List[ScoredCandidate] = Field(default_factory=list)

    def summary_table(self) -> str:
        """Render a readable terminal/text table of discoveries."""
        if not self.candidates:
            return "No candidates evaluated."
        lines = [
            f"=== BOOM CONNECTOR: DISCOVERY LEADERBOARD ({len(self.candidates)} objects) ===",
            f"{'OBJ ID':<16} {'VERDICT':<20} {'PRIORITY':<10} {'DISCOVERY':<11} {'LABEL':<10} {'HYPOTHESIS'}",
            "-" * 95
        ]
        for c in self.candidates:
            hypo = (c.scientific_hypothesis[:35] + "...") if len(c.scientific_hypothesis) > 35 else c.scientific_hypothesis
            lines.append(
                f"{c.obj_id:<16} {c.verdict:<20} {c.priority:<10} {c.discovery_potential:<11.2f} {c.flare_label:<10} {hypo}"
            )
        return "\n".join(lines)


class BoomScanner:
    """Scanner orchestrator querying the broker and dispatching multi-agent investigations."""

    def __init__(self, client: Optional[BoomConnectorClient] = None, model: str = "flare"):
        self.client = client or BoomConnectorClient()
        self.model = model

    def scan(
        self,
        object_ids: Optional[List[str]] = None,
        limit: int = 20,
        min_detections: int = 2,
        max_mag: float = 20.0,
        use_llm: bool = False,
        horizon_days: float = 100.0
    ) -> DiscoveryLeaderboard:
        """Run multi-agent scan across broker alerts and rank discoveries."""
        # 1. Collect candidate IDs
        if object_ids is None:
            object_ids = self.client.query_recent_candidates(
                min_detections=min_detections,
                max_mag=max_mag,
                limit=limit
            )

        scored: List[ScoredCandidate] = []
        high_pri = 0

        # 2. Process each object
        for oid in object_ids:
            # Ingest & seed from broker
            adapt_boom_object(oid, client=self.client, seed_core=True)

            # Run multi-agent investigation
            report = collaborate(oid, model=self.model, horizon_days=horizon_days, use_llm=use_llm)

            if report.priority >= 2:
                high_pri += 1

            scored.append(ScoredCandidate(
                obj_id=oid,
                verdict=report.verdict,
                priority=report.priority,
                discovery_potential=report.discovery_potential,
                flare_label=report.flare_label,
                credibility=report.credibility,
                scientific_hypothesis=report.scientific_hypothesis,
                suggested_action=report.suggested_action,
                report=report
            ))

        # 3. Sort by discovery potential descending, then priority descending
        scored.sort(key=lambda x: (x.discovery_potential, x.priority), reverse=True)

        return DiscoveryLeaderboard(
            total_scanned=len(scored),
            high_priority_count=high_pri,
            candidates=scored
        )
