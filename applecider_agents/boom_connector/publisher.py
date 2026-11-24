"""Publisher and UI formatter for pushing agent results back into the BOOM broker.

Formats multi-agent verdicts, conformal sets, and discovery scores as
broker annotations and native BOOM webapp panels.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from .client import BoomConnectorClient
from ..collaboration.schemas import DiscoveryReport


class BoomPublisher:
    """Handles publishing multi-agent reports back to the BOOM broker database."""

    def __init__(self, client: Optional[BoomConnectorClient] = None):
        self.client = client or BoomConnectorClient()

    def format_annotation(self, report: DiscoveryReport) -> Dict[str, Any]:
        """Format a DiscoveryReport into a BOOM-compatible annotation document."""
        tags = [
            f"verdict:{report.verdict}",
            f"priority:{report.priority}",
            f"pred:{report.flare_label}"
        ]
        if report.priority == 3:
            tags.append("urgent:too")
        if report.discovery_potential >= 0.8:
            tags.append("novelty:high")

        return {
            "origin": "applecider-agents-connector",
            "obj_id": report.obj_id,
            "verdict": report.verdict,
            "priority": report.priority,
            "discovery_potential": report.discovery_potential,
            "credibility": report.credibility,
            "flare_label": report.flare_label,
            "prediction_set": report.prediction_set,
            "scientific_hypothesis": report.scientific_hypothesis,
            "suggested_action": report.suggested_action,
            "tags": tags,
            "conflicts_resolved": report.conflicts_resolved
        }

    def render_boom_tile_card(self, report: DiscoveryReport) -> str:
        """Render an HTML card formatted with BOOM frontend design tokens."""
        # BOOM design thresholds: 0.7 green, 0.4 amber, else red
        green, amber, red = "#10b981", "#f59e0b", "#ef4444"
        disc_color = green if report.discovery_potential >= 0.7 else (amber if report.discovery_potential >= 0.4 else red)

        return f"""<div style="background: oklch(0.21 0.006 285.885); border: 1px solid oklch(1 0 0 / 10%); border-radius: 8px; padding: 12px; color: oklch(0.985 0 0); font-family: ui-monospace, sans-serif; font-size: 12px;">
  <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid oklch(1 0 0 / 10%); padding-bottom: 6px; margin-bottom: 8px;">
    <span style="font-weight: bold; color: #58a6ff;">[BOOM CONNECTOR] {report.obj_id}</span>
    <span style="background: {disc_color}; color: #000; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 10px;">DISCOVERY {report.discovery_potential:.2f}</span>
  </div>
  <div style="margin-bottom: 6px;">
    <strong>Verdict:</strong> {report.verdict} (Priority {report.priority}) | <strong>Pred:</strong> {report.flare_label} {{{", ".join(report.prediction_set)}}}
  </div>
  <div style="color: oklch(0.705 0.015 286.067); margin-bottom: 6px;">
    {report.scientific_hypothesis}
  </div>
  <div style="font-size: 11px; color: #38bdf8;">
    Action: {report.suggested_action}
  </div>
</div>"""
