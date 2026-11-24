"""Multi-Agent collaborative workflow for transient discovery.

Coordinates specialists and the Lead Scientist to evaluate transient candidates,
resolve classification/context disputes, and formulate scientific hypotheses.
"""
from __future__ import annotations
import json
import time
from typing import Optional
from .schemas import DiscoveryReport
from .lead_scientist import LeadScientist
from ..core import config as C


def collaborate(
    obj_id: str,
    model: str = "flare",
    horizon_days: float = 100.0,
    use_llm: bool = False
) -> DiscoveryReport:
    """Run the multi-agent discovery workflow on an astronomical object.

    Args:
        obj_id: Identifier of the transient (e.g. 'ZTF22aazcdan')
        model: Underlying model suite ('flare', 'astranet', or 'applecider')
        horizon_days: Truncation window for the light curve (days from first detection)
        use_llm: Whether to invoke the LLM debate layer (requires configured API key)

    Returns:
        DiscoveryReport with specialist evaluations, debate transcript, and scientific hypothesis.
    """
    t0 = time.time()
    scientist = LeadScientist(model_name=model)
    report = scientist.run_investigation(obj_id, horizon_days=horizon_days, use_llm=use_llm)

    # Log multi-agent report
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = C.LOG_DIR / "collaboration_reports.jsonl"
    record = {
        "timestamp": time.time(),
        "elapsed_sec": round(time.time() - t0, 2),
        "obj_id": obj_id,
        "model": model,
        "use_llm": use_llm,
        "report": report.model_dump()
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    return report
