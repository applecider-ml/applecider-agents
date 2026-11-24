"""Anonymised evaluation: the model sees an opaque identifier, never the survey name, the
coordinates or calendar dates of an object, so that it cannot recall a known source.

`register` seeds the photometry, the catalogue context and the archive history of one object
under an anonymous id; every object tool and model tool then works on that id without any
network query. The archive history is reduced to counts and spans (no epochs). `leak_scan`
runs every tool the model can see on an anonymous id and reports any survey name, coordinate
or calendar epoch that would reach the model. Catalogue and TNS tools stay hidden through
`blind=True` in the loop, as for any evaluation on objects with a known type.
"""
from __future__ import annotations
import json, re
import numpy as np
from . import data as DATA

SAFE_HISTORY_KEYS = ("found", "n_detections", "activity_span_days", "n_per_band")
NAME_PATTERNS = (r"ZTF\d{2}[a-z]{3,}", r"\bAT ?20\d{2}[a-z]+", r"\bSN ?20\d{2}[a-z]+", r"\b(19|20)\d{2}-\d{2}-\d{2}")


def anonymous_ids(real_ids, seed: int = 0, prefix: str = "obj_"):
    """A shuffled one-to-one map real id -> prefix + zero-padded index (order carries no information)."""
    real_ids = list(dict.fromkeys(real_ids)); order = np.random.default_rng(seed).permutation(len(real_ids))
    width = max(4, len(str(len(real_ids))))
    return {real_ids[j]: f"{prefix}{i + 1:0{width}d}" for i, j in enumerate(order)}


def sanitise_history(history: dict | None) -> dict:
    if not history or not history.get("found"):
        return dict(found=False)
    return {k: history[k] for k in SAFE_HISTORY_KEYS if k in history}


def register(anon_id: str, events_array, context: dict, history: dict | None = None) -> None:
    """Seed one object under its anonymous id (photometry in days since first detection,
    context without positions, history without epochs)."""
    ctx = {k: v for k, v in dict(context).items() if k.lower() not in ("ra", "dec", "obj_id", "ztfid") and not k.startswith("_")}
    DATA.seed(anon_id, events_array, ctx, history=sanitise_history(history))


def leak_scan(anon_id: str, real_id: str, ra: float | None = None, dec: float | None = None,
              tool_names=("get_photometry", "get_context", "light_curve_history", "flare_classify", "flare_anomaly"),
              model: str = "flare", allowed: set | None = None) -> list[str]:
    """Run the model-visible tools on anon_id and return every leak found (empty list = clean).
    allowed: strings that may match a pattern because they are the same for every object
    (e.g. calibration dates in the model metadata, see `model_metadata_dates`)."""
    from . import object_tools as O, registry
    tools = {t.name: t for t in list(O.OBJECT_TOOLS) + list(registry.get(model).tools)}
    leaks = []
    for name in tool_names:
        out = json.dumps(tools[name].call(dict(obj_id=anon_id)), default=str)
        if real_id in out:
            leaks.append(f"{name}: real id")
        for pat in NAME_PATTERNS:
            for hit in re.finditer(pat, out):
                if hit.group(0) not in (allowed or set()):
                    leaks.append(f"{name}: {hit.group(0)}")
        for x in re.findall(r"-?\d+\.\d+|\b\d{5}\b", out):
            if 40000 < abs(float(x)) < 70000:
                leaks.append(f"{name}: MJD-like number {x}")
        for label, v in (("ra", ra), ("dec", dec)):
            if v is not None and re.search(re.escape(f"{v:.3f}"), out):
                leaks.append(f"{name}: {label} {v:.3f}")
        if '"error"' in out:
            leaks.append(f"{name}: tool error {out[:200]}")
    return leaks


def model_metadata_dates(model_dir) -> set:
    """Calendar dates written in a model directory's JSON files (identical for every object)."""
    from pathlib import Path
    found = set()
    for f in Path(model_dir).rglob("*.json"):
        try:
            found.update(m.group(0) for m in re.finditer(NAME_PATTERNS[3], f.read_text(errors="ignore")))
        except OSError:
            pass
    return found
