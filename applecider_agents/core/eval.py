"""Evaluate the triage on lists with known outcomes and compare with the plain energy ranking."""
from __future__ import annotations
import pandas as pd, numpy as np
from . import config as C
from .loop import triage
from .data import benchmark_file


def object_list(name: str, n: int | None = None):
    if name == "ood":            # the benchmark's out-of-taxonomy hold-out (public flare repo), family truth
        a = pd.read_csv(benchmark_file("splits/anomaly.csv"))
        truth = np.where(a.group == "ambiguous", "nova", a.bts_type) if "group" in a else a.bts_type
        df = pd.DataFrame(dict(obj_id=a.obj_id, truth=truth, kind="ood"))
    elif name == "test":         # classified test objects: the agent should call these ordinary/already_classified
        t = pd.read_csv(benchmark_file("splits/test.csv")).sample(frac=1, random_state=0)
        df = pd.DataFrame(dict(obj_id=t.obj_id, truth=t.cls, kind="known"))
    elif name.endswith(".csv"):  # any list of your own: columns obj_id and (optionally) truth
        t = pd.read_csv(name)
        df = pd.DataFrame(dict(obj_id=t.obj_id, truth=t.truth if "truth" in t else "", kind="custom"))
    else:
        raise ValueError(f"{name}: use ood, test, or a CSV path")
    return df.head(n) if n else df


def run(model: str, name: str, n: int | None = None, mock: bool = True, out=None):
    """Benchmark lists (ood, test) are run blind: their objects have a spectroscopic type in the
    catalogue, so the catalogue and TNS tools are hidden and the verdict rests on the data alone."""
    df = object_list(name, n); rows = []; blind = name in ("ood", "test")
    for r in df.itertuples():
        try:
            rep = triage(model, r.obj_id, mock=mock, blind=blind)
            rows.append(dict(obj_id=r.obj_id, truth=r.truth, verdict=rep.verdict, priority=rep.priority, label=rep.flare_label,
                             set="|".join(rep.prediction_set), credibility=rep.credibility, energy_pct=rep.energy_percentile, summary=rep.summary))
        except Exception as e:
            rows.append(dict(obj_id=r.obj_id, truth=r.truth, verdict="ERROR", priority=-1, label="", set="", credibility=np.nan, energy_pct=np.nan, summary=str(e)[:200]))
        print(f"  {r.obj_id} {r.truth:<14} -> {rows[-1]['verdict']:<22} p{rows[-1]['priority']} {rows[-1]['label']}", flush=True)
    res = pd.DataFrame(rows)
    out = out or (C.LOG_DIR / f"eval_{model}_{name}_{'rules' if mock else 'llm'}.csv"); C.LOG_DIR.mkdir(parents=True, exist_ok=True); res.to_csv(out, index=False)
    print(f"\nwrote {out}")
    if name == "ood":
        flagged = res.priority >= 1
        print(f"out-of-taxonomy objects flagged (priority >= 1): {flagged.sum()}/{len(res)}; by family:")
        print(res.groupby("truth")["priority"].apply(lambda p: f"{(p >= 1).sum()}/{len(p)}").to_string())
        k = int(flagged.sum()); top_e = res.sort_values("energy_pct", ascending=False).head(k)
        print(f"energy ranking alone at the same budget ({k} objects) would flag {len(top_e)} (trivially all): the useful comparison is on a mixed stream, see eval 'test'")
    if name == "test":
        print("verdicts on classified objects:", res.verdict.value_counts().to_dict())
    return res
