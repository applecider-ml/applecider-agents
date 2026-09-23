"""FLARE-specific tools: the hierarchical classifier, the anomaly energy, and an
inline-photometry entry for clients that already hold the light curve."""
from __future__ import annotations
import math
import numpy as np, pandas as pd
from anthropic.lib.tools import beta_tool
from ..core import config as C, data as DATA
from flare import data as D
from flare.config import BTS6
from flare.skyportal import feature_frame

_CLF = None
_CACHE: dict = {}
BAND = {0: "g", 1: "r", 2: "i"}


def clf():
    global _CLF
    if _CLF is None:
        from flare.hierarchical import HierarchicalFlare
        _CLF = HierarchicalFlare.from_pretrained(BTS6)
    return _CLF


def _f(x, nd=3):
    try:
        v = float(x)
        return None if not math.isfinite(v) else round(v, nd)
    except (TypeError, ValueError):
        return None


def prepared(obj_id: str, horizon_days: float = 100.0):
    """Events, context, feature frame and observed peak for one object (cached per process)."""
    key = (obj_id, float(horizon_days))
    if key not in _CACHE:
        ev = DATA.events(obj_id, horizon_days)
        peak0 = float(23.9 - 2.5 * ev[:, D.COL["logflux"]].max())
        ctx = DATA.context(obj_id, peak_mag=peak0)
        X, peak, _ = feature_frame(ev, {k: v for k, v in ctx.items() if not k.startswith("_")}, None)
        X = X.reindex(columns=clf().feature_names)
        _CACHE[key] = (ev, ctx, X, peak)
    return _CACHE[key]


@beta_tool
def flare_classify(obj_id: str, horizon_days: float = 100.0) -> dict:
    """Run the FLARE hierarchical classifier (six classes: SN_Ia, SN_CC, SLSN,
    AGN, TDE, CV) on the object's light curve plus its catalogue context.
    Returns class probabilities, the argmax, Mondrian conformal p-values, the
    90% prediction set, credibility (largest p-value; low means the object
    matches no class well) and whether the set excludes its own argmax."""
    try:
        _, ctx, X, _ = prepared(obj_id, horizon_days)
        c = clf()
        P = c.predict_proba(X)[0]; pv = c.p_values(X)[0]; s = c.prediction_sets(X, alpha=BTS6.alpha)[0]
        conf = c.conformal_report(X)[0]
        lab = BTS6.classes[int(P.argmax())]
        return dict(obj_id=obj_id, predicted=lab, probabilities={k: _f(v) for k, v in zip(BTS6.classes, P)},
                    prediction_set=list(s), alpha=BTS6.alpha, p_values={k: _f(v) for k, v in zip(BTS6.classes, pv)},
                    credibility=_f(pv.max()), confidence=_f(1 - np.sort(pv)[-2]) if len(pv) > 1 else None,
                    argmax_excluded_from_set=lab not in s,
                    empirical_coverage_of_predicted_class=_f(conf.get("empirical_coverage")),
                    context_used=sorted(k for k in ctx if not k.startswith("_")), context_source=ctx.get("_source"),
                    note="Probabilities are out-of-fold calibrated on the BTS benchmark; the set carries the 90% per-class guarantee.")
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


def _magnitude_matched_threshold(peakmag):
    """99th percentile of energy among classified test objects within +-0.25 mag (None if too few)."""
    if peakmag is None or C.TEST_SCORES is None or not C.TEST_SCORES.exists():
        return None, 0
    t = pd.read_csv(C.TEST_SCORES)
    sel = t[(t.peakmag >= peakmag - 0.25) & (t.peakmag < peakmag + 0.25)].energy
    return (float(np.percentile(sel, 99)), int(len(sel))) if len(sel) >= 100 else (None, int(len(sel)))


@beta_tool
def flare_anomaly(obj_id: str, horizon_days: float = 100.0) -> dict:
    """Blind anomaly score from FLARE: the energy of the classifier's logits in
    the anomaly feature space, its percentile among known-class objects, a
    distribution-free novelty p-value, the likelihood ratio and P(anomaly) at
    the configured base rate. Also gives the 1% false-alarm thresholds, both
    the shipped one and one recalibrated on classified objects of matched
    brightness (fainter, noisier curves have higher energy for reasons that
    are not physics)."""
    try:
        ev, _, X, peak = prepared(obj_id, horizon_days)
        c = clf(); a = c.anomaly_probability(X, base_rate=C.BASE_RATE)[0]
        known = np.asarray(c.anomaly_cal["known_energies"], float)
        thr = float(np.percentile(known, 99))
        thr_m, n_m = _magnitude_matched_threshold(peak)
        return dict(obj_id=obj_id, energy=_f(a["energy"]), energy_percentile=_f(a.get("energy_percentile"), 1),
                    novelty_p=_f(a.get("novelty_p"), 4), likelihood_ratio=_f(a["likelihood_ratio"], 2),
                    p_anomaly_at_base_rate=_f(a["p_anomaly"]), base_rate=C.BASE_RATE,
                    shipped_threshold_1pct_far=_f(thr), above_shipped_threshold=bool(a["energy"] > thr),
                    magnitude_matched_threshold_1pct_far=_f(thr_m), n_matched_known=n_m,
                    above_matched_threshold=None if thr_m is None else bool(a["energy"] > thr_m),
                    note="The base rate is the benchmark's (3.3%); a raw alert stream is far poorer in rarities, so quote the likelihood ratio when the prior is unknown.")
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")




def flare_analyze_photometry(mjd: list[float], filter: list[str], mag: list[float], magerr: list[float], obj_id: str = "obj",
                             redshift: float | None = None, ra: float | None = None, dec: float | None = None,
                             horizon_days: float = 100.0, context: str = "live") -> dict:
    """Run FLARE directly on photometry the caller already holds (e.g. SkyPortal's get_photometry): lists of mjd, filter (ztfg/ztfr/ztfi), mag, magerr; optional redshift, ra, dec. Returns the FLARE analysis-service result (classification, anomaly, context, rule verdict, SkyPortal-shaped annotations and classification)."""
    from flare.skyportal import analyze
    fid = {"ztfg": 1, "ztfr": 2, "ztfi": 3, "g": 1, "r": 2, "i": 3}
    rows = sorted((float(t), fid[f.lower()], float(m), float(e)) for t, f, m, e in zip(mjd, filter, mag, magerr) if f.lower() in fid)
    params = dict(horizon_days=horizon_days, context=context, plot=False)
    if ra is not None and dec is not None:
        params.update(ra=ra, dec=dec)
    out = analyze(rows, redshift, params, resource_id=obj_id, work_dir="/tmp"); out.pop("plot_files", None)
    import json; return json.loads(json.dumps(out, default=float))


TOOLS = [flare_classify, flare_anomaly]
