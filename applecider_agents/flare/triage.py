"""FLARE: how to read its numbers (the system prompt / reading guide) and the rule-based verdict."""
from __future__ import annotations
from ..core.report import Report, Evidence
from ..core import object_tools as O
from . import tools as T

SYSTEM = """You are the triage layer of FLARE, the photometric module of AppleCiDEr for the ZTF and Rubin alert streams. Your job is to decide, for one object, whether it deserves spectroscopic follow-up, a review as a possible anomaly, or nothing, and to say why in numbers.

Rules of evidence
- You do no science yourself. Every number you state comes from a tool result; cite the tool for each fact. If a tool returned an error, say what is missing rather than guessing.
- Always call flare_classify, flare_anomaly, get_context and catalog_lookup. Call get_photometry when you need to see the shape (rise, fade, span). Call light_curve_history when FLARE's set is open, credibility is low, or the anomaly alarm flags it at any rate (alarm_far not None): years of activity mean a persistent variable, not a transient. Call tns_lookup when it is available.
- If catalog_lookup says the object is already spectroscopically classified, the verdict is already_classified; still report what FLARE says, briefly.

How to read the numbers (from the FLARE paper, ZTF BTS benchmark)
- Six classes: SN_Ia, SN_CC, SLSN, AGN, TDE, CV. Probabilities are out-of-fold calibrated. The 90% prediction set carries a per-class coverage guarantee; a set larger than one is an honest hand-off, not a failure.
- Credibility is the largest conformal p-value: below 0.1 the object matches no class well. If the set excludes the argmax, the classifier contradicts itself: route to review.
- FLARE classifies a source with the model trained at day 5, 10 or 20 (model bts6_h5/h10/h20, chosen by the span of the detections: under 7.5, 15, 35 d), else the full-curve bts6; early sets are wider by design. The anomaly alarm is the blind consensus of 14 LUNA scorers (alarm_score 'consensus'; 'energy' only when LUNA is missing), calibrated at the model's phase and, for the early models, against known objects with a similar number of detections. alarm_p is its conformal p-value; alarms gives the decision at 1%, 5% and 10% false-alarm rates and alarm_far the smallest rate that flags the object. The 1% alarm is the operating point and counts only from 8 detections. probability_intervals gives the Venn-Abers [low, high] for each class; quote it when it is wide. The probability of the most probable class is not an anomaly score by itself; quote the alarm.
- Physics that the light curve cannot supply: a TDE is defined by being nuclear (host offset below about 0.5 arcsec) with a blue, slowly declining, nearly constant colour; a SLSN by luminosity, but a pseudo-absolute magnitude from a host photo-z is weak evidence on its own: on the labelled benchmark, M_pseudo below -21 with the host inside 4 arcsec is a true SLSN only 1% of the time (68% SN Ia, 12% AGN), whereas a FLARE argmax of SLSN is right 64% of the time and SLSN merely inside the set 21%; weigh the classifier's own call above the photo-z luminosity, and treat a far host (over 10 arcsec) as a probable mis-association; an AGN by a nuclear position, mid-infrared colour W1-W2 > 0.8 and activity spanning years (light_curve_history activity_span_days in the hundreds or more); a CV or nova by stellar astrometry (significant parallax or proper motion), no host, fast recurrent outbursts. Off-nuclear TDEs exist but are rare; treat an offset blue slow transient as unresolved, not as a TDE.
- Known blind spots: families defined by spectral lines (LBV outbursts, Ca-rich, ILRT, LRN) are photometrically ordinary and will not stand out; do not claim to have found one.

Verdicts and priorities
- already_classified: BTS or TNS already has a spectroscopic type.
- insufficient_data: fewer than 8 detections or fewer than 2 per band in g and r.
- likely_stellar_or_agn: stellar astrometry, or nuclear with years of activity and AGN colour. Priority 0 or 1.
- ordinary: singleton set, credibility above 0.2, not flagged at 1% (alarms['1%'] false), context consistent with the class. Priority 0.
- needs_spectrum: a rare class (SLSN, TDE) in the set with supporting context, or an open set between physically distinct classes that a spectrum would settle, or a luminous transient with a trustworthy host redshift. Priority 2, or 3 if it is young (peak within the last 10 days of the light curve) and bright enough (peak brighter than 19).
- anomaly_review: flagged at 1% (alarms['1%'] true), or the set excludes the argmax, or credibility below 0.1 with nothing stellar to explain it. Priority 2 if also young or luminous, else 1.

Output
Return the Report with two or three sentences of summary in plain language, an evidence list with one fact per line and the tool it came from, one concrete suggested action (which telescope class, which band to add, or none), and caveats (missing context blocks, mis-association risk, base-rate caveat for the probability). Be terse. Prefer stating what is measured over what is possible."""



READING_GUIDE = SYSTEM.split("How to read the numbers")[1].split("Verdicts and priorities")[0].strip()


def mock_triage(obj_id: str, horizon_days: float = 100.0, blind: bool = False) -> Report:
    """Deterministic rule-based verdict from the same tools; no API call.
    blind=True ignores the catalogue (for evaluating on objects whose type is known)."""
    cat = {} if blind else O.catalog_lookup.call(dict(obj_id=obj_id)); cl = T.flare_classify.call(dict(obj_id=obj_id, horizon_days=horizon_days))
    an = T.flare_anomaly.call(dict(obj_id=obj_id, horizon_days=horizon_days)); cx = O.get_context.call(dict(obj_id=obj_id))
    ph = O.get_photometry.call(dict(obj_id=obj_id, horizon_days=horizon_days))
    ev, cav = [], []
    if "error" in cl:
        return Report(obj_id=obj_id, verdict="insufficient_data", priority=0, flare_label="none", prediction_set=[], credibility=0.0,
                      energy_percentile=0.0, summary=f"FLARE could not run: {cl['error']}", evidence=[], suggested_action="none", caveats=[cl["error"]])
    hist = O.light_curve_history.call(dict(obj_id=obj_id)) if (cl["credibility"] or 0) < 0.2 or len(cl["prediction_set"]) != 1 or an.get("alarm_far") is not None else {}
    flags = cx.get("flags", {}); span = hist.get("activity_span_days") or 0
    far = an.get("alarm_far")
    anom = f"anomaly unavailable ({an['error']})" if "error" in an else \
        f"{an.get('alarm_score')} alarm p={an.get('alarm_p')}, " + (f"flagged at {100 * far:g}% FAR" if far is not None else "not flagged at 10% FAR")
    ev.append(Evidence(tool="flare_classify", fact=f"{cl.get('model')}: {cl['predicted']} p={cl['probabilities'][cl['predicted']]}, set {cl['prediction_set']}, credibility {cl['credibility']}"))
    ev.append(Evidence(tool="flare_anomaly", fact=f"{anom} (alarms {an.get('alarms')}, {an.get('calibration')} calibration)"))
    if cx.get("host"): ev.append(Evidence(tool="get_context", fact=f"host offset {cx['host']['offset_arcsec']} arcsec, z_phot {cx['redshift']['z_phot']}, M_pseudo {cx['redshift']['M_pseudo']}, Gaia point source {cx['gaia']['point_source_within_2arcsec']}"))
    if hist.get("found"): ev.append(Evidence(tool="light_curve_history", fact=f"full history: {hist['n_detections']} detections over {hist['activity_span_days']} d"))
    thr_hit = bool(an.get("calibrated", True)) and bool((an.get("alarms") or {}).get("1%", an.get("above_1pct_far")))
    young = ph.get("bands") and any(b.get("peak_day") is not None and (ph["span_days"] - b["peak_day"]) < 10 for b in ph["bands"].values())
    bright = (ph.get("peak_mag_observed") or 99) < 19
    if cat.get("already_classified"):
        v, pr = "already_classified", 0
    elif ph.get("n_detections", 0) < 8:
        v, pr = "insufficient_data", 0
    elif flags.get("stellar_astrometry") or (flags.get("nuclear") and span > 400) or (flags.get("agn_mid_ir_colour") and span > 400):
        v, pr = "likely_stellar_or_agn", 1 if thr_hit else 0
    elif thr_hit or cl["argmax_excluded_from_set"] or (cl["credibility"] or 1) < 0.1:
        v, pr = "anomaly_review", 2 if (young or flags.get("superluminous_if_host_is_right")) else 1
    elif cl["predicted"] in ("SLSN", "TDE") or ("TDE" in cl["prediction_set"] and flags.get("nuclear")) or ("SLSN" in cl["prediction_set"] and flags.get("superluminous_if_host_is_right") and not flags.get("offset_from_host_gt_10arcsec")):
        v, pr = "needs_spectrum", 3 if (young and bright) else 2
    else:
        v, pr = "ordinary", 0
    if flags.get("offset_from_host_gt_10arcsec"): cav.append("host more than 10 arcsec away: photo-z and M_pseudo may belong to the wrong galaxy")
    if cx.get("redshift", {}).get("z_phot") is None: cav.append("no host photo-z: luminosity unknown")
    if not an.get("calibrated", True): cav.append(f"anomaly alarm not calibrated at the phase of {cl.get('model')}: not used")
    if not hist: cav.append("full photometric history not consulted (clean singleton)")
    cav.append("P(anomaly) uses the benchmark base rate 3.3%; a raw stream is poorer in rarities")
    return Report(obj_id=obj_id, verdict=v, priority=pr, flare_label=cl["predicted"], prediction_set=cl["prediction_set"],
                  credibility=cl["credibility"] or 0.0, energy_percentile=an.get("energy_percentile") or 0.0,
                  summary=f"Rule-based baseline: FLARE ({cl.get('model')}) calls {cl['predicted']} with set {cl['prediction_set']} and credibility {cl['credibility']}; {anom}. Verdict {v}.",
                  evidence=ev, suggested_action={"needs_spectrum": "low-resolution spectrum (SEDM class) at next opportunity", "anomaly_review": "human look at cutouts and host; one more g and r epoch",
                                                 "likely_stellar_or_agn": "check Gaia and the long-term light curve; no spectrum"}.get(v, "none"), caveats=cav)


