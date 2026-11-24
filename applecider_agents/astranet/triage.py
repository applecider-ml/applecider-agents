"""ASTRANet: how to read its numbers (the system prompt / reading guide) and the rule-based verdict."""
from __future__ import annotations
from typing import ClassVar
from ..core.report import Report, Evidence
from ..core import object_tools as O
from . import tools as T

RARE_TRAINED = ("SLSN", "TDE")


class SpectralReport(Report):
    """The shared Report, read for a spectroscopic model: `flare_label` holds the ASTRANet class,
    `prediction_set` the classes predicted by at least one spectrum, `credibility` the novelty
    p-value and `energy_percentile` the percentile of the novelty score among ordinary objects."""

    ORIGIN: ClassVar[str] = "ASTRANet-agent"      # "ASTRANet" belongs to the ASTRANet analysis service

    def annotation(self) -> dict:
        return dict(verdict=self.verdict, priority=self.priority, astranet_class=self.flare_label, classes_over_spectra=self.prediction_set,
                    novelty_p=self.credibility, novelty_percentile=self.energy_percentile)

    def comment(self) -> str:
        s = "|".join(self.prediction_set)
        head = f"[ASTRANet] {self.verdict.replace('_', ' ')} (priority {self.priority}): {self.flare_label}, classes over spectra {{{s}}}, novelty p {self.credibility:.3f}."
        return head + " " + self.summary + (f" Action: {self.suggested_action}" if self.suggested_action else "")


SYSTEM = """You are the triage layer of ASTRANet, the spectroscopic module of AppleCiDEr. Your job is to decide, for one object that already has at least one spectrum, whether the spectroscopic class can be trusted, whether the object deserves a review as a possible new kind of transient, or whether another spectrum or a redshift is needed, and to say why in a way an astronomer can check.

Rules of evidence
- You do no science yourself. Every number you state comes from a tool result; cite the tool for each fact. If a tool returned an error, say what is missing.
- Always call get_spectra, astranet_classify, astranet_novelty and catalog_lookup. When the source has a redshift and the class is SLSN, TDE or AGN, or the top probability is below 0.6, call astranet_classify again with use_redshift=false and report whether the class changes. Ignore horizon_days: the spectroscopic tools use every spectrum of the source.
- If catalog_lookup says the object is already spectroscopically classified, the verdict is already_classified; still report what ASTRANet says, briefly, and whether it agrees.

How to read the numbers (from the ASTRANet paper; 1,666 test objects, 60 rare objects in nine classes)
- Seven classes: Ia, Ibc, SLSN, II, AGN, CV, TDE. The service has no spectroscopic phase. Accuracy is 88.5% with the redshift and 86.2% from the spectrum alone; the redshift matters most for SLSN, whose recall falls from 83% to 67% without it. A class that changes when the redshift is withheld rests on the redshift.
- The object probability is the mean over the spectra of the source. Spectra that disagree with the object class, or a member agreement well below one, mark a weak call. Removing the 10% least confident spectra raises accuracy from 89.0% to 93.7%, so a low top probability is informative.
- Novelty is one p-value per object against ordinary objects of the same redshift stratum, with alarms at 1, 5, 10 and 15%. The alarm rate is the fraction of ordinary objects that alarm (measured 1.3 / 4.8 / 10.0 / 14.6%). Rare objects of a class the detector never saw alarm 52 / 70 / 84 / 89% of the time at those rates.
- An alarm at 1% is the strong one: FBOTs (84%), afterglows (86%) and QSOs (63%) are recovered there, and 0.1% of SN Ia alarm. Calcium-rich transients (1%) and ILRTs (0%) are missed at 1% and appear only at the wider rates.
- Rare trained classes alarm often: at 1%, 18% of SLSN, 17% of AGN and 9% of TDE alarm; at 15%, 95% of SLSN, 92% of TDE and 87% of AGN alarm, with 47% of Ibc and 29% of II. An alarm at 10 or 15% on an object classified SLSN, TDE or AGN therefore says the object belongs to a sparsely populated class, and the class call stands.
- Novae are ordinary for the novelty stage (3 / 9 / 15 / 21% of them alarm); the classifier has no nova class and usually calls them CV.
- AGN, QSO, blazar and BL Lac share continua: a QSO or blazar typically appears as class AGN with a novelty alarm.

Verdicts and priorities
- already_classified: BTS or TNS already has a spectroscopic type.
- insufficient_data: no spectrum of the source could be placed on the model grid.
- anomaly_review: novelty alarm at 1% (priority 2), or at 5% when the class is one of Ia, Ibc, II, CV (priority 1).
- needs_spectrum: here it means another spectrum or a redshift is needed: the spectra disagree on the class, the top probability is below 0.6, or the class is SLSN or TDE without a redshift. Priority 2 for SLSN or TDE, else 1.
- likely_stellar_or_agn: class AGN or CV with probability of at least 0.8 and no alarm at 1%. Priority 0.
- ordinary: a confident, consistent class with no alarm at 1 or 5%. Priority 0.

Output
Return the Report. `flare_label` is the ASTRANet class, `prediction_set` the classes predicted by at least one spectrum, `credibility` the novelty p-value and `energy_percentile` the percentile of the novelty score among ordinary objects (both from astranet_novelty; 1.0 and 0.0 when the novelty stage is unavailable). Give two or three sentences of summary, an evidence list with one fact per line and the tool it came from, one concrete suggested action, and caveats (no redshift, a single spectrum, rejected spectra, novelty stage unavailable)."""

READING_GUIDE = SYSTEM.split("How to read the numbers")[1].split("Verdicts and priorities")[0].strip()

ACTIONS = {"anomaly_review": "human look at the spectrum against the class templates; check the light curve and host with the photometric tools",
           "needs_spectrum": "obtain a redshift or a second spectrum with higher signal to noise",
           "likely_stellar_or_agn": "no further spectrum; check the long-term light curve"}


def _empty(obj_id, why):
    return SpectralReport(obj_id=obj_id, verdict="insufficient_data", priority=0, flare_label="none", prediction_set=[], credibility=1.0, energy_percentile=0.0,
                          summary=f"ASTRANet could not run: {why}", evidence=[], suggested_action="none", caveats=[why])


def mock_triage(obj_id: str, horizon_days: float = 100.0, blind: bool = False) -> Report:
    """Deterministic rule-based verdict from the same tools; no API call. horizon_days is unused
    (every spectrum of the source enters). blind=True ignores the catalogue."""
    cat = {} if blind else O.catalog_lookup.call(dict(obj_id=obj_id))
    sp = T.get_spectra.call(dict(obj_id=obj_id))
    if "error" in sp or not sp.get("n_spectra"):
        return _empty(obj_id, sp.get("error", "the source has no spectrum"))
    cl = T.astranet_classify.call(dict(obj_id=obj_id))
    if "error" in cl:
        return _empty(obj_id, cl["error"])
    nv = T.astranet_novelty.call(dict(obj_id=obj_id)); ok = nv.get("status") == "ok"
    alarm = nv.get("alarm", {}) if ok else {}; cls, p = cl["predicted"], cl["probability"] or 0.0
    has_z = cl["redshift_used"] is not None; agree = cl["spectra_agreeing"] == cl["n_spectra"]
    ev = [Evidence(tool="get_spectra", fact=f"{sp['n_spectra']} spectra from {sp['source']}, redshift {sp['redshift']}"),
          Evidence(tool="astranet_classify", fact=f"{cls} p={p} from {cl['n_spectra']} spectra, {cl['spectra_agreeing']} agreeing, {cl['n_rejected']} rejected; classes over spectra {cl['classes_predicted_by_a_spectrum']}")]
    cav = []
    noz = {}
    if has_z and (cls in RARE_TRAINED + ("AGN",) or p < 0.6):
        noz = T.astranet_classify.call(dict(obj_id=obj_id, use_redshift=False))
        if "error" not in noz:
            ev.append(Evidence(tool="astranet_classify", fact=f"redshift withheld: {noz['predicted']} p={noz['probability']}"))
            if noz["predicted"] != cls: cav.append(f"the class changes to {noz['predicted']} when the redshift is withheld")
    if ok: ev.append(Evidence(tool="astranet_novelty", fact=f"novelty p {nv['p_value']} in stratum {nv['stratum']} ({nv['n_calibration']} calibration objects), strictest alarm {nv['strictest_alarm_rate']}"))
    if cat.get("already_classified"):
        v, pr = "already_classified", 0
    elif alarm.get("1%"):
        v, pr = "anomaly_review", 2
    elif alarm.get("5%") and cls in ("Ia", "Ibc", "II", "CV"):
        v, pr = "anomaly_review", 1
    elif not agree or p < 0.6 or (cls in RARE_TRAINED and not has_z):
        v, pr = "needs_spectrum", 2 if cls in RARE_TRAINED else 1
    elif cls in ("AGN", "CV") and p >= 0.8:
        v, pr = "likely_stellar_or_agn", 0
    else:
        v, pr = "ordinary", 0
    if not has_z: cav.append("no redshift: accuracy 86.2% against 88.5% with it; SLSN recall 67% against 83%")
    if cl["n_spectra"] == 1: cav.append("a single spectrum: no check of consistency between epochs")
    if cl["n_rejected"]: cav.append(f"{cl['n_rejected']} spectra rejected for coverage or quality")
    if not ok: cav.append(f"novelty stage {nv.get('status', 'failed')}: {nv.get('reason') or nv.get('error')}")
    if ok and any(alarm.values()) and not alarm.get("1%") and cls in RARE_TRAINED + ("AGN",): cav.append(f"alarm at {nv['strictest_alarm_rate']} on class {cls}: rare trained classes alarm often at the wider rates (95% of SLSN, 92% of TDE, 87% of AGN at 15%)")
    return SpectralReport(obj_id=obj_id, verdict=v, priority=pr, flare_label=cls, prediction_set=cl["classes_predicted_by_a_spectrum"],
                          credibility=nv["p_value"] if ok else 1.0, energy_percentile=nv["percentile_among_ordinary"] if ok else 0.0,
                          summary=f"Rule-based baseline: ASTRANet calls {cls} with probability {p} from {cl['n_spectra']} spectra ({cl['spectra_agreeing']} agreeing)" + (f"; novelty p {nv['p_value']}, strictest alarm {nv['strictest_alarm_rate']}" if ok else "") + f". Verdict {v}.",
                          evidence=ev, suggested_action=ACTIONS.get(v, "none"), caveats=cav)
