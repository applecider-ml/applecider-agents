"""Specialist agents for collaborative transient investigation.

Each specialist evaluates the object strictly using tool calls from their domain:
- TriageSpecialist: data quality, cadence, long-term history, existing catalog records
- PhotometrySpecialist: FLARE/AppleCiDEr phase-matched classification and conformal sets
- AnomalySpecialist: blind consensus anomaly alarm, false-alarm rates, out-of-distribution metrics
- ContextSpecialist: host galaxy association, photo-z / pseudo-luminosity, Gaia astrometry, WISE infrared
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from .schemas import SpecialistAssessment
from ..core import object_tools as O
from ..flare import tools as F


class TriageSpecialist:
    """Evaluates light-curve quality, long-term baseline, and existing catalog status."""
    role = "triage"

    def assess(self, obj_id: str, horizon_days: float = 100.0, require_photometry: bool = True, blind: bool = False) -> SpecialistAssessment:
        phot = O.get_photometry.call(dict(obj_id=obj_id, horizon_days=horizon_days)) if require_photometry else {}
        hist = O.light_curve_history.call(dict(obj_id=obj_id)) if require_photometry else {}
        cat = {} if blind else O.catalog_lookup.call(dict(obj_id=obj_id))
        tns = {} if blind else O.tns_lookup.call(dict(obj_id=obj_id))

        findings = []
        concerns = ["Catalogue and TNS labels withheld for blind evaluation."] if blind else []
        flags: Dict[str, Any] = {}

        n_det = phot.get("n_detections", 0)
        span = phot.get("span_days", 0.0)
        peak_mag = phot.get("peak_mag_observed")
        findings.append(f"Observed {n_det} detections over {span:.1f} days (peak mag {peak_mag})")

        for name, result in (("photometry", phot), ("history", hist), ("catalogue", cat), ("TNS", tns)):
            if result.get("error") or result.get("available") is False:
                concerns.append(f"{name} unavailable: {result.get('error') or result.get('reason')}")
        typed = [m for m in tns.get("matches", []) if m.get("object_type") not in (None, "", "Other", "Unknown")]
        if typed and not cat.get("already_classified"):
            cat = dict(already_classified=True, spectroscopic_type=typed[0]["object_type"], redshift=typed[0].get("redshift"))

        # 1. Check existing BTS / TNS records
        if cat.get("already_classified"):
            spec_type = cat.get("spectroscopic_type")
            findings.append(f"Already classified in BTS/TNS as '{spec_type}' (z={cat.get('redshift')})")
            return SpecialistAssessment(
                role=self.role,
                verdict_candidate="already_classified",
                confidence=0.99,
                key_findings=findings,
                flags={"already_classified": True, "bts_type": spec_type},
                concerns=concerns + ["Object is already known; not an unclassified discovery."],
                raw_data={"photometry": phot, "catalog": cat, "tns": tns}
            )

        # 2. Check data sufficiency
        if require_photometry and n_det < 2:
            concerns.append("Fewer than 2 detections inside the horizon.")
            return SpecialistAssessment(
                role=self.role,
                verdict_candidate="insufficient_data",
                status="unavailable",
                confidence=0.0,
                key_findings=findings,
                flags={"insufficient_detections": True},
                concerns=concerns,
                raw_data={"photometry": phot}
            )

        # 3. Check archival history for multi-year baseline activity
        hist_span = hist.get("activity_span_days") or hist.get("span_days") or 0.0
        hist_ndet = hist.get("n_detections") or 0
        flags["archive_span_days"] = hist_span
        flags["archive_n_detections"] = hist_ndet

        if hist_span > 365.0:
            findings.append(f"Archival baseline shows activity across {hist_span:.0f} days ({hist_ndet} detections)")
            concerns.append("Multi-year activity suggests variability; inspect the historical light curve before excluding a transient.")
            return SpecialistAssessment(
                role=self.role,
                verdict_candidate="likely_stellar_or_agn",
                confidence=0.88,
                key_findings=findings,
                flags=flags,
                concerns=concerns,
                raw_data={"photometry": phot, "history": hist}
            )

        # Clean transient candidate
        findings.append(f"No known classification found in available catalogues; {n_det} photometric detections.")
        return SpecialistAssessment(
            role=self.role,
            verdict_candidate="ordinary",
            confidence=0.85,
            key_findings=findings,
            flags=flags,
            concerns=concerns,
            raw_data={"photometry": phot, "history": hist}
        )


class PhotometrySpecialist:
    """Evaluates photometric classification, conformal sets, and class probabilities."""
    role = "photometry"

    def assess(self, obj_id: str, horizon_days: float = 100.0) -> SpecialistAssessment:
        cl = F.flare_classify.call(dict(obj_id=obj_id, horizon_days=horizon_days))
        if "error" in cl:
            return SpecialistAssessment(
                role=self.role,
                verdict_candidate="insufficient_data",
                confidence=0.1,
                key_findings=[f"Classification failed: {cl['error']}"],
                status="error",
                raw_data=cl,
                concerns=["Model could not run on object."]
            )

        model = cl.get("model", "unknown")
        pred = cl.get("predicted", "unknown")
        probs = cl.get("probabilities", {})
        pset = cl.get("prediction_set", [])
        cred = cl.get("credibility", 0.0) or 0.0
        conf = cl.get("confidence", 0.0) or 0.0

        findings = [
            f"Phase model '{model}' predicted argmax: {pred} (P={probs.get(pred, 0.0):.3f})",
            f"90% Conformal Prediction Set: {{{', '.join(pset)}}}",
            f"Credibility: {cred:.3f}, Confidence: {conf:.3f}"
        ]
        concerns = []
        flags = {
            "model": model,
            "predicted": pred,
            "prediction_set": pset,
            "credibility": cred,
            "confidence": conf,
            "probabilities": probs,
            "argmax_excluded": cl.get("argmax_excluded_from_set", False)
        }

        # Conformal credibility check: does the light curve fit any known class?
        if cred < 0.05:
            concerns.append(f"Low credibility ({cred:.3f} < 0.05): Object poorly matches all 6 BTS classes in training distribution.")
            cand_verdict = "anomaly_review"
        elif pred in ("SLSN", "TDE"):
            findings.append(f"High-interest rare transient predicted: {pred}")
            cand_verdict = "needs_spectrum"
        elif "SLSN" in pset or "TDE" in pset:
            findings.append(f"Rare class ({', '.join([c for c in pset if c in ('SLSN', 'TDE')])}) included in 90% conformal set.")
            cand_verdict = "needs_spectrum"
        elif pred in ("AGN", "CV"):
            cand_verdict = "likely_stellar_or_agn"
        elif len(pset) > 2:
            concerns.append(f"Ambiguous prediction set with {len(pset)} classes.")
            cand_verdict = "ambiguous"
        else:
            cand_verdict = "ordinary"

        return SpecialistAssessment(
            role=self.role,
            verdict_candidate=cand_verdict,
            confidence=conf,
            confidence_kind="FLARE conformal confidence; not discovery probability",
            key_findings=findings,
            flags=flags,
            concerns=concerns,
            raw_data=cl
        )


class AnomalySpecialist:
    """Evaluates blind consensus anomaly alarms, out-of-distribution scorers, and novelty."""
    role = "anomaly"

    def assess(self, obj_id: str, horizon_days: float = 100.0) -> SpecialistAssessment:
        anom = F.flare_anomaly.call(dict(obj_id=obj_id, horizon_days=horizon_days))
        if "error" in anom:
            return SpecialistAssessment(
                role=self.role,
                verdict_candidate="insufficient_data",
                status="error",
                confidence=0.0,
                key_findings=[f"Anomaly alarm failed: {anom['error']}"],
                raw_data=anom,
                concerns=["Anomaly scorer unavailable."]
            )

        # flare_anomaly returns the decisions as {"1%": bool, "5%": bool, "10%": bool};
        # keep only the rates that fired, and none when the alarm is uncalibrated at this phase.
        raw_alarms = anom.get("alarms") or []
        if isinstance(raw_alarms, dict):
            alarms = [rate for rate, fired in raw_alarms.items() if fired is True]
        else:
            alarms = [str(rate) for rate in raw_alarms]
        calibrated = anom.get("calibrated", True) is not False
        if not calibrated:
            alarms = []
        far = anom.get("alarm_far")
        pct = anom.get("energy_percentile")
        lr = anom.get("likelihood_ratio")
        score = anom.get("alarm_score", "consensus")

        findings = []
        concerns = [] if calibrated else ["Anomaly alarm is uncalibrated at this phase; its decisions are withheld."]
        flags = {
            "alarms": alarms,
            "calibrated": calibrated,
            "alarm_far": far,
            "energy_percentile": pct,
            "likelihood_ratio": lr,
            "consensus_score": score
        }

        if "1%" in alarms:
            findings.append("TRIGGERED 1% False-Alarm-Rate Blind Consensus Alarm (strong anomaly)")
            cand_verdict = "anomaly_review"
            confidence = 0.92
        elif "5%" in alarms:
            findings.append("Triggered 5% False-Alarm-Rate Alarm (moderate anomaly)")
            cand_verdict = "anomaly_review"
            confidence = 0.78
        elif "10%" in alarms:
            findings.append("Triggered 10% False-Alarm-Rate Alarm (mild outlier)")
            cand_verdict = "anomaly_review"
            confidence = 0.60
        else:
            findings.append("No anomaly alarm triggered across 1%, 5%, 10% FAR thresholds.")
            cand_verdict = "ordinary"
            confidence = 0.85

        if pct is not None:
            findings.append(f"Novelty energy percentile: {pct:.1f}%")
            if pct > 95.0:
                findings.append(f"Extreme novelty energy (>95th percentile).")

        return SpecialistAssessment(
            role=self.role,
            verdict_candidate=cand_verdict,
            confidence=confidence,
            key_findings=findings,
            flags=flags,
            concerns=concerns,
            raw_data=anom
        )


class ContextSpecialist:
    """Evaluates host galaxy context, Gaia astrometric point source checks, and WISE colors."""
    role = "context"

    def assess(self, obj_id: str) -> SpecialistAssessment:
        ctx = O.get_context.call(dict(obj_id=obj_id))
        if "error" in ctx:
            return SpecialistAssessment(
                role=self.role,
                verdict_candidate="insufficient_data",
                status="error",
                confidence=0.0,
                key_findings=[f"Context lookup failed: {ctx['error']}"],
                raw_data=ctx,
                concerns=["Catalogue context unavailable."]
            )

        findings = []
        concerns = []
        flags = ctx.get("flags", {})
        host = ctx.get("host", {})
        z_dict = ctx.get("redshift", {})
        gaia = ctx.get("gaia", {})
        wise = ctx.get("wise", {})

        # 1. Gaia Galactic star check
        if flags.get("stellar_astrometry"):
            findings.append(f"Significant Gaia parallax/proper-motion (pm_over_err={gaia.get('pm_over_error')}, plx_over_err={gaia.get('parallax_over_error')})")
            concerns.append("Gaia astrometry favors a Galactic counterpart; verify the association before excluding an extragalactic transient.")
            return SpecialistAssessment(
                role=self.role,
                verdict_candidate="likely_stellar_or_agn",
                confidence=0.98,
                key_findings=findings,
                flags=flags,
                concerns=concerns,
                raw_data=ctx
            )

        # 2. Host offset & location
        sep = host.get("offset_arcsec")
        if flags.get("hostless"):
            findings.append("No host association supplied; this does not establish a hostless transient.")
        elif flags.get("nuclear") and sep is not None:
            findings.append(f"Nuclear location: offset {sep:.2f} arcsec from host center.")
        elif sep is not None:
            findings.append(f"Host galaxy separated by {sep:.2f} arcsec (r_mag={host.get('r_mag')}).")

        # 3. Redshift and absolute magnitude
        z = z_dict.get("z_phot") if z_dict.get("z_phot") is not None else z_dict.get("z_spec_host")
        m_pseudo = z_dict.get("M_pseudo")
        if z is not None:
            findings.append(f"Redshift z={z:.3f}; pseudo-absolute peak M_pseudo={m_pseudo} mag.")
            if m_pseudo is not None and m_pseudo < -21.0:
                findings.append("Apparent superluminous regime (M < -21).")
                concerns.append("Pitfall warning: M_pseudo < -21 alone marks a true SLSN only 1% of the time due to photo-z catastrophic outliers.")

        # 4. WISE AGN Mid-IR Color
        if flags.get("agn_mid_ir_colour") and wise.get("w1_w2") is not None:
            w1w2 = wise.get("w1_w2")
            findings.append(f"Mid-IR WISE color W1-W2={w1w2:.2f} > 0.8 marks AGN/QSO.")
            concerns.append("Infrared colors strongly favor active galactic nucleus.")

        cand_verdict = "likely_stellar_or_agn" if flags.get("agn_mid_ir_colour") else "ordinary"
        return SpecialistAssessment(
            role=self.role,
            verdict_candidate=cand_verdict,
            confidence=0.80,
            key_findings=findings,
            flags=flags,
            concerns=concerns,
            raw_data=ctx
        )


class SpectroscopySpecialist:
    """ASTRANet evidence, with its own taxonomy and novelty calibration."""
    role = "spectroscopy"

    def assess(self, obj_id: str) -> SpecialistAssessment:
        from ..astranet.triage import mock_triage
        from ..astranet import tools as S
        report = mock_triage(obj_id, blind=True)
        if report.verdict == "insufficient_data":
            return SpecialistAssessment(role=self.role, status="unavailable", confidence=0,
                verdict_candidate="insufficient_data", key_findings=[report.summary],
                concerns=report.caveats, raw_data={"report": report.model_dump()})
        classification = S.astranet_classify.call(dict(obj_id=obj_id))
        novelty = S.astranet_novelty.call(dict(obj_id=obj_id))
        return SpecialistAssessment(role=self.role, verdict_candidate=report.verdict,
            confidence=classification.get("probability") or 0,
            confidence_kind="ASTRANet softmax probability; not calibrated discovery probability",
            key_findings=[e.fact for e in report.evidence], concerns=report.caveats,
            flags=dict(predicted=report.flare_label, priority=report.priority,
                       novelty_available=novelty.get("status") == "ok"),
            raw_data=dict(report=report.model_dump(), classification=classification, novelty=novelty))
