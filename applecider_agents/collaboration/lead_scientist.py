"""Lead Scientist agent that conducts the multi-agent debate and synthesizes a final DiscoveryReport.

Follows the core AppleCiDEr principle:
Every number and measurement comes strictly from deterministic tools;
the reasoning engine evaluates credibility, balances specialist views, resolves pitfalls,
and formulates the final scientific verdict and follow-up strategy.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from .schemas import SpecialistAssessment, DebateMessage, DiscoveryReport
from .specialists import TriageSpecialist, PhotometrySpecialist, AnomalySpecialist, ContextSpecialist
from ..core.report import Evidence, Verdict


class LeadScientist:
    """The Principal Investigator agent synthesizing multi-specialist transient evaluations."""

    def __init__(self, model_name: str = "flare"):
        self.model_name = model_name
        self.triage_spec = TriageSpecialist()
        self.phot_spec = PhotometrySpecialist()
        self.anom_spec = AnomalySpecialist()
        self.ctx_spec = ContextSpecialist()

    def run_investigation(self, obj_id: str, horizon_days: float = 100.0, use_llm: bool = False) -> DiscoveryReport:
        """Run the multi-agent investigation: gathering assessments, debating, and synthesizing."""
        # 1. Independent specialist assessments
        triage_assess = self.triage_spec.assess(obj_id, horizon_days=horizon_days)
        phot_assess = self.phot_spec.assess(obj_id, horizon_days=horizon_days)
        anom_assess = self.anom_spec.assess(obj_id, horizon_days=horizon_days)
        ctx_assess = self.ctx_spec.assess(obj_id)

        assessments = {
            "triage": triage_assess,
            "photometry": phot_assess,
            "anomaly": anom_assess,
            "context": ctx_assess
        }

        # 2. Conduct debate / cross-examination
        transcript, conflicts_resolved, verdict, priority, discovery_score, hypothesis, action = self._debate_and_reconcile(
            obj_id, assessments
        )

        # 3. Gather tool evidence
        evidence = self._collect_evidence(assessments)

        # 4. Flare labels & conformal prediction set
        pred_set = phot_assess.flags.get("prediction_set", [])
        pred_label = phot_assess.flags.get("predicted", "none")
        cred = phot_assess.flags.get("credibility", 0.0) or 0.0
        energy_pct = anom_assess.flags.get("energy_percentile", 0.0) or 0.0

        # 5. Build summary
        summary = (
            f"Multi-agent investigation determined verdict '{verdict}' (priority {priority}). "
            f"Hypothesis: {hypothesis}. "
            f"Discovery score: {discovery_score:.2f}. "
            f"Reconciled {len(conflicts_resolved)} cross-specialist factors."
        )

        report = DiscoveryReport(
            obj_id=obj_id,
            verdict=verdict,
            priority=priority,
            flare_label=pred_label,
            prediction_set=pred_set,
            credibility=cred,
            energy_percentile=energy_pct,
            summary=summary,
            evidence=evidence,
            suggested_action=action,
            caveats=[c for a in assessments.values() for c in a.concerns],
            scientific_hypothesis=hypothesis,
            discovery_potential=discovery_score,
            specialist_assessments=assessments,
            debate_transcript=transcript,
            conflicts_resolved=conflicts_resolved
        )
        return report

    def _debate_and_reconcile(
        self, obj_id: str, assessments: Dict[str, SpecialistAssessment]
    ) -> Tuple[List[DebateMessage], List[str], Verdict, int, float, str, str]:
        """Debate loop resolving discrepancies between specialist assessments."""
        transcript: List[DebateMessage] = []
        conflicts: List[str] = []

        triage = assessments["triage"]
        phot = assessments["photometry"]
        anom = assessments["anomaly"]
        ctx = assessments["context"]

        # Initial remarks from specialists
        transcript.append(DebateMessage(
            sender="TriageSpecialist",
            content=f"Initial screening complete: candidate verdict '{triage.verdict_candidate}' with confidence {triage.confidence:.2f}."
        ))
        transcript.append(DebateMessage(
            sender="PhotometrySpecialist",
            content=f"Model predicted '{phot.flags.get('predicted')}' (P={phot.flags.get('probabilities', {}).get(phot.flags.get('predicted', ''), 0.0):.3f}) with set {{{', '.join(phot.flags.get('prediction_set', []))}}}. Credibility: {phot.flags.get('credibility', 0.0):.3f}."
        ))
        transcript.append(DebateMessage(
            sender="AnomalySpecialist",
            content=f"Novelty analysis: alarms={anom.flags.get('alarms')}, smallest FAR={anom.flags.get('alarm_far')}, energy_pct={anom.flags.get('energy_percentile')}."
        ))
        transcript.append(DebateMessage(
            sender="ContextSpecialist",
            content=f"Context analysis: hostless={ctx.flags.get('hostless')}, stellar={ctx.flags.get('stellar_astrometry')}, agn_color={ctx.flags.get('agn_mid_ir_colour')}."
        ))

        # --- RECONCILIATION RULES & DEBATE ---

        # Case 1: Already classified
        if triage.verdict_candidate == "already_classified":
            verdict: Verdict = "already_classified"
            priority = 0
            discovery_score = 0.0
            spec_type = triage.flags.get("bts_type", "known")
            hypothesis = f"Known transient already spectroscopically typed in BTS as {spec_type}."
            action = "Archive record; no new observational action required."
            transcript.append(DebateMessage(
                sender="LeadScientist",
                content=f"Source is confirmed as {spec_type} in BTS. Concluding already_classified."
            ))
            return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action

        # Case 2: Insufficient data
        if triage.verdict_candidate == "insufficient_data":
            verdict = "insufficient_data"
            priority = 0
            discovery_score = 0.05
            hypothesis = "Under-sampled event with insufficient detections for reliable classification."
            action = "Wait for subsequent survey visits to accumulate photometric baseline."
            transcript.append(DebateMessage(
                sender="LeadScientist",
                content="Insufficient detections inside the horizon. Pausing analysis."
            ))
            return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action

        # Case 3: Galactic Star Confirmed by Gaia Astrometry
        if ctx.flags.get("stellar_astrometry"):
            conflict_msg = "Overruled extragalactic transient classification: Gaia parallax/proper-motion confirms Galactic star."
            conflicts.append(conflict_msg)
            transcript.append(DebateMessage(
                sender="ContextSpecialist",
                recipient="PhotometrySpecialist",
                content="Challenging photometric classification: Gaia astrometry shows significant parallax/proper motion (>3 sigma). This is a Galactic variable, not a supernova."
            ))
            transcript.append(DebateMessage(
                sender="LeadScientist",
                content="Resolved: Astrometry is definitive. Reclassifying as likely_stellar_or_agn."
            ))
            verdict = "likely_stellar_or_agn"
            priority = 0
            discovery_score = 0.05
            hypothesis = "Galactic stellar flare or cataclysmic variable with significant proper motion or parallax."
            action = "Ignore or flag as stellar variable."
            return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action

        # Case 4: Long-term Historical Baseline Activity
        if triage.flags.get("archive_span_days", 0.0) > 365.0:
            conflict_msg = f"Reconciled multi-year activity ({triage.flags.get('archive_span_days', 0):.0f} days) against transient hypothesis: identified as persistent AGN or nuclear variable."
            conflicts.append(conflict_msg)
            transcript.append(DebateMessage(
                sender="TriageSpecialist",
                recipient="LeadScientist",
                content=f"Caution: Archival broker history reveals detections spanning {triage.flags.get('archive_span_days', 0):.0f} days. Explosive transients do not persist across years."
            ))
            verdict = "likely_stellar_or_agn"
            priority = 1 if anom.flags.get("alarms") else 0
            discovery_score = 0.2 if anom.flags.get("alarms") else 0.05
            hypothesis = "Persistent active galactic nucleus (AGN) exhibiting ongoing stochastic variability."
            action = "Cross-check with known AGN/QSO catalogues; monitor for unusual flaring."
            return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action

        # Case 5: Strong Anomaly Discovery (1% False Alarm Rate Alarm)
        is_1pct_alarm = "1%" in (anom.flags.get("alarms") or [])
        cred = phot.flags.get("credibility", 1.0) or 1.0
        pred = phot.flags.get("predicted", "")

        if is_1pct_alarm:
            transcript.append(DebateMessage(
                sender="AnomalySpecialist",
                recipient="all",
                content="ALERT: 1% False-Alarm-Rate blind consensus alarm triggered across 14 LUNA scorers. Light curve is highly atypical."
            ))
            discovery_score = 0.95
            priority = 3
            verdict = "anomaly_review"

            if cred < 0.05:
                hypothesis = "High-confidence novel transient out-of-distribution with respect to known BTS taxonomy (e.g. FBOT, Kilonova, or exotic relativistic explosion)."
                action = "URGENT: Request immediate Target of Opportunity (ToO) spectroscopy and multi-color follow-up tonight."
            elif pred in ("SLSN", "TDE"):
                hypothesis = f"Candidate rare energetic transient ({pred}) exhibiting anomalous photometric evolution."
                action = "Schedule spectroscopic follow-up at high priority to verify rare class nature."
            else:
                hypothesis = f"Photometrically anomalous event (nominally predicted as {pred}, but with 1% anomaly alarm)."
                action = "Flag for expert visual inspection and prompt spectroscopic validation."
            return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action

        # Case 6: SLSN Pitfall Check
        if pred == "SLSN" or "SLSN" in phot.flags.get("prediction_set", []):
            m_pseudo = ctx.raw_data.get("redshift", {}).get("M_pseudo")
            if m_pseudo is not None and m_pseudo > -20.0:
                conflict_msg = f"Down-weighted SLSN classification: M_pseudo={m_pseudo:.1f} is fainter than true SLSN regime (M < -21); likely photo-z inaccuracy or SN II/Ia."
                conflicts.append(conflict_msg)
                transcript.append(DebateMessage(
                    sender="ContextSpecialist",
                    recipient="PhotometrySpecialist",
                    content=f"Skeptical critique: The model flags SLSN, but the host photo-z gives M_pseudo={m_pseudo:.1f}, which is insufficiently luminous for a true superluminous supernova."
                ))
                transcript.append(DebateMessage(
                    sender="LeadScientist",
                    content="Skeptical critique accepted: True SLSN precision on BTS is 21-64%. Re-classifying as needs_spectrum with medium priority."
                ))
                verdict = "needs_spectrum"
                priority = 2
                discovery_score = 0.70
                hypothesis = "Candidate luminous transient with uncertain host redshift; potential core-collapse SN or sub-luminous SLSN."
                action = "Obtain spectrum if telescope queue allows to measure spectroscopic redshift."
                return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action
            else:
                verdict = "needs_spectrum"
                priority = 3
                discovery_score = 0.88
                hypothesis = f"Strong Superluminous Supernova candidate (M_pseudo={m_pseudo}) in rise or near peak."
                action = "Request rapid spectroscopy tonight to capture pre-peak UV/optical spectral features."
                return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action

        # Case 7: TDE Candidate
        if pred == "TDE" or "TDE" in phot.flags.get("prediction_set", []):
            is_nuclear = ctx.flags.get("nuclear", False)
            if is_nuclear:
                verdict = "needs_spectrum"
                priority = 3
                discovery_score = 0.90
                hypothesis = "Nuclear Tidal Disruption Event (TDE) candidate in galactic center."
                action = "Trigger prompt UV/Optical spectroscopy and Swift X-ray/UVOT monitoring."
            else:
                conflict_msg = "TDE predicted by photometry, but transient is non-nuclear (offset > 0.5 arcsec). Resolving in favor of core-collapse supernova."
                conflicts.append(conflict_msg)
                verdict = "needs_spectrum"
                priority = 2
                discovery_score = 0.65
                hypothesis = "Non-nuclear transient resembling TDE light curve; likely peculiar core-collapse supernova."
                action = "Spectroscopy recommended if time permits."
            return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action

        # Case 8: Moderate Anomaly (5% or 10% Alarm)
        if anom.flags.get("alarms"):
            verdict = "anomaly_review"
            priority = 2
            discovery_score = 0.60
            hypothesis = f"Transient showing moderate deviation from standard light curve models ({', '.join(anom.flags.get('alarms', []))} alarm)."
            action = "Inspect light curve residuals and consider spectral confirmation."
            return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action

        # Case 9: Ambiguous Conformal Set
        if phot.verdict_candidate == "ambiguous":
            verdict = "ambiguous"
            priority = 1
            discovery_score = 0.35
            hypothesis = f"Standard transient with ambiguous classification {{{', '.join(phot.flags.get('prediction_set', []))}}}."
            action = "Monitor light curve evolution for further color and decay differentiation."
            return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action

        # Case 10: Standard Ordinary Transient (SN Ia, SN CC)
        verdict = "ordinary"
        priority = 1 if pred in ("SN_Ia", "SN_CC") else 0
        discovery_score = 0.20
        hypothesis = f"Standard extragalactic transient matching {pred} light-curve morphology."
        action = "Standard survey follow-up; no special intervention needed."
        return transcript, conflicts, verdict, priority, discovery_score, hypothesis, action

    def _collect_evidence(self, assessments: Dict[str, SpecialistAssessment]) -> List[Evidence]:
        """Aggregate evidence items from all specialists."""
        ev = []
        for role, assess in assessments.items():
            for finding in assess.key_findings:
                ev.append(Evidence(tool=f"{role}_specialist", fact=finding))
        return ev
