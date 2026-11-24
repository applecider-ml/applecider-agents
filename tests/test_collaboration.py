"""Tests for the Multi-Agent collaborative transient investigation framework."""
import pytest
from applecider_agents.collaboration import (
    TriageSpecialist,
    PhotometrySpecialist,
    AnomalySpecialist,
    ContextSpecialist,
    LeadScientist,
    collaborate,
    DiscoveryReport
)

KNOWN = "ZTF18acbwazl"     # Ca-rich transient in the benchmark's out-of-taxonomy hold-out
TEST = "ZTF19acbzgog"      # the paper's worked-example nova


def test_specialists_individual_assessments():
    triage_spec = TriageSpecialist()
    t_known = triage_spec.assess(KNOWN, horizon_days=100.0)
    assert t_known.role == "triage"
    assert t_known.verdict_candidate == "already_classified"
    assert t_known.confidence > 0.9

    phot_spec = PhotometrySpecialist()
    p_test = phot_spec.assess(TEST, horizon_days=100.0)
    assert p_test.role == "photometry"
    assert "prediction_set" in p_test.flags
    assert len(p_test.key_findings) > 0

    anom_spec = AnomalySpecialist()
    a_test = anom_spec.assess(TEST, horizon_days=100.0)
    assert a_test.role == "anomaly"
    assert a_test.confidence >= 0.0

    ctx_spec = ContextSpecialist()
    c_known = ctx_spec.assess(KNOWN)
    assert c_known.role == "context"
    assert "hostless" in c_known.flags


def test_lead_scientist_debate_reconciliation():
    scientist = LeadScientist(model_name="flare")
    report = scientist.run_investigation(KNOWN, horizon_days=100.0)

    assert isinstance(report, DiscoveryReport)
    assert report.obj_id == KNOWN
    assert report.verdict == "already_classified"
    assert report.priority == 0
    assert len(report.debate_transcript) >= 4
    assert len(report.specialist_assessments) == 4


def test_collaborate_workflow_and_summary():
    report = collaborate(TEST, model="flare", horizon_days=100.0, use_llm=False)
    assert isinstance(report, DiscoveryReport)
    assert report.obj_id == TEST
    assert report.discovery_potential >= 0.0
    assert len(report.scientific_hypothesis) > 0

    # Ensure compatibility with SkyPortal comment
    cmt = report.comment()
    assert cmt.startswith("[FLARE]")

    # Check detailed multi-agent summary rendering
    det = report.detailed_summary()
    assert "MULTI-AGENT DISCOVERY REPORT" in det
    assert "[TRIAGE]" in det
    assert "[PHOTOMETRY]" in det
    assert "[ANOMALY]" in det
    assert "[CONTEXT]" in det


if __name__ == "__main__":
    test_specialists_individual_assessments()
    test_lead_scientist_debate_reconciliation()
    test_collaborate_workflow_and_summary()
    print("All multi-agent tests passed successfully!")
