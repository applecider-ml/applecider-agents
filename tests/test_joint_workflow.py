"""Offline regression tests for orchestration, not model accuracy claims."""
from types import SimpleNamespace
import json
import pytest
from applecider_agents.collaboration import LeadScientist, collaborate
from applecider_agents.collaboration import specialists as S
from applecider_agents.astranet import tools as A
from applecider_agents.core import config as C


def tool(value):
    return SimpleNamespace(call=lambda args: value)


@pytest.fixture
def outputs(monkeypatch, tmp_path):
    values = {
        "get_photometry": dict(n_detections=12, span_days=40., peak_mag_observed=18.),
        "light_curve_history": dict(found=True, activity_span_days=40., n_detections=12),
        "catalog_lookup": dict(already_classified=False),
        "tns_lookup": dict(available=True, matches=[]),
        "get_context": dict(flags={}, host={}, redshift={}, gaia={}, wise={}),
        "flare_classify": dict(predicted="SN_Ia", probabilities={"SN_Ia": .9}, prediction_set=["SN_Ia"], credibility=.7, confidence=.8),
        "flare_anomaly": dict(alarms=[], energy_percentile=30., alarm_score="consensus"),
        "get_spectra": dict(n_spectra=1, source="provided", redshift=.03),
        "astranet_classify": dict(predicted="Ia", probability=.9, n_spectra=1, spectra_agreeing=1, n_rejected=0,
                                  redshift_used=.03, classes_predicted_by_a_spectrum=["Ia"]),
        "astranet_novelty": dict(status="ok", p_value=.4, stratum="redshift", n_calibration=50,
                                strictest_alarm_rate=None, alarm={}, percentile_among_ordinary=60.),
    }
    for name, value in values.items():
        module = A if name.startswith("astranet_") or name == "get_spectra" else S.F if name.startswith("flare_") else S.O
        monkeypatch.setattr(module, name, tool(value))
    monkeypatch.setattr(C, "LOG_DIR", tmp_path)
    return values


def test_joint_runs_both_preserves_separate_numbers(outputs):
    report = collaborate("X", model="joint")
    assert report.data_status == {"flare": "ok", "astranet": "ok"}
    assert report.model_results["flare"]["classification"]["credibility"] == .7
    assert report.model_results["astranet"]["novelty"]["p_value"] == .4
    assert report.verdict == "ordinary"
    assert len(report.specialist_assessments) == 5
    assert report.ORIGIN == "AppleCiDEr-collaboration"
    assert report.annotation()["astranet_class"] == "Ia"
    assert "not a probability" in report.score_kind


@pytest.mark.parametrize("spectral_label,disagree", [("II", False), ("Ibc", False), ("Ia", True)])
def test_taxonomy_mapping(outputs, spectral_label, disagree):
    outputs["flare_classify"]["predicted"] = "SN_CC"
    outputs["astranet_classify"]["predicted"] = spectral_label
    report = collaborate("X", model="joint")
    assert (report.verdict == "ambiguous") == disagree
    if disagree:
        assert report.priority >= 2
        assert report.provenance["issues_for_review"]
        assert not report.conflicts_resolved


def test_astranet_mode_does_not_run_flare(outputs, monkeypatch):
    def forbidden(**args):
        raise AssertionError("FLARE must not run")
    monkeypatch.setattr(S.F, "flare_classify", SimpleNamespace(call=forbidden))
    monkeypatch.setattr(S.O, "get_photometry", SimpleNamespace(call=forbidden))
    monkeypatch.setattr(S.O, "get_context", SimpleNamespace(call=forbidden))
    report = collaborate("X", model="astranet")
    assert set(report.model_results) == {"astranet"}
    assert "photometry" not in report.specialist_assessments
    assert report.verdict == "ordinary"


def test_missing_spectra_keeps_flare_and_marks_partial(outputs):
    outputs["get_spectra"].update(n_spectra=0)
    report = collaborate("X", model="joint")
    assert report.data_status["astranet"] == "unavailable"
    assert report.model_results["flare"]["predicted"] == "SN_Ia"
    assert any("Incomplete joint" in c for c in report.caveats)


def test_spectra_can_rescue_missing_photometry(outputs):
    outputs["get_photometry"].update(n_detections=0, error="no detections")
    outputs["flare_classify"].update(error="no detections")
    outputs["flare_anomaly"].update(error="no detections")
    report = collaborate("X", model="joint")
    assert report.data_status == {"flare": "unavailable", "astranet": "ok"}
    assert report.verdict == "ordinary"
    assert report.model_results["astranet"]["predicted"] == "Ia"


def test_missing_models_never_ordinary(outputs):
    outputs["flare_classify"].update(error="weights unavailable")
    outputs["get_spectra"].update(error="weights unavailable")
    report = collaborate("X", model="joint")
    assert report.verdict == "insufficient_data"


def test_failed_anomaly_not_ordinary(outputs):
    outputs["flare_anomaly"]["error"] = "missing model"
    report = collaborate("X", model="flare")
    assert report.verdict == "ambiguous"
    assert report.data_status["flare"] == "partial"


def test_tns_classification_is_used(outputs):
    outputs["tns_lookup"]["matches"] = [dict(object_type="SN Ia", name="SN X")]
    report = collaborate("X", model="joint")
    assert report.verdict == "already_classified"
    assert report.discovery_potential == 0
    assert report.model_results["astranet"]["predicted"] == "Ia"


def test_zero_credibility_retained(outputs):
    outputs["flare_classify"]["credibility"] = 0.
    outputs["flare_anomaly"]["alarms"] = ["1%"]
    report = collaborate("X", model="flare")
    assert report.credibility == 0
    assert "Low-credibility" in report.scientific_hypothesis
    assert "Kilonova" not in report.scientific_hypothesis


def test_sparse_alarm_not_urgent(outputs):
    outputs["get_photometry"]["n_detections"] = 3
    outputs["flare_anomaly"]["alarms"] = ["1%"]
    report = collaborate("X", model="flare")
    assert report.priority < 3


def test_context_nulls_do_not_crash(outputs):
    outputs["get_context"].update(host={"offset_arcsec": None}, redshift={"z_spec_host": .1, "M_pseudo": None})
    report = collaborate("X", model="joint")
    assert report.specialist_assessments["context"].status == "ok"


def test_new_llm_flag_calls_api_without_overwriting_measurements(outputs, monkeypatch):
    import anthropic
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Review spectroscopy evidence.")])
    monkeypatch.setattr(anthropic, "Anthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create)))
    report = collaborate("X", model="joint", use_llm=True, provenance={"snapshot_sha256": "abc"})
    assert len(calls) == 1
    payload = json.loads(calls[0]["messages"][0]["content"])
    assert payload["provenance"]["snapshot_sha256"] == "abc"
    assert report.llm_review == "Review spectroscopy evidence."
    assert report.model_results["astranet"]["novelty"]["p_value"] == .4
    logged = json.loads((C.LOG_DIR / "collaboration_reports.jsonl").read_text())
    assert logged["use_llm"] and logged["report"]["llm_review"]


def test_spectral_novelty_alarm_contributes(outputs):
    outputs["astranet_novelty"]["alarm"] = {"1%": True}
    outputs["astranet_novelty"]["strictest_alarm_rate"] = "1%"
    report = collaborate("X", model="joint")
    assert report.verdict == "anomaly_review"
    assert report.priority >= 2


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        LeadScientist("typo")


def test_llm_failure_preserves_rules_and_is_explicit(outputs, monkeypatch):
    def fail(report):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(LeadScientist, "_llm_review", staticmethod(fail))
    report = collaborate("X", model="joint", use_llm=True)
    assert report.llm_status == "error"
    assert "LLM unavailable" in report.llm_error
    assert report.verdict == "ordinary"
    assert report.data_status["astranet"] == "ok"


def test_low_credibility_without_alarm_is_reviewed(outputs):
    outputs["flare_classify"]["credibility"] = 0.01
    report = collaborate("X", model="flare")
    assert report.verdict == "anomaly_review"


def test_astranet_missing_novelty_is_explicit(outputs):
    outputs["astranet_novelty"].update(status="unavailable", reason="missing bundle")
    report = collaborate("X", model="astranet")
    assert report.data_status["astranet"] == "partial"
    assert report.verdict != "ordinary"


def test_blind_evaluation_hides_catalogue_and_tns(outputs, monkeypatch):
    def forbidden(args):
        pytest.fail("known labels leaked into blind investigation")
    monkeypatch.setattr(S.O, "catalog_lookup", SimpleNamespace(call=forbidden))
    monkeypatch.setattr(S.O, "tns_lookup", SimpleNamespace(call=forbidden))
    report = collaborate("X", model="joint", blind=True)
    assert report.verdict == "ordinary"
    assert report.provenance["blind"]


def test_mcp_exposes_joint_tools(outputs, monkeypatch):
    import asyncio
    from applecider_agents.core.mcp import build
    server = build(["flare", "astranet"])
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {"joint_triage", "skyportal_joint_triage", "flare_triage", "astranet_triage"} <= names
    result = asyncio.run(server.call_tool("joint_triage", {"obj_id": "X"}))
    raw = getattr(result, "structured_content", None) or json.loads(result.content[0].text)
    raw = raw.get("result", raw)
    assert raw["data_status"] == {"flare": "ok", "astranet": "ok"}


def test_eval_routes_joint_in_blind_mode(outputs, monkeypatch, tmp_path):
    import pandas as pd
    from applecider_agents.core import eval as E
    monkeypatch.setattr(E, "object_list", lambda *args: pd.DataFrame([dict(obj_id="X", truth="Ia", kind="known")]))
    def forbidden(args):
        pytest.fail("evaluation leaked label")
    monkeypatch.setattr(S.O, "catalog_lookup", SimpleNamespace(call=forbidden))
    monkeypatch.setattr(S.O, "tns_lookup", SimpleNamespace(call=forbidden))
    result = E.run("joint", "test", out=tmp_path / "eval.csv")
    assert result.iloc[0].verdict == "ordinary"
    assert "astranet" in result.iloc[0].data_status


def test_skyportal_cli_dispatches_joint_without_publishing(outputs, monkeypatch, capsys):
    from applecider_agents import cli
    from applecider_agents.core import skyportal_client
    from applecider_agents import skyportal_connector
    calls = []
    monkeypatch.setattr(skyportal_client, "SkyPortal", lambda: object())
    def investigate(oid, **kwargs):
        calls.append(kwargs)
        return collaborate(oid, model=kwargs["model"])
    monkeypatch.setattr(skyportal_connector, "investigate", investigate)
    monkeypatch.setattr(skyportal_client, "publish", lambda *args, **kwargs: pytest.fail("unexpected publication"))
    cli.main(["skyportal", "triage", "X"])
    assert calls[0]["model"] == "joint"
    assert json.loads(capsys.readouterr().out)["data_status"]["astranet"] == "ok"


def test_spectral_class_challenges_stellar_crossmatch(outputs):
    outputs["get_context"]["flags"]["stellar_astrometry"] = True
    report = collaborate("X", model="joint")
    assert report.verdict == "ambiguous"
    assert report.priority >= 2
    assert any("Gaia" in issue for issue in report.provenance["issues_for_review"])
