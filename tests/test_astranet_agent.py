"""Runs without an API key and without network. Needs the astranet package, its twelve
checkpoints (ASTRANET_MODELS) and, for the novelty assertions, the novelty bundle
(ASTRANET_NOVELTY or <models>/novelty.joblib). The spectrum is synthetic, so the test checks
the plumbing (tools, verdict, MCP), not the science."""
import asyncio, json, os, tempfile
from pathlib import Path
import numpy as np

OBJ = "TESTobj"
TMP = Path(tempfile.mkdtemp())
w = np.linspace(3700, 9200, 2200); rng = np.random.default_rng(0)
for i in range(2):
    f = 1e-16 * ((w / 6000.0) ** -1.5) * (1 - 0.4 * np.exp(-0.5 * ((w - 6150) / 60) ** 2)) + 2e-18 * rng.standard_normal(w.size)
    np.savetxt(TMP / f"{OBJ}_{i}.dat", np.column_stack([w, f]))
os.environ["APPLECIDER_SPECTRA_DIRS"] = str(TMP)

from applecider_agents.core import registry                 # noqa: E402
from applecider_agents.core.report import Report            # noqa: E402
from applecider_agents.astranet import tools as T, AGENT    # noqa: E402


def test_astranet_tools():
    s = T.get_spectra.call(dict(obj_id=OBJ)); assert s["n_spectra"] == 2 and s["source"] == "local"
    c = T.astranet_classify.call(dict(obj_id=OBJ, redshift=0.03)); assert "error" not in c, c
    assert abs(sum(c["probabilities"].values()) - 1) < 0.02 and c["n_spectra"] == 2 and c["redshift_used"] == 0.03
    z = T.astranet_classify.call(dict(obj_id=OBJ, redshift=0.03, use_redshift=False)); assert z["redshift_used"] is None
    n = T.astranet_novelty.call(dict(obj_id=OBJ, redshift=0.03)); assert "error" not in n, n
    if n["status"] == "ok":
        assert 0 < n["p_value"] <= 1 and n["stratum"] == "redshift" and set(n["alarm"]) == {"1%", "5%", "10%", "15%"}
    assert "error" in T.get_spectra.call(dict(obj_id="NOSUCHobj")) or T.get_spectra.call(dict(obj_id="NOSUCHobj"))["n_spectra"] == 0


def test_astranet_verdict():
    rep = AGENT.triage(OBJ, 100.0, blind=True)
    assert isinstance(rep, Report) and rep.comment().startswith("[ASTRANet]") and rep.verdict != "already_classified"
    assert rep.flare_label in ("Ia", "Ibc", "SLSN", "II", "AGN", "CV", "TDE") and any(e.tool == "astranet_classify" for e in rep.evidence)
    assert registry.get("astranet").name == "astranet" and "no redshift" in " ".join(rep.caveats)
    empty = AGENT.triage("NOSUCHobj", 100.0, blind=True); assert empty.verdict == "insufficient_data"


def test_mcp_server_serves_astranet():
    from applecider_agents.core.mcp import build
    server = build(["astranet"])
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"astranet_triage", "get_spectra", "astranet_classify", "astranet_novelty", "astranet_analyze_spectrum"} <= names
    r = asyncio.run(server.call_tool("astranet_classify", {"obj_id": OBJ}))
    d = getattr(r, "structured_content", None) or json.loads(r.content[0].text); d = d.get("result", d)
    assert d["predicted"] in d["probabilities"]


def test_publish_uses_the_model_origin():
    from applecider_agents.core import skyportal_client as S
    sent = []

    class Fake:
        def comment(self, obj_id, text, group_ids=None): sent.append(("comment", text)); return dict(status="success")
        def annotation(self, obj_id, data, origin="FLARE", group_ids=None): sent.append((origin, data)); return dict(status="success")
    real, S.SkyPortal = S.SkyPortal, Fake
    try:
        rep = AGENT.triage(OBJ, 100.0, blind=True)
        assert S.publish(rep, mode="dry-run")["posted"] is False and not sent
        assert S.publish(rep, mode="comment")["posted"] and sent[0][1].startswith("[ASTRANet]")
        assert sent[1][0] == "ASTRANet-agent" and {"astranet_class", "novelty_p"} <= set(sent[1][1])
        base = Report(obj_id="x", verdict="ordinary", priority=0, flare_label="SN_Ia", prediction_set=["SN_Ia"], credibility=0.5, energy_percentile=10.0,
                           summary="s", evidence=[], suggested_action="none", caveats=[])
        S.publish(base, mode="comment"); assert sent[3][0] == "FLARE-agent" and "energy_percentile" in sent[3][1]
    finally:
        S.SkyPortal = real
    assert registry.get("astranet").report_cls.__name__ == "SpectralReport"


if __name__ == "__main__":
    for f in (test_astranet_tools, test_astranet_verdict, test_mcp_server_serves_astranet, test_publish_uses_the_model_origin):
        f(); print("ok", f.__name__)
