"""Runs without an API key. Needs network: the benchmark tables are fetched from the flare
repository on first use and light curves come from the broker archive through flare.fetch."""
import json, asyncio
from applecider_agents.core import object_tools as O, registry
from applecider_agents.flare import tools as T, AGENT
from applecider_agents.core.report import Report

KNOWN = "ZTF18acbwazl"     # Ca-rich transient in the benchmark's out-of-taxonomy hold-out
TEST = "ZTF19acbzgog"      # the paper's worked-example nova


def test_object_tools():
    r = O.get_photometry.call(dict(obj_id=KNOWN, horizon_days=100.0)); assert "error" not in r and r["n_detections"] >= 2
    x = O.get_context.call(dict(obj_id=KNOWN)); assert "flags" in x and x["source"] in ("benchmark", "live")
    k = O.catalog_lookup.call(dict(obj_id=KNOWN)); assert k["already_classified"] and k["spectroscopic_type"] == "Ca-rich"
    h = O.light_curve_history.call(dict(obj_id=KNOWN)); assert "found" in h


def test_flare_tools_and_verdict():
    c = T.flare_classify.call(dict(obj_id=TEST, horizon_days=100.0)); assert abs(sum(c["probabilities"].values()) - 1) < 0.02
    a = T.flare_anomaly.call(dict(obj_id=TEST, horizon_days=100.0)); assert a["energy_percentile"] is not None
    rep = AGENT.triage(TEST, 100.0); assert isinstance(rep, Report) and rep.comment().startswith("[FLARE]")
    assert registry.get("flare").name == "flare" and AGENT.triage(KNOWN, 100.0).verdict == "already_classified"


def test_mcp_server_builds_and_serves():
    from applecider_agents.core.mcp import build
    server = build(["flare"])
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"flare_triage", "flare_classify", "flare_anomaly", "get_context", "get_photometry", "light_curve_history", "flare_analyze_photometry"} <= names
    r = asyncio.run(server.call_tool("flare_classify", {"obj_id": TEST}))
    d = getattr(r, "structured_content", None) or json.loads(r.content[0].text); d = d.get("result", d)
    assert d["predicted"] in d["probabilities"]


if __name__ == "__main__":
    for f in (test_object_tools, test_flare_tools_and_verdict, test_mcp_server_builds_and_serves):
        f(); print("ok", f.__name__)
