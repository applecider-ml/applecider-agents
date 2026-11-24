"""The OpenAI-compatible loop with a scripted endpoint: no network, no model."""
import json
from types import SimpleNamespace
import pytest
from applecider_agents.core import provider as P, config as C
from applecider_agents.core.report import Report

REPORT = dict(obj_id="obj_0001", verdict="anomaly_review", priority=2, flare_label="SN_Ia", prediction_set=["SN_Ia"],
              credibility=0.4, energy_percentile=97.0, summary="alarm at 1%", evidence=[dict(tool="probe", fact="p = 0.004")],
              suggested_action="inspect", caveats=[])


def _reply(name, args, content=""):
    return dict(model="local-model", usage=dict(prompt_tokens=10, completion_tokens=5),
                choices=[dict(message=dict(content=content, tool_calls=[dict(id="c1", function=dict(name=name, arguments=json.dumps(args)))]))])


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(C, "BASE_URL", "http://localhost:1/v1"); monkeypatch.setattr(C, "MODEL", "local-model")
    seen = []

    def install(replies):
        it = iter(replies)
        monkeypatch.setattr(P, "_post", lambda path, payload: (seen.append(json.loads(json.dumps(payload))), next(it))[1])
        return seen
    return install


def test_tools_then_report(scripted):
    tool = SimpleNamespace(name="probe", description="d", input_schema=dict(type="object", properties={}), call=lambda a: dict(p=0.004))
    seen = scripted([_reply("probe", dict(obj_id="obj_0001")), _reply(P.SUBMIT, REPORT)])
    rep, rec = P.run("sys", "user", [tool], Report)
    assert isinstance(rep, Report) and rep.priority == 2
    assert rec["tool_calls"] == [dict(tool="probe", input=dict(obj_id="obj_0001"))]
    assert rec["usage"] == dict(input=20, output=10) and rec["served_by"] == ["local-model"]
    assert json.loads(seen[1]["messages"][-1]["content"]) == dict(p=0.004)
    assert {t["function"]["name"] for t in seen[0]["tools"]} == {"probe", P.SUBMIT}


def test_invalid_report_is_returned_once(scripted):
    bad = dict(REPORT, priority=9)
    scripted([_reply(P.SUBMIT, bad), _reply(P.SUBMIT, REPORT)])
    rep, rec = P.run("sys", "user", [], Report)
    assert rep.priority == 2 and rec["report_retries"] == 1


def test_plain_text_is_pushed_to_submit(scripted):
    text = dict(model="local-model", usage={}, choices=[dict(message=dict(content="It is ordinary."))])
    seen = scripted([text, _reply(P.SUBMIT, REPORT)])
    rep, _ = P.run("sys", "user", [], Report)
    assert rep.verdict == "anomaly_review" and P.SUBMIT in seen[1]["messages"][-1]["content"]
