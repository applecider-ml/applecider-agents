"""The LLM tool-use loop, shared by every model agent: Claude drives the object tools plus
the model's own tools and returns a typed Report; the model module supplies the system
prompt and a rule-based `triage` that runs the same tools without any LLM (the baseline,
the smoke test, and what the MCP server serves)."""
from __future__ import annotations
import json, time
from . import config as C
from .report import Report


def _client():
    import anthropic
    return anthropic.Anthropic()


def _log(record: dict):
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.LOG_DIR / "reports.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def triage(model: str, obj_id: str, horizon_days: float = 100.0, mock: bool = False, client=None, blind: bool = False) -> Report:
    """One object through one model agent: rule-based when mock, else the Claude tool-use loop.
    blind=True hides the catalogue and TNS tools (evaluation on objects with a known type)."""
    from . import registry
    M = registry.get(model); t0 = time.time()
    if mock:
        rep = M.triage(obj_id, horizon_days, blind=blind) if blind else M.triage(obj_id, horizon_days); _log(dict(model=model, obj_id=obj_id, mode="rules", seconds=round(time.time() - t0, 1), report=rep.model_dump())); return rep
    from .object_tools import OBJECT_TOOLS
    tools = [x for x in OBJECT_TOOLS if not (blind and x.name in ("catalog_lookup", "tns_lookup"))] + list(M.tools)
    system = M.system_prompt + ("\n\nEvaluation mode: the catalogue tools are unavailable; judge from the light curve, the model outputs and the context alone." if blind else "")
    user = f"Triage {obj_id} using the first {horizon_days:g} days of its light curve (horizon_days={horizon_days:g} in every tool call). Return the Report."
    if C.PROVIDER == "openai":
        from . import provider
        rep, rec = provider.run(system, user, tools, M.report_cls or Report)
        _log(dict(model=model, obj_id=obj_id, mode="llm", provider="openai", llm=C.MODEL, seconds=round(time.time() - t0, 1), report=rep.model_dump(), **rec))
        return rep
    client = client or _client()
    kw = dict(model=C.MODEL, max_tokens=4096, system=system, tools=tools, output_format=M.report_cls or Report,
              thinking={"type": "adaptive"}, output_config={"effort": C.EFFORT}, max_iterations=C.MAX_ITERATIONS,
              cache_control={"type": "ephemeral"},
              messages=[{"role": "user", "content": f"Triage {obj_id} using the first {horizon_days:g} days of its light curve (horizon_days={horizon_days:g} in every tool call). Return the Report."}])
    if C.FALLBACKS:
        kw.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    runner = client.beta.messages.tool_runner(**kw)
    calls, usage, served = [], dict(input=0, output=0, cache_read=0, cache_write=0), set()
    for message in runner:
        u = message.usage; served.add(getattr(message, "model", None))
        usage["input"] += u.input_tokens; usage["output"] += u.output_tokens
        usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        for b in message.content:
            if getattr(b, "type", "") == "tool_use":
                calls.append(dict(tool=b.name, input=b.input))
    final = runner.until_done()
    if final.stop_reason == "refusal":
        raise RuntimeError(f"model refused: {getattr(final, 'stop_details', None)}")
    rep = final.parsed_output
    if rep is None:
        text = " ".join(getattr(b, "text", "") for b in final.content)
        raise RuntimeError(f"no structured Report returned (stop_reason={final.stop_reason}); text: {text[:500]}")
    _log(dict(model=model, obj_id=obj_id, mode="llm", llm=C.MODEL, served_by=sorted(m for m in served if m), effort=C.EFFORT, seconds=round(time.time() - t0, 1), usage=usage, tool_calls=calls, report=rep.model_dump()))
    return rep
