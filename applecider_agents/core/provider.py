"""Tool-use loop against any OpenAI-compatible chat endpoint (vLLM, llama.cpp, Ollama, a hosted
API), so the agents can run on an open-weight model served next to the data. Selected with
APPLECIDER_AGENTS_PROVIDER=openai and APPLECIDER_AGENTS_BASE_URL (e.g. http://localhost:8000/v1);
the Anthropic loop in `loop.py` stays the default. Standard library only.

The model drives the same deterministic tools and ends by calling `submit_report`, whose
arguments are validated against the typed Report; a report that fails validation is returned
to the model once with the error."""
from __future__ import annotations
import json, urllib.request
from . import config as C

SUBMIT = "submit_report"


def _post(path: str, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if C.API_KEY:
        headers["Authorization"] = f"Bearer {C.API_KEY}"
    req = urllib.request.Request(C.BASE_URL.rstrip("/") + path, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=C.REQUEST_TIMEOUT) as r:
        return json.load(r)


def _function(name, description, schema):
    return dict(type="function", function=dict(name=name, description=description, parameters=schema))


def run(system: str, user: str, tools, report_cls, max_iterations: int | None = None):
    """Returns (report, record): record holds the tool calls, token counts and the model that served each turn."""
    if not C.BASE_URL:
        raise RuntimeError("APPLECIDER_AGENTS_BASE_URL must be set for the openai provider")
    by_name = {t.name: t for t in tools}
    specs = [_function(t.name, t.description, t.input_schema) for t in tools]
    specs.append(_function(SUBMIT, "Return the final Report. Call this once, after the tools, with every field filled.",
                           report_cls.model_json_schema()))
    messages = [dict(role="system", content=system + f"\n\nWhen you are done, call the {SUBMIT} function with the Report; do not answer in plain text."),
                dict(role="user", content=user)]
    calls, usage, served, retries = [], dict(input=0, output=0), [], 0
    limit = max_iterations or C.MAX_ITERATIONS
    for turn in range(limit + 2):
        payload = dict(model=C.MODEL, messages=messages, tools=specs, temperature=C.TEMPERATURE, max_tokens=4096)
        payload["tool_choice"] = "auto" if turn < limit else dict(type="function", function=dict(name=SUBMIT))
        out = _post("/chat/completions", payload)
        u = out.get("usage") or {}
        usage["input"] += u.get("prompt_tokens", 0); usage["output"] += u.get("completion_tokens", 0)
        served.append(out.get("model"))
        msg = out["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        messages.append(dict(role="assistant", content=msg.get("content") or "", tool_calls=tool_calls) if tool_calls
                        else dict(role="assistant", content=msg.get("content") or ""))
        if not tool_calls:
            messages.append(dict(role="user", content=f"Call the {SUBMIT} function with the Report now."))
            continue
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError as e:
                args, result = {}, dict(error=f"arguments are not valid JSON: {e}")
            else:
                result = None
            if name == SUBMIT and result is None:
                try:
                    report = report_cls.model_validate(args)
                    return report, dict(tool_calls=calls, usage=usage, served_by=sorted({s for s in served if s}), turns=turn + 1, report_retries=retries)
                except Exception as e:  # noqa: BLE001 - returned to the model
                    retries += 1; result = dict(error=f"Report failed validation: {str(e)[:800]}")
                    if retries > 2:
                        raise RuntimeError(f"no valid Report after {retries} attempts: {str(e)[:300]}") from e
            elif result is None:
                calls.append(dict(tool=name, input=args))
                result = by_name[name].call(args) if name in by_name else dict(error=f"unknown tool {name}")
            messages.append(dict(role="tool", tool_call_id=tc.get("id"), content=json.dumps(result, default=str)))
    raise RuntimeError(f"no Report after {limit + 2} turns")
