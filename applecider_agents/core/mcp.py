"""All AppleCiDEr model agents as MCP tools, for SkyPortal's assistant or any MCP client.

The language model is the client's (SkyPortal's assistant runs its own); nothing here
calls an LLM. Object tools are shared; each registered model adds `<model>_triage` (rule
verdict + evidence + reading guide) and its own tools.

Run:  applecider-agents mcp [--transport stdio|streamable-http] [--port 8765] [--models flare,...]
"""
from __future__ import annotations
import json
from mcp.server.mcpserver import MCPServer
from . import registry
from .object_tools import OBJECT_TOOLS


def _j(d):
    return json.loads(json.dumps(d, default=float))


def build(models=None) -> MCPServer:
    names = models or ["flare"]
    server = MCPServer("applecider", title="AppleCiDEr model agents",
                       instructions="Tools return measured numbers from AppleCiDEr's models and public catalogues. "
                                    "Cite them; do not invent values. For one object call <model>_triage first, then drill in with the other tools.")
    for t in OBJECT_TOOLS:
        _wrap(server, t)
    for name in names:
        M = registry.get(name)
        for t in M.tools:
            _wrap(server, t)
        def make_triage(M):
            def triage(obj_id: str, horizon_days: float = 100.0) -> dict:
                rep = M.triage(obj_id, horizon_days).model_dump(); rep["reading_guide"] = M.reading_guide; return _j(rep)
            triage.__name__ = f"{M.name}_triage"
            triage.__doc__ = f"Rule-based triage of one object with {M.name.upper()}: {M.description} Returns verdict (ordinary / needs_spectrum / anomaly_review / likely_stellar_or_agn / already_classified / insufficient_data), priority 0-3, evidence per tool, and the guide for reading the numbers. horizon_days limits the light curve to its first N days."
            return triage
        server.tool(name=f"{M.name}_triage")(make_triage(M))
        for fn in M.mcp_extra:
            server.tool(name=fn.__name__, description=fn.__doc__)(fn)
    return server


def _wrap(server, tool):
    """Expose a @beta_tool as an MCP tool with the same name, description and (typed) signature."""
    fn = tool.func if hasattr(tool, "func") else tool
    def call(**kwargs):
        return _j(tool.call(kwargs))
    call.__name__ = tool.name; call.__doc__ = tool.description
    import inspect
    call.__signature__ = inspect.signature(fn)
    server.tool(name=tool.name, description=tool.description)(call)


def main(transport: str = "stdio", port: int = 8765, host: str = "127.0.0.1", models=None):
    server = build(models)
    if transport == "stdio":
        server.run("stdio")
    else:
        server.settings.host, server.settings.port = host, port
        server.run(transport)
