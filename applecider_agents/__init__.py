"""applecider-agents: triage agents for the AppleCiDEr models (FLARE now; ASTRANet next).

Agents are kept separate from the model packages. Each model module contributes its
tools, its reading guide and a rule-based verdict; the core supplies the object tools,
the typed Report, the LLM loop, one MCP server for all models, and the SkyPortal gate.
"""
from .core.report import Report  # noqa: F401
__version__ = "0.2.0"
