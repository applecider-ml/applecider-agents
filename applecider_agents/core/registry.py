"""One entry per model agent. A model module builds a ModelAgent and registers it on import."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional

_REGISTRY: dict = {}


@dataclass
class ModelAgent:
    name: str                               # "flare", "astranet", ...
    description: str
    tools: List[object]                     # @beta_tool functions specific to the model
    triage: Callable                        # rule-based verdict: (obj_id, horizon_days) -> Report
    system_prompt: str                      # for the LLM loop: what the tools mean and how to read them
    reading_guide: str                      # the part of the prompt a foreign assistant needs (served by MCP)
    mcp_extra: List[object] = field(default_factory=list)   # plain functions to expose over MCP only (e.g. inline-photometry variants)

    def register(self):
        _REGISTRY[self.name] = self
        return self


def get(name: str) -> ModelAgent:
    if name not in _REGISTRY:
        import importlib
        importlib.import_module(f"applecider_agents.{name}")   # registers on import
    return _REGISTRY[name]


def names():
    return sorted(_REGISTRY)
