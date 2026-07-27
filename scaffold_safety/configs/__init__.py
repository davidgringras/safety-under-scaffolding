"""Scaffold configurations: Direct, ReAct, Multi-Agent, Map-Reduce."""

from scaffold_safety.configs.base import BaseScaffold, ScaffoldResult
from scaffold_safety.configs.direct import DirectScaffold
from scaffold_safety.configs.react import ReActScaffold
from scaffold_safety.configs.multi_agent import MultiAgentScaffold
from scaffold_safety.configs.map_reduce import MapReduceScaffold

SCAFFOLD_REGISTRY: dict[str, type[BaseScaffold]] = {
    "direct": DirectScaffold,
    "react": ReActScaffold,
    "multi_agent": MultiAgentScaffold,
    "map_reduce": MapReduceScaffold,
}

__all__ = [
    "BaseScaffold",
    "ScaffoldResult",
    "DirectScaffold",
    "ReActScaffold",
    "MultiAgentScaffold",
    "MapReduceScaffold",
    "SCAFFOLD_REGISTRY",
]
