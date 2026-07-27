"""Scaffold implementations: Direct, ReAct, Multi-Agent, Map-Reduce."""

from pipeline.scaffolds.base import BaseScaffold
from pipeline.scaffolds.direct import DirectScaffold
from pipeline.scaffolds.react import ReActScaffold
from pipeline.scaffolds.multi_agent import MultiAgentScaffold
from pipeline.scaffolds.map_reduce import MapReduceScaffold

SCAFFOLD_REGISTRY: dict[str, type[BaseScaffold]] = {
    "direct": DirectScaffold,
    "react": ReActScaffold,
    "multi_agent": MultiAgentScaffold,
    "map_reduce": MapReduceScaffold,
}

__all__ = [
    "BaseScaffold",
    "DirectScaffold",
    "ReActScaffold",
    "MultiAgentScaffold",
    "MapReduceScaffold",
    "SCAFFOLD_REGISTRY",
]
