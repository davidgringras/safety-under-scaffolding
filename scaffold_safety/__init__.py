"""
ScaffoldSafety: Open Evaluation Framework for Deployment-Configuration-Aware Safety Testing.

Measures how inference-time scaffolding configurations (direct API, ReAct agent,
multi-agent debate, map-reduce delegation) alter frontier language model safety
properties across standardised benchmarks.

Developed as part of "Safety Under Scaffolding: Do Safety Benchmark Scores
Transfer Across Inference Paradigms?" (Gringras, 2026).

Quick start::

    from scaffold_safety import ScaffoldSafetyEval

    eval = ScaffoldSafetyEval(
        models=["claude-opus-4-6", "gpt-5.2"],
        configs=["direct", "react"],
        benchmarks=["truthfulqa", "sycophancy"],
    )
    results = eval.run()
    results.scorecard()

For custom scaffolds and benchmarks::

    from scaffold_safety import register_scaffold, register_benchmark
    from scaffold_safety.configs.base import BaseScaffold, ScaffoldResult
    from scaffold_safety.benchmarks.base import BenchmarkLoader, BenchmarkCase, ScoreResult

See https://github.com/dgringras/scaffold-safety for full documentation.
"""

__version__ = "0.1.0"

from scaffold_safety.core import ScaffoldSafetyEval
from scaffold_safety.results import EvalResults

# Re-export commonly needed types for custom scaffolds/benchmarks
from scaffold_safety.configs.base import BaseScaffold, ScaffoldResult
from scaffold_safety.benchmarks.base import (
    BenchmarkLoader,
    BenchmarkCase,
    BenchmarkSpec,
    ScoreResult,
)
from scaffold_safety.utils.providers import ModelSpec, resolve_model
from scaffold_safety.analysis.scorecard import ScaffoldSafetyScorecard
from scaffold_safety.analysis.statistics import SafetyRate, EffectSize, TOSTResult

# Registry access and convenience registration functions
from scaffold_safety.configs import SCAFFOLD_REGISTRY
from scaffold_safety.benchmarks import BENCHMARK_REGISTRY


def register_scaffold(scaffold_id: str, scaffold_cls: type[BaseScaffold]) -> None:
    """Register a custom scaffold configuration.

    Parameters
    ----------
    scaffold_id : str
        Short identifier (e.g. ``"my_scaffold"``). Must be unique.
    scaffold_cls : type[BaseScaffold]
        A class that inherits from :class:`BaseScaffold`.

    Raises
    ------
    TypeError
        If *scaffold_cls* does not inherit from :class:`BaseScaffold`.
    ValueError
        If *scaffold_id* is already registered.

    Examples
    --------
    ::

        from scaffold_safety import register_scaffold
        from scaffold_safety.configs.base import BaseScaffold, ScaffoldResult

        class MyScaffold(BaseScaffold):
            @property
            def scaffold_id(self) -> str:
                return "my_scaffold"

            def run(self, system_prompt, user_prompt, model, **kwargs):
                ...

        register_scaffold("my_scaffold", MyScaffold)
    """
    if not (isinstance(scaffold_cls, type) and issubclass(scaffold_cls, BaseScaffold)):
        raise TypeError(
            f"scaffold_cls must be a subclass of BaseScaffold, "
            f"got {type(scaffold_cls).__name__}: {scaffold_cls}"
        )
    if scaffold_id in SCAFFOLD_REGISTRY:
        raise ValueError(
            f"Scaffold '{scaffold_id}' is already registered. "
            f"Use a different ID or unregister the existing one first."
        )
    SCAFFOLD_REGISTRY[scaffold_id] = scaffold_cls


def register_benchmark(benchmark_id: str, loader_cls: type[BenchmarkLoader]) -> None:
    """Register a custom benchmark loader.

    Parameters
    ----------
    benchmark_id : str
        Short identifier (e.g. ``"my_benchmark"``). Must be unique.
    loader_cls : type[BenchmarkLoader]
        A class that inherits from :class:`BenchmarkLoader`.

    Raises
    ------
    TypeError
        If *loader_cls* does not inherit from :class:`BenchmarkLoader`.
    ValueError
        If *benchmark_id* is already registered.

    Examples
    --------
    ::

        from scaffold_safety import register_benchmark
        from scaffold_safety.benchmarks.base import BenchmarkLoader, BenchmarkCase, ScoreResult

        class MyBenchmark(BenchmarkLoader):
            @property
            def spec(self):
                ...

            def load_cases(self, data_path):
                ...

            def score(self, case, response):
                ...

        register_benchmark("my_benchmark", MyBenchmark)
    """
    if not (isinstance(loader_cls, type) and issubclass(loader_cls, BenchmarkLoader)):
        raise TypeError(
            f"loader_cls must be a subclass of BenchmarkLoader, "
            f"got {type(loader_cls).__name__}: {loader_cls}"
        )
    if benchmark_id in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Benchmark '{benchmark_id}' is already registered. "
            f"Use a different ID or unregister the existing one first."
        )
    BENCHMARK_REGISTRY[benchmark_id] = loader_cls


__all__ = [
    # Core API
    "ScaffoldSafetyEval",
    "EvalResults",
    # Registration
    "register_scaffold",
    "register_benchmark",
    "SCAFFOLD_REGISTRY",
    "BENCHMARK_REGISTRY",
    # Base classes for extension
    "BaseScaffold",
    "ScaffoldResult",
    "BenchmarkLoader",
    "BenchmarkCase",
    "BenchmarkSpec",
    "ScoreResult",
    "ModelSpec",
    "resolve_model",
    # Result types
    "ScaffoldSafetyScorecard",
    "SafetyRate",
    "EffectSize",
    "TOSTResult",
    # Version
    "__version__",
]
