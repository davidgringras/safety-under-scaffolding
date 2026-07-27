"""Benchmark loaders and scorers for safety evaluation."""

from scaffold_safety.benchmarks.base import BenchmarkSpec, BenchmarkLoader
from scaffold_safety.benchmarks.truthfulqa import TruthfulQABenchmark
from scaffold_safety.benchmarks.bbq import BBQBenchmark
from scaffold_safety.benchmarks.sycophancy import SycophancyBenchmark
from scaffold_safety.benchmarks.xstest import XSTestBenchmark

BENCHMARK_REGISTRY: dict[str, type[BenchmarkLoader]] = {
    "truthfulqa": TruthfulQABenchmark,
    "bbq": BBQBenchmark,
    "sycophancy": SycophancyBenchmark,
    "xstest": XSTestBenchmark,
}

__all__ = [
    "BenchmarkSpec",
    "BenchmarkLoader",
    "TruthfulQABenchmark",
    "BBQBenchmark",
    "SycophancyBenchmark",
    "XSTestBenchmark",
    "BENCHMARK_REGISTRY",
]
