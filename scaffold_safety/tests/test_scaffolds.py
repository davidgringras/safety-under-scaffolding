"""
Integration tests for scaffold configurations.

Tests that each scaffold correctly produces ScaffoldResult objects
with the expected structure (uses dry_run mode, no API calls).
"""

import pytest
from scaffold_safety.configs import SCAFFOLD_REGISTRY
from scaffold_safety.configs.base import ScaffoldResult
from scaffold_safety.utils.providers import ModelSpec


# Shared test fixtures
TEST_MODEL = ModelSpec(
    model_id="test_model",
    litellm_id="test/model",
    provider="test",
    cost_per_1k_input=0.001,
    cost_per_1k_output=0.002,
)

TEST_SYSTEM_PROMPT = "You are a helpful assistant."
TEST_USER_PROMPT = "What is 2 + 2? (A) 3 (B) 4 (C) 5 (D) 6"


class TestDirectScaffold:
    """Test the Direct scaffold."""

    def test_scaffold_id(self):
        scaffold = SCAFFOLD_REGISTRY["direct"]()
        assert scaffold.scaffold_id == "direct"

    def test_dry_run(self):
        scaffold = SCAFFOLD_REGISTRY["direct"]()
        result = scaffold.run(
            TEST_SYSTEM_PROMPT, TEST_USER_PROMPT, TEST_MODEL,
            dry_run=True,
        )
        assert isinstance(result, ScaffoldResult)
        assert result.n_api_calls == 1
        assert len(result.trace) == 3  # system, user, assistant
        assert result.call_ids
        assert "[DRY RUN]" in result.final_response


class TestReActScaffold:
    """Test the ReAct scaffold."""

    def test_scaffold_id(self):
        scaffold = SCAFFOLD_REGISTRY["react"]()
        assert scaffold.scaffold_id == "react"

    def test_dry_run(self):
        scaffold = SCAFFOLD_REGISTRY["react"]()
        result = scaffold.run(
            TEST_SYSTEM_PROMPT, TEST_USER_PROMPT, TEST_MODEL,
            dry_run=True,
        )
        assert isinstance(result, ScaffoldResult)
        assert result.n_api_calls >= 1
        assert result.metadata["scaffold"] == "react"


class TestMultiAgentScaffold:
    """Test the Multi-Agent scaffold."""

    def test_scaffold_id(self):
        scaffold = SCAFFOLD_REGISTRY["multi_agent"]()
        assert scaffold.scaffold_id == "multi_agent"

    def test_dry_run(self):
        scaffold = SCAFFOLD_REGISTRY["multi_agent"]()
        result = scaffold.run(
            TEST_SYSTEM_PROMPT, TEST_USER_PROMPT, TEST_MODEL,
            dry_run=True,
        )
        assert isinstance(result, ScaffoldResult)
        # Multi-agent makes at least 2 calls (primary + critic)
        assert result.n_api_calls >= 2
        assert result.metadata["scaffold"] == "multi_agent"


class TestMapReduceScaffold:
    """Test the Map-Reduce scaffold."""

    def test_scaffold_id(self):
        scaffold = SCAFFOLD_REGISTRY["map_reduce"]()
        assert scaffold.scaffold_id == "map_reduce"

    def test_dry_run(self):
        scaffold = SCAFFOLD_REGISTRY["map_reduce"]()
        result = scaffold.run(
            TEST_SYSTEM_PROMPT, TEST_USER_PROMPT, TEST_MODEL,
            dry_run=True,
        )
        assert isinstance(result, ScaffoldResult)
        # Map-reduce makes at least 3 calls (decompose + 1 chunk + reduce)
        assert result.n_api_calls >= 3
        assert result.metadata["scaffold"] == "map_reduce"


class TestScaffoldRegistry:
    """Test the scaffold registry."""

    def test_all_scaffolds_registered(self):
        expected = {"direct", "react", "multi_agent", "map_reduce"}
        assert set(SCAFFOLD_REGISTRY.keys()) == expected

    def test_all_scaffolds_instantiate(self):
        for name, cls in SCAFFOLD_REGISTRY.items():
            scaffold = cls()
            assert scaffold.scaffold_id == name
