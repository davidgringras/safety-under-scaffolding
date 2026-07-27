"""
Unit tests for the blinding protocol.
"""

import tempfile
from pathlib import Path

import pytest
from scaffold_safety.blinding.protocol import BlindingProtocol


class TestBlindingProtocol:
    def test_blind_response(self):
        protocol = BlindingProtocol(seed=42)
        blinded = protocol.blind_response(
            sanitized_response="The answer is B.",
            user_prompt="What is 2+2?",
            benchmark_id="truthfulqa",
            model_id="opus",
            config_id="direct",
            case_id="case_001",
        )
        assert blinded.response_uuid
        assert blinded.sanitized_response == "The answer is B."
        assert blinded.model_id == "opus"
        assert protocol.n_blinded == 1

    def test_shuffled_order(self):
        protocol = BlindingProtocol(seed=42)
        for i in range(10):
            protocol.blind_response(
                sanitized_response=f"Response {i}",
                user_prompt="Q",
                benchmark_id="test",
                model_id="m",
                config_id="c",
                case_id=f"case_{i:03d}",
            )
        shuffled = protocol.get_shuffled_for_judging()
        assert len(shuffled) == 10
        # Should be deterministically shuffled (not in original order)
        original_order = [f"case_{i:03d}" for i in range(10)]
        shuffled_order = [s.case_id for s in shuffled]
        # Very unlikely to be in original order with seed=42 and 10 items
        assert shuffled_order != original_order or len(shuffled) <= 1

    def test_seal_mapping(self):
        protocol = BlindingProtocol(seed=42)
        protocol.blind_response(
            sanitized_response="Test",
            user_prompt="Q",
            benchmark_id="test",
            model_id="m",
            config_id="c",
            case_id="case_001",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            sealed = protocol.seal_mapping(Path(tmpdir))
            assert sealed.sha256_hash
            assert sealed.file_path
            assert sealed.file_path.exists()

            # Verify
            is_valid = BlindingProtocol.verify_sealed_mapping(
                sealed.file_path, sealed.sha256_hash,
            )
            assert is_valid is True

    def test_verify_tampered_fails(self):
        protocol = BlindingProtocol(seed=42)
        protocol.blind_response(
            sanitized_response="Test",
            user_prompt="Q",
            benchmark_id="test",
            model_id="m",
            config_id="c",
            case_id="case_001",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            sealed = protocol.seal_mapping(Path(tmpdir))
            # Tamper with the file
            with open(sealed.file_path, "a") as f:
                f.write("\ntampered")
            is_valid = BlindingProtocol.verify_sealed_mapping(
                sealed.file_path, sealed.sha256_hash,
            )
            assert is_valid is False
