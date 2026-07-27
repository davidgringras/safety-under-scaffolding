"""
Blinding protocol for assessor-blinded evaluation.

Implements clinical-trial-inspired single-blind methodology:
1. Generate a UUID for each response.
2. Create a sealed mapping file (response_uuid -> model_id, config_id, case_id).
3. Compute SHA-256 hash of the mapping file (published before scoring).
4. Shuffle response order for judging (deterministic with fixed seed).
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BlindedResponse:
    """A response with its true identity hidden behind a UUID."""
    response_uuid: str
    sanitized_response: str
    user_prompt: str
    benchmark_id: str
    # Sealed -- never sent to judge
    model_id: str
    config_id: str
    case_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SealedMapping:
    """The mapping file and its integrity hash."""
    mapping: dict[str, dict[str, str]]
    sha256_hash: str
    sealed_at: str
    file_path: Path | None = None


class BlindingProtocol:
    """Generate blinded response sets and seal the mapping."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = random.Random(self.seed)
        self._mapping: dict[str, dict[str, Any]] = {}
        self._blinded_responses: list[BlindedResponse] = []
        self._lock = threading.Lock()

    def blind_response(
        self,
        sanitized_response: str,
        user_prompt: str,
        benchmark_id: str,
        model_id: str,
        config_id: str,
        case_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> BlindedResponse:
        """Assign a UUID to a response and register it in the mapping."""
        response_uuid = str(uuid.uuid4())
        blinded = BlindedResponse(
            response_uuid=response_uuid,
            sanitized_response=sanitized_response,
            user_prompt=user_prompt,
            benchmark_id=benchmark_id,
            model_id=model_id,
            config_id=config_id,
            case_id=case_id,
            metadata=metadata or {},
        )
        with self._lock:
            self._mapping[response_uuid] = {
                "model_id": model_id,
                "config_id": config_id,
                "case_id": case_id,
                "benchmark_id": benchmark_id,
                **(metadata or {}),
            }
            self._blinded_responses.append(blinded)
        return blinded

    def get_shuffled_for_judging(self) -> list[BlindedResponse]:
        """Return all blinded responses in deterministic shuffled order."""
        shuffled = list(self._blinded_responses)
        self._rng.shuffle(shuffled)
        return shuffled

    def seal_mapping(self, output_dir: Path) -> SealedMapping:
        """Write the mapping file and compute its SHA-256 hash."""
        assert len(self._mapping) > 0, "No responses have been blinded yet."
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sealed_at = datetime.now(timezone.utc).isoformat()
        mapping_data = {
            "sealed_at": sealed_at,
            "seed": self.seed,
            "n_responses": len(self._mapping),
            "mapping": self._mapping,
        }
        mapping_json = json.dumps(mapping_data, sort_keys=True, indent=2)
        mapping_bytes = mapping_json.encode("utf-8")

        file_path = output_dir / "blinding_mapping.json"
        with open(file_path, "w") as f:
            f.write(mapping_json)

        sha256_hash = hashlib.sha256(mapping_bytes).hexdigest()

        hash_file = output_dir / "blinding_mapping.sha256"
        with open(hash_file, "w") as f:
            f.write(f"{sha256_hash}  blinding_mapping.json\n")
            f.write(f"Sealed at: {sealed_at}\n")
            f.write(f"N responses: {len(self._mapping)}\n")

        return SealedMapping(
            mapping=dict(self._mapping),
            sha256_hash=sha256_hash,
            sealed_at=sealed_at,
            file_path=file_path,
        )

    @staticmethod
    def verify_sealed_mapping(mapping_file: Path, expected_hash: str) -> bool:
        """Verify that a mapping file matches its expected hash."""
        with open(mapping_file, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        return actual_hash == expected_hash

    @property
    def n_blinded(self) -> int:
        return len(self._mapping)
