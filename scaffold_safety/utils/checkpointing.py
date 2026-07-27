"""
Checkpoint management for experiment state.

Supports incremental writes with fsync for crash safety,
resume from interruption, and deduplication.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class Checkpointer:
    """Thread-safe JSONL checkpointing with deduplication support.

    Parameters
    ----------
    output_path : Path
        Path to the JSONL results file.
    key_fields : tuple[str, ...]
        Fields that together form a unique key for deduplication.
    """

    def __init__(
        self,
        output_path: Path,
        key_fields: tuple[str, ...] = ("case_id", "model_id", "config_id"),
    ) -> None:
        self.output_path = Path(output_path)
        self.key_fields = key_fields
        self._lock = threading.Lock()
        self._completed_keys: set[str] = set()
        self._load_existing()

    def _make_key(self, record: dict[str, Any]) -> str:
        return "|".join(str(record.get(f, "")) for f in self.key_fields)

    def _load_existing(self) -> None:
        """Load completed keys from existing checkpoint file."""
        if not self.output_path.exists():
            return
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._completed_keys.add(self._make_key(record))
                except json.JSONDecodeError:
                    continue

    def is_completed(self, record: dict[str, Any]) -> bool:
        """Check if this case has already been processed."""
        return self._make_key(record) in self._completed_keys

    def save(self, record: dict[str, Any]) -> None:
        """Append a result to the checkpoint file (thread-safe, crash-safe)."""
        key = self._make_key(record)
        with self._lock:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._completed_keys.add(key)

    def save_batch(self, records: list[dict[str, Any]]) -> None:
        """Append multiple results atomically."""
        with self._lock:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "a") as f:
                for record in records:
                    f.write(json.dumps(record, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            for record in records:
                self._completed_keys.add(self._make_key(record))

    @property
    def n_completed(self) -> int:
        return len(self._completed_keys)

    def deduplicate(self) -> tuple[int, int]:
        """Deduplicate the checkpoint file, keeping the last entry per key.

        Returns
        -------
        tuple[int, int]
            (original_count, deduplicated_count)
        """
        if not self.output_path.exists():
            return 0, 0

        records: list[dict[str, Any]] = []
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        original_count = len(records)

        # Keep last occurrence per key
        seen: dict[str, int] = {}
        for idx, record in enumerate(records):
            seen[self._make_key(record)] = idx
        deduped = [records[i] for i in sorted(seen.values())]

        # Rewrite
        with open(self.output_path, "w") as f:
            for record in deduped:
                f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self._completed_keys = {self._make_key(r) for r in deduped}
        return original_count, len(deduped)
