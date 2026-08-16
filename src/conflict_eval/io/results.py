"""Incremental, resumable JSONL result storage.

Results are written one record at a time, flushed immediately, so an
interrupted run leaves a valid, readable partial file rather than
corrupting an in-memory-only buffer (docs/methodology.md, section 9).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultWriter:
    """Append-only JSONL writer that tracks completed `record_key`s.

    On construction, any existing file at `path` is read to rebuild the
    set of already-completed keys, so `is_completed` reflects prior runs
    without holding the whole file in memory beyond that key set.
    """

    def __init__(self, path: str | Path, key_field: str = "record_key") -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed_keys = self._load_existing_keys()

    def _load_existing_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        keys: set[str] = set()
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if self.key_field in record:
                    keys.add(record[self.key_field])
        return keys

    def is_completed(self, key: str) -> bool:
        return key in self._completed_keys

    def write(self, record: dict[str, Any]) -> None:
        key = record.get(self.key_field)
        if key is None:
            raise ValueError(f"Record is missing '{self.key_field}'; cannot track resumability")
        if key in self._completed_keys:
            # Silently ignore a duplicate write for an already-completed
            # key rather than corrupting the file with a repeated record.
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._completed_keys.add(key)

    def __len__(self) -> int:
        return len(self._completed_keys)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
