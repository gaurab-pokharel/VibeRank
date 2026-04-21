from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any


class JSONLResponseLogger:
    """
    Append-only logger for raw LLM pairwise comparison responses.

    Resume logic:
    - each response record stores tie_index and repeat_index
    - on restart, we scan the JSONL and reconstruct which
      (tie_index, repeat_index) attempts already exist
    - any logged response counts as completed, even if it has error != None
    """

    def __init__(
        self,
        log_path: str | Path,
        *,
        flush_every: int = 1,
        store_prompts: bool = True,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.flush_every = flush_every
        self.store_prompts = store_prompts
        self._buffer: list[str] = []
        self._seen_views: set[tuple[int, str]] = set()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def pair_key(item_a: str, item_b: str) -> str:
        """
        Directional pair key.
        """
        return f"{str(item_a)}__{str(item_b)}"

    @staticmethod
    def prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _write_record(self, record: dict[str, Any]) -> None:
        self._buffer.append(json.dumps(record, ensure_ascii=False) + "\n")
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.writelines(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()

    def log_run_start(
        self,
        *,
        run_id: str,
        dataset_name: str,
        model_name: str,
        prompt_version: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        record = {
            "event": "run_start",
            "timestamp": self._now(),
            "run_id": run_id,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "prompt_version": prompt_version,
        }
        if extra is not None:
            record["extra"] = extra
        self._write_record(record)

    def register_pair_view(
        self,
        *,
        tie_index: int,
        item_a: str,
        item_b: str,
        order: str,
        left_item: str,
        right_item: str,
        prompt: str,
    ) -> None:
        """
        Stores the rendered prompt once per tie_index.
        """
        pair_key = self.pair_key(item_a, item_b)
        seen_key = (tie_index, pair_key)

        if seen_key in self._seen_views:
            return

        record = {
            "event": "pair_view",
            "timestamp": self._now(),
            "tie_index": tie_index,
            "pair_key": pair_key,
            "pair_direction": [str(item_a), str(item_b)],
            "order": order,
            "left_item": str(left_item),
            "right_item": str(right_item),
            "prompt_hash": self.prompt_hash(prompt),
        }

        if self.store_prompts:
            record["prompt"] = prompt

        self._write_record(record)
        self._seen_views.add(seen_key)

    def log_response(
        self,
        *,
        tie_index: int,
        item_a: str,
        item_b: str,
        order: str,
        repeat_index: int,
        seed: Optional[int],
        raw_response: Optional[str],
        left_item: Optional[str] = None,
        right_item: Optional[str] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        record = {
            "event": "response",
            "timestamp": self._now(),
            "tie_index": tie_index,
            "pair_key": self.pair_key(item_a, item_b),
            "pair_direction": [str(item_a), str(item_b)],
            "order": order,
            "repeat_index": repeat_index,
            "seed": seed,
            "raw_response": raw_response,
            "latency_ms": latency_ms,
            "error": error,
        }

        if left_item is not None:
            record["left_item"] = str(left_item)
        if right_item is not None:
            record["right_item"] = str(right_item)
        if extra is not None:
            record["extra"] = extra

        self._write_record(record)

    def log_run_end(
        self,
        *,
        run_id: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        record = {
            "event": "run_end",
            "timestamp": self._now(),
            "run_id": run_id,
        }
        if extra is not None:
            record["extra"] = extra
        self._write_record(record)
        self.flush()

    def load_completed_repeats(self) -> dict[int, set[int]]:
        """
        Reconstruct completed work from the JSONL log.

        Returns:
            completed[tie_index] = {repeat_index_1, repeat_index_2, ...}

        Any logged response counts as completed, even if error != None.
        This matches your desired behavior.
        """
        completed: dict[int, set[int]] = {}

        if not self.log_path.exists():
            return completed

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # ignore a possibly truncated final line after a crash
                    continue

                if record.get("event") != "response":
                    continue

                tie_index = record.get("tie_index")
                repeat_index = record.get("repeat_index")

                if tie_index is None or repeat_index is None:
                    continue

                if tie_index not in completed:
                    completed[tie_index] = set()

                completed[tie_index].add(repeat_index)

        return completed

    def __enter__(self) -> "JSONLResponseLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()