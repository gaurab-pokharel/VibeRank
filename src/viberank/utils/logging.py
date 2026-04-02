from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any


class JSONLResponseLogger:
    """
    Append-only logger for raw LLM pairwise comparison responses.

    This version assumes the tie sheet is directed:
        (A, B) and (B, A) are distinct comparisons.

    So:
    - pair_key is directional
    - pair_view is stored once per directed pair
    - response records are logged per directed pair
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
        self._seen_views: set[str] = set()

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
        item_a: str,
        item_b: str,
        order: str,
        left_item: str,
        right_item: str,
        prompt: str,
    ) -> None:
        """
        Registers the rendered prompt once for one directed comparison.

        In the full ordered-matrix setup:
            item_a -> item_b
        is distinct from:
            item_b -> item_a

        Usually order will just be "as_given".
        """
        pair_key = self.pair_key(item_a, item_b)

        if pair_key in self._seen_views:
            return

        record = {
            "event": "pair_view",
            "timestamp": self._now(),
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
        self._seen_views.add(pair_key)

    def log_response(
        self,
        *,
        item_a: str,
        item_b: str,
        order: str,
        repeat_index: int,
        seed: Optional[int],
        raw_response: str,
        left_item: Optional[str] = None,
        right_item: Optional[str] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        record = {
            "event": "response",
            "timestamp": self._now(),
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

    def __enter__(self) -> "JSONLResponseLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()