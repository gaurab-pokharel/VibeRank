from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from itertools import combinations
import shutil
from typing import Any
import random

import pandas as pd
import yaml
import math

@dataclass
class MIMICRankCentralityConfig:
    dataset_name: str
    raw_root: Path
    processed_root: Path
    prompt_filename: str = "prompt.txt"
    selected_patients_filename: str = "medical_cohort_30.csv"
    json_dirname: str = "medical_jsons"
    responses_dirname: str = "rc_responses"
    flat_selected_dirname: str = "_selected_jsons"
    clear_flat_selected_dir: bool = True
    selected_filters: dict[str, Any] = field(default_factory=dict)
    run_settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "MIMICRankCentralityConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            dataset_name=data["dataset_name"],
            raw_root=Path(data["raw_root"]),
            processed_root=Path(data["processed_root"]),
            prompt_filename=data.get("prompt_filename", "prompt.txt"),
            selected_patients_filename=data.get(
                "selected_patients_filename", "medical_cohort_30.csv"
            ),
            json_dirname=data.get("json_dirname", "medical_jsons"),
            responses_dirname=data.get("responses_dirname", "rc_responses"),
            flat_selected_dirname=data.get("flat_selected_dirname", "_selected_jsons"),
            clear_flat_selected_dir=data.get("clear_flat_selected_dir", True),
            selected_filters=data.get("selected_filters", {}),
            run_settings=data.get("run_settings", {}),
        )

    @property
    def raw_dataset_dir(self) -> Path:
        return self.raw_root / self.dataset_name

    @property
    def processed_dataset_dir(self) -> Path:
        return self.processed_root / self.dataset_name

    @property
    def prompt_path(self) -> Path:
        return self.raw_root / self.prompt_filename

    @property
    def selected_patients_path(self) -> Path:
        return self.processed_dataset_dir / self.selected_patients_filename

    @property
    def json_source_dir(self) -> Path:
        return self.raw_root / self.json_dirname

    @property
    def responses_dir(self) -> Path:
        return self.raw_dataset_dir / self.responses_dirname

    @property
    def flat_selected_dir(self) -> Path:
        return self.responses_dir / self.flat_selected_dirname


class MIMICRankCentralityDataLoader:
    """
    Data loader for MIMIC-IV ED triage Rank Centrality experiments.

    Same patient loading and JSON prep as MIMICPairwiseDataLoader,
    but the tie sheet is always C(n, 2) unordered pairs.

    The interface mirrors RankCentralityDataLoader (homelessness)
    so that downstream comparators can consume either interchangeably.
    """

    def __init__(self, config: MIMICRankCentralityConfig):
        self.config = config
        self._selected_df: pd.DataFrame | None = None
        self._items: list[str] | None = None
        self._pairs: list[tuple[str, str]] | None = None

    def set_fraction_pairs(self,fraction_pairs):
        self.fraction_pairs = fraction_pairs

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "MIMICRankCentralityDataLoader":
        return cls(MIMICRankCentralityConfig.from_yaml(config_path))

    # ── data loading ──────────────────────────────────────────────

    def load_selected_patients(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.selected_patients_path).copy()

        df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
        if "esi_acuity" in df.columns:
            df["esi_acuity"] = pd.to_numeric(df["esi_acuity"], errors="coerce")

        filt = self.config.selected_filters or {}
        if filt.get("drop_missing_stay_id", True):
            df = df.dropna(subset=["stay_id"])
        if filt.get("drop_missing_acuity", True):
            df = df.dropna(subset=["esi_acuity"])

        df = df.copy()
        df["uid"] = df["stay_id"].astype(int).astype(str)

        sort_cols = [c for c in ["esi_acuity", "uid"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols).reset_index(drop=True)

        self._selected_df = df
        return df

    @property
    def selected_df(self) -> pd.DataFrame:
        if self._selected_df is None:
            return self.load_selected_patients()
        return self._selected_df

    # ── JSON preparation ──────────────────────────────────────────

    def prepare_flat_selected_jsons(self) -> Path:
        flat_dir = self.config.flat_selected_dir
        flat_dir.mkdir(parents=True, exist_ok=True)

        if self.config.clear_flat_selected_dir:
            for p in flat_dir.glob("*.json"):
                p.unlink()

        missing = []

        for _, row in self.selected_df.iterrows():
            uid = row["uid"]

            src = self.config.json_source_dir / f"stay_{uid}.json"
            dst = flat_dir / f"{uid}.json"

            if not src.exists():
                missing.append(str(src))
                continue

            shutil.copy2(src, dst)

        if missing:
            raise FileNotFoundError(
                "Some selected JSON patient files were not found:\n"
                + "\n".join(missing)
            )

        return flat_dir

    # ── items and pairs ───────────────────────────────────────────

    def get_items(self) -> list[str]:
        if self._items is None:
            self._items = self.selected_df["uid"].astype(str).tolist()
        return self._items

    def build_pairs(self, seed: int | None = 42) -> list[tuple[str, str]]:
        """
        Build unordered C(n, 2) pairs over selected items.

        If self.fraction_pairs is:
        - 1 or None: use all pairs
        - between 0 and 1: randomly keep that fraction of pairs
        """

        items = self.get_items()
        pairs = list(combinations(items, 2))

        rng = random.Random(seed)

        if seed is not None:
            rng.shuffle(pairs)

        fraction_pairs = getattr(self, "fraction_pairs", 1)

        if fraction_pairs is None:
            fraction_pairs = 1

        if not (0 < fraction_pairs <= 1):
            raise ValueError(
                f"fraction_pairs must be in (0, 1], got {fraction_pairs}"
            )

        if fraction_pairs < 1:
            num_pairs_to_keep = math.ceil(len(pairs) * fraction_pairs)
            pairs = pairs[:num_pairs_to_keep]

        self._pairs = pairs
        return self._pairs
    
   

    @property
    def pairs(self) -> list[tuple[str, str]]:
        if self._pairs is None:
            return self.build_pairs()
        return self._pairs

    @property
    def num_items(self) -> int:
        return len(self.get_items())

    @property
    def num_pairs(self) -> int:
        return len(self.pairs)

    # ── orchestration ─────────────────────────────────────────────

    def prepare(self,seed) -> None:
        self.load_selected_patients()
        self.prepare_flat_selected_jsons()
        self.build_pairs(seed=seed)
        self.config.responses_dir.mkdir(parents=True, exist_ok=True)

    def get_comparator_kwargs(self) -> dict[str, Any]:
        return {
            "items": self.get_items(),
            "pairs": self.pairs,
            "data_folder": self.config.flat_selected_dir,
            "prompt_path": self.config.prompt_path,
            "results_folder": self.config.responses_dir,
        }

    def get_run_metadata(self) -> dict[str, Any]:
        return {
            "dataset_name": self.config.dataset_name,
            "num_items": self.num_items,
            "num_pairs": self.num_pairs,
            "prompt_path": str(self.config.prompt_path),
            "selected_patients_path": str(
                self.config.selected_patients_path
            ),
        }