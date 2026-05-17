from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
import shutil
from typing import Any
import pandas as pd
import yaml


@dataclass
class MIMICPairwiseConfig:
    dataset_name: str
    raw_root: Path
    processed_root: Path
    prompt_filename: str = "prompt_medical.txt"
    selected_patients_filename: str = "medical_cohort_10.csv"
    json_dirname: str = "medical_jsons"
    responses_dirname: str = "responses"
    flat_selected_dirname: str = "_selected_jsons"
    tie_sheet_mode: str = "full_ordered_matrix"
    clear_flat_selected_dir: bool = True
    selected_filters: dict[str, Any] | None = None
    run_settings: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "MIMICPairwiseConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            dataset_name=data["dataset_name"],
            raw_root=Path(data["raw_root"]),
            processed_root=Path(data["processed_root"]),
            prompt_filename=data.get("prompt_filename", "prompt.txt"),
            selected_patients_filename=data.get("selected_patients_filename", "medical_cohort_10.csv"),
            json_dirname=data.get("json_dirname", "medical_jsons"),
            responses_dirname=data.get("responses_dirname", "responses"),
            flat_selected_dirname=data.get("flat_selected_dirname", "_selected_jsons"),
            tie_sheet_mode=data.get("tie_sheet_mode", "full_ordered_matrix"),
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
        """Directory where the per-patient JSONs were exported."""
        return self.raw_root / self.json_dirname

    @property
    def responses_dir(self) -> Path:
        return self.raw_dataset_dir / self.responses_dirname

    @property
    def flat_selected_dir(self) -> Path:
        return self.responses_dir / self.flat_selected_dirname


class MIMICPairwiseDataLoader:
    """
    Data loader for the MIMIC-IV ED triage pairwise-comparison setup.

    Responsibilities:
    - load the selected patient cohort CSV
    - copy selected JSON patient files into a flat temporary folder
    - expose items and tie_sheet in the format expected by comparators
    - expose prompt/data folder paths needed by the comparator

    The interface is identical to HMISPairwiseDataLoader so that
    downstream comparators can consume either loader interchangeably.
    """

    def __init__(self, config: MIMICPairwiseConfig):
        self.config = config
        self._selected_df: pd.DataFrame | None = None
        self._items: list[str] | None = None
        self._tie_sheet: list[tuple[str, str]] | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "MIMICPairwiseDataLoader":
        return cls(MIMICPairwiseConfig.from_yaml(config_path))

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

            # MIMIC JSONs are flat: stay_{stay_id}.json
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

    # ── items and tie sheet ───────────────────────────────────────

    def get_items(self) -> list[str]:
        if self._items is None:
            self._items = self.selected_df["uid"].astype(str).tolist()
        return self._items

    def build_tie_sheet(self) -> list[tuple[str, str]]:
        items = self.get_items()

        if self.config.tie_sheet_mode == "full_ordered_matrix":
            tie_sheet = [(a, b) for a in items for b in items if a != b]
        elif self.config.tie_sheet_mode == "unordered_pairs":
            tie_sheet = list(combinations(items, 2))
        else:
            raise ValueError(f"Unknown tie_sheet_mode: {self.config.tie_sheet_mode}")

        self._tie_sheet = tie_sheet
        return tie_sheet

    @property
    def tie_sheet(self) -> list[tuple[str, str]]:
        if self._tie_sheet is None:
            return self.build_tie_sheet()
        return self._tie_sheet

    # ── orchestration ─────────────────────────────────────────────

    def prepare(self) -> None:
        self.load_selected_patients()
        self.prepare_flat_selected_jsons()
        self.build_tie_sheet()
        self.config.responses_dir.mkdir(parents=True, exist_ok=True)

    def get_comparator_kwargs(self) -> dict[str, Any]:
        return {
            "items": self.get_items(),
            "data_folder": self.config.flat_selected_dir,
            "prompt_path": self.config.prompt_path,
            "results_folder": self.config.responses_dir,
        }

    def get_run_metadata(self) -> dict[str, Any]:
        return {
            "dataset_name": self.config.dataset_name,
            "num_items": len(self.get_items()),
            "num_pairs": len(self.tie_sheet),
            "tie_sheet_mode": self.config.tie_sheet_mode,
            "prompt_path": str(self.config.prompt_path),
            "selected_patients_path": str(self.config.selected_patients_path),
        }