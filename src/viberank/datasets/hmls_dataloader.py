from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
import shutil
from typing import Any
import pandas as pd
import yaml


@dataclass
class HMISPairwiseConfig:
    dataset_name: str
    raw_root: Path
    processed_root: Path
    prompt_filename: str = "prompt.txt"
    selected_households_filename: str = "selected_households.csv"
    responses_dirname: str = "responses"
    flat_selected_dirname: str = "_selected_jsons"
    tie_sheet_mode: str = "full_ordered_matrix"
    clear_flat_selected_dir: bool = True
    selected_filters: dict[str, Any] | None = None
    run_settings: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "HMISPairwiseConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            dataset_name=data["dataset_name"],
            raw_root=Path(data["raw_root"]),
            processed_root=Path(data["processed_root"]),
            prompt_filename=data.get("prompt_filename", "prompt.txt"),
            selected_households_filename=data.get("selected_households_filename", "selected_households.csv"),
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
    def selected_households_path(self) -> Path:
        return self.processed_dataset_dir / self.selected_households_filename

    @property
    def responses_dir(self) -> Path:
        return self.raw_dataset_dir / self.responses_dirname

    @property
    def flat_selected_dir(self) -> Path:
        return self.responses_dir / self.flat_selected_dirname


class HMISPairwiseDataLoader:
    """
    Data loader for the homelessness pairwise-comparison setup.

    Responsibilities:
    - load selected_households.csv
    - copy selected JSON household files into a flat temporary folder
    - expose items and tie_sheet in the format expected by comparators
    - expose prompt/data folder paths needed by the comparator
    """

    def __init__(self, config: HMISPairwiseConfig):
        self.config = config
        self._selected_df: pd.DataFrame | None = None
        self._items: list[str] | None = None
        self._tie_sheet: list[tuple[str, str]] | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "HMISPairwiseDataLoader":
        return cls(HMISPairwiseConfig.from_yaml(config_path))

    def load_selected_households(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.selected_households_path).copy()

        df["Client Uid"] = pd.to_numeric(df["Client Uid"], errors="coerce").astype("Int64")
        if "GRAND TOTAL" in df.columns:
            df["GRAND TOTAL"] = pd.to_numeric(df["GRAND TOTAL"], errors="coerce")

        if "priority_band" not in df.columns:
            raise ValueError("selected_households.csv must contain a 'priority_band' column.")

        df["priority_band"] = df["priority_band"].astype(str).str.lower()

        filt = self.config.selected_filters or {}
        if filt.get("drop_missing_uid", True):
            df = df.dropna(subset=["Client Uid"])
        if filt.get("drop_missing_band", True):
            df = df.dropna(subset=["priority_band"])

        df = df.copy()
        df["uid"] = df["Client Uid"].astype(int).astype(str)

        sort_cols = [c for c in ["priority_band", "within_band_position", "GRAND TOTAL", "uid"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols).reset_index(drop=True)

        self._selected_df = df
        return df

    @property
    def selected_df(self) -> pd.DataFrame:
        if self._selected_df is None:
            return self.load_selected_households()
        return self._selected_df

    def prepare_flat_selected_jsons(self) -> Path:
        flat_dir = self.config.flat_selected_dir
        flat_dir.mkdir(parents=True, exist_ok=True)

        if self.config.clear_flat_selected_dir:
            for p in flat_dir.glob("*.json"):
                p.unlink()

        missing = []

        for _, row in self.selected_df.iterrows():
            uid = row["uid"]
            band = row["priority_band"]

            src = self.config.raw_dataset_dir / band / f"{uid}.json"
            dst = flat_dir / f"{uid}.json"

            if not src.exists():
                missing.append(str(src))
                continue

            shutil.copy2(src, dst)

        if missing:
            raise FileNotFoundError(
                "Some selected JSON household files were not found:\n" + "\n".join(missing)
            )

        return flat_dir

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

    def prepare(self) -> None:
        self.load_selected_households()
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
            "selected_households_path": str(self.config.selected_households_path),
        }