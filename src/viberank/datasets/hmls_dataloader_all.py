from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
import random
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

    # "full_ordered_matrix": all ordered pairs (a, b), (b, a)
    # "unordered_pairs": only nC2 pairs
    tie_sheet_mode: str = "unordered_pairs"

    # "all" -> load all raw household JSON files
    # "selected" -> load selected_households.csv only
    household_source: str = "all"

    # Keep only this fraction of the candidate tie sheet
    tie_sheet_fraction: float = 0.40
    tie_sheet_seed: int = 42

    # Progress logging
    verbose: bool = False
    progress_every: int = 500

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
            selected_households_filename=data.get(
                "selected_households_filename",
                "selected_households.csv",
            ),
            responses_dirname=data.get("responses_dirname", "responses"),

            tie_sheet_mode=data.get("tie_sheet_mode", "unordered_pairs"),
            household_source=data.get("household_source", "all"),

            tie_sheet_fraction=data.get("tie_sheet_fraction", 0.40),
            tie_sheet_seed=data.get("tie_sheet_seed", 42),

            verbose=data.get("verbose", False),
            progress_every=data.get("progress_every", 500),

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
    


class HMISPairwiseDataLoader:
    """
    No-copy dataloader.

    Instead of copying JSONs into a flat folder, this exposes:

        item_json_paths: dict[str, Path]

    where:

        item_json_paths[uid] = original JSON file path
    """

    def __init__(self, config: HMISPairwiseConfig):
        self.config = config
        self._selected_df: pd.DataFrame | None = None
        self._items: list[str] | None = None
        self._item_json_paths: dict[str, Path] | None = None
        self._tie_sheet: list[tuple[str, str]] | None = None
        self._full_num_pairs_before_sampling: int | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "HMISPairwiseDataLoader":
        return cls(HMISPairwiseConfig.from_yaml(config_path))

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[HMISPairwiseDataLoader] {msg}", flush=True)

    def _reset_cached_items_paths_and_tie_sheet(self) -> None:
        self._items = None
        self._item_json_paths = None
        self._tie_sheet = None
        self._full_num_pairs_before_sampling = None

    def load_selected_households(self) -> pd.DataFrame:
        self._log(f"Loading selected households from: {self.config.selected_households_path}")

        df = pd.read_csv(self.config.selected_households_path).copy()

        df["Client Uid"] = pd.to_numeric(
            df["Client Uid"],
            errors="coerce",
        ).astype("Int64")

        if "GRAND TOTAL" in df.columns:
            df["GRAND TOTAL"] = pd.to_numeric(
                df["GRAND TOTAL"],
                errors="coerce",
            )

        if "priority_band" not in df.columns:
            raise ValueError(
                "selected_households.csv must contain a 'priority_band' column."
            )

        df["priority_band"] = df["priority_band"].astype(str).str.lower()

        filt = self.config.selected_filters or {}

        before = len(df)
        if filt.get("drop_missing_uid", True):
            df = df.dropna(subset=["Client Uid"])
        self._log(f"After drop_missing_uid: {len(df):,} kept, {before - len(df):,} dropped")

        before = len(df)
        if filt.get("drop_missing_band", True):
            df = df.dropna(subset=["priority_band"])
        self._log(f"After drop_missing_band: {len(df):,} kept, {before - len(df):,} dropped")

        df = df.copy()
        df["uid"] = df["Client Uid"].astype(int).astype(str)

        # Build original JSON path directly.
        df["json_path"] = df.apply(
            lambda row: self.config.raw_dataset_dir
            / str(row["priority_band"])
            / f"{row['uid']}.json",
            axis=1,
        )

        sort_cols = [
            c
            for c in ["priority_band", "within_band_position", "GRAND TOTAL", "uid"]
            if c in df.columns
        ]

        if sort_cols:
            self._log(f"Sorting selected households by: {sort_cols}")
            df = df.sort_values(by=sort_cols).reset_index(drop=True)

        self._validate_json_paths(df)

        self._selected_df = df
        self._reset_cached_items_paths_and_tie_sheet()

        self._log(f"Finished loading selected households: {len(df):,}")

        return df

    def load_all_households(self) -> pd.DataFrame:
        self._log("Loading ALL households from raw JSON folders")

        rows = []
        raw_dataset_dir = self.config.raw_dataset_dir

        self._log(f"raw_dataset_dir: {raw_dataset_dir}")

        if not raw_dataset_dir.exists():
            raise FileNotFoundError(f"raw_dataset_dir does not exist: {raw_dataset_dir}")

        band_dirs = [
            p
            for p in raw_dataset_dir.iterdir()
            if p.is_dir()
            and p.name != self.config.responses_dirname
            and not p.name.startswith(".")
            and not p.name.startswith("_")
        ]

        self._log(f"Found {len(band_dirs):,} candidate priority-band folders")

        for band_idx, band_dir in enumerate(band_dirs, start=1):
            priority_band = band_dir.name.lower()
            json_paths = list(band_dir.glob("*.json"))

            self._log(
                f"[{band_idx:,}/{len(band_dirs):,}] "
                f"Scanning band='{priority_band}' with {len(json_paths):,} JSON files"
            )

            for json_idx, json_path in enumerate(json_paths, start=1):
                uid = json_path.stem

                rows.append(
                    {
                        "uid": str(uid),
                        "Client Uid": str(uid),
                        "priority_band": priority_band,
                        "json_path": json_path,
                    }
                )

                if self.config.verbose and json_idx % self.config.progress_every == 0:
                    self._log(
                        f"  scanned {json_idx:,}/{len(json_paths):,} JSONs "
                        f"in band='{priority_band}'"
                    )

        df = pd.DataFrame(rows)

        if df.empty:
            raise ValueError(f"No household JSON files found under {raw_dataset_dir}")

        self._log(f"Total raw household JSONs found: {len(df):,}")

        duplicate_uids = df[df["uid"].duplicated()]["uid"].unique().tolist()
        if duplicate_uids:
            raise ValueError(
                "Duplicate household uid values found across raw folders. "
                f"Examples: {duplicate_uids[:10]}"
            )

        df = df.sort_values(by=["priority_band", "uid"]).reset_index(drop=True)

        self._validate_json_paths(df)

        self._selected_df = df
        self._reset_cached_items_paths_and_tie_sheet()

        self._log(f"Finished loading all households: {len(df):,}")

        return df

    def _validate_json_paths(self, df: pd.DataFrame) -> None:
        if "json_path" not in df.columns:
            raise ValueError("Internal error: df must contain a 'json_path' column.")

        missing = []

        total = len(df)

        for idx, path in enumerate(df["json_path"], start=1):
            path = Path(path)

            if not path.exists():
                missing.append(str(path))

            if self.config.verbose and (
                idx == 1
                or idx == total
                or idx % self.config.progress_every == 0
            ):
                self._log(f"  validated {idx:,}/{total:,} JSON paths")

        if missing:
            raise FileNotFoundError(
                f"{len(missing):,} household JSON files were not found. "
                "First few missing paths:\n"
                + "\n".join(missing[:20])
            )

        self._log(f"Validated {total:,} JSON paths")

    @property
    def selected_df(self) -> pd.DataFrame:
        if self._selected_df is None:
            if self.config.household_source == "all":
                return self.load_all_households()

            if self.config.household_source == "selected":
                return self.load_selected_households()

            raise ValueError(
                "household_source must be either 'all' or 'selected', "
                f"got {self.config.household_source}"
            )

        return self._selected_df

    def get_items(self) -> list[str]:
        if self._items is None:
            self._log("Building item list")
            self._items = self.selected_df["uid"].astype(str).tolist()
            self._log(f"Built item list with {len(self._items):,} households")

        return self._items

    def get_item_json_paths(self) -> dict[str, Path]:
        if self._item_json_paths is None:
            self._log("Building uid -> json_path mapping")

            df = self.selected_df

            self._item_json_paths = {
                str(row["uid"]): Path(row["json_path"])
                for _, row in df.iterrows()
            }

            self._log(
                f"Built item_json_paths mapping for "
                f"{len(self._item_json_paths):,} households"
            )

        return self._item_json_paths

    def build_tie_sheet(self) -> list[tuple[str, str]]:
        items = self.get_items()
        n = len(items)

        self._log("Building tie sheet")
        self._log(f"num_items: {n:,}")
        self._log(f"tie_sheet_mode: {self.config.tie_sheet_mode}")
        self._log(f"tie_sheet_fraction: {self.config.tie_sheet_fraction}")
        self._log(f"tie_sheet_seed: {self.config.tie_sheet_seed}")

        if self.config.tie_sheet_mode == "full_ordered_matrix":
            full_tie_sheet = [(a, b) for a in items for b in items if a != b]

        elif self.config.tie_sheet_mode == "unordered_pairs":
            full_tie_sheet = list(combinations(items, 2))

        else:
            raise ValueError(f"Unknown tie_sheet_mode: {self.config.tie_sheet_mode}")

        fraction = self.config.tie_sheet_fraction

        if not (0 < fraction <= 1):
            raise ValueError(f"tie_sheet_fraction must be in (0, 1], got {fraction}")

        if len(full_tie_sheet) == 0:
            raise ValueError("Tie sheet is empty. Need at least 2 households.")

        self._full_num_pairs_before_sampling = len(full_tie_sheet)

        num_keep = max(1, int(round(len(full_tie_sheet) * fraction)))

        self._log(f"Candidate pairs before sampling: {len(full_tie_sheet):,}")
        self._log(f"Pairs to keep after sampling: {num_keep:,}")

        rng = random.Random(self.config.tie_sheet_seed)

        tie_sheet = full_tie_sheet.copy()
        rng.shuffle(tie_sheet)

        tie_sheet = tie_sheet[:num_keep]

        # Keeps the sampled set stable/readable.
        # Remove if you want randomized run order.
        tie_sheet = sorted(tie_sheet)

        self._tie_sheet = tie_sheet

        self._log(f"Finished tie sheet: {len(tie_sheet):,} pairs")

        if self.config.verbose:
            preview_n = min(10, len(tie_sheet))
            self._log(f"First {preview_n} pairs: {tie_sheet[:preview_n]}")

        return tie_sheet

    @property
    def tie_sheet(self) -> list[tuple[str, str]]:
        if self._tie_sheet is None:
            return self.build_tie_sheet()

        return self._tie_sheet
    @property
    def pairs(self) -> list[tuple[str, str]]:
        """
        Backward-compatible alias for experiment runners that expect dataloader.pairs.
        In this dataloader, pairs == tie_sheet.
        """
        return self.tie_sheet


    def build_pairs(self) -> list[tuple[str, str]]:
        """
        Backward-compatible alias for older runners/loaders that call build_pairs().
        """
        return self.build_tie_sheet()

    def prepare(self) -> None:
        self._log("=" * 80)
        self._log("Starting dataloader.prepare()")
        self._log(f"dataset_name: {self.config.dataset_name}")
        self._log(f"household_source: {self.config.household_source}")
        self._log(f"raw_root: {self.config.raw_root}")
        self._log(f"processed_root: {self.config.processed_root}")
        self._log(f"responses_dir: {self.config.responses_dir}")
        self._log("=" * 80)

        if self.config.household_source == "all":
            self.load_all_households()

        elif self.config.household_source == "selected":
            self.load_selected_households()

        else:
            raise ValueError(
                "household_source must be either 'all' or 'selected', "
                f"got {self.config.household_source}"
            )

        # No copying. Just build the path mapping and tie sheet.
        self.get_item_json_paths()
        self.build_tie_sheet()

        self.config.responses_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"Ensured responses_dir exists: {self.config.responses_dir}")
        self._log("Finished dataloader.prepare()")
        self._log("=" * 80)

    def get_comparator_kwargs(self) -> dict[str, Any]:
        return {
            "items": self.get_items(),
            "tie_sheet": self.tie_sheet,

            # New no-copy interface:
            # comparator should open item_json_paths[uid]
            "item_json_paths": self.get_item_json_paths(),

            "prompt_path": self.config.prompt_path,
            "results_folder": self.config.responses_dir,
        }

    def get_run_metadata(self) -> dict[str, Any]:
        items = self.get_items()
        tie_sheet = self.tie_sheet
        item_json_paths = self.get_item_json_paths()

        return {
            "dataset_name": self.config.dataset_name,
            "household_source": self.config.household_source,

            "num_items": len(items),
            "num_item_json_paths": len(item_json_paths),

            "num_pairs_before_sampling": self._full_num_pairs_before_sampling,
            "num_pairs": len(tie_sheet),

            "tie_sheet_mode": self.config.tie_sheet_mode,
            "tie_sheet_fraction": self.config.tie_sheet_fraction,
            "tie_sheet_seed": self.config.tie_sheet_seed,

            "verbose": self.config.verbose,
            "progress_every": self.config.progress_every,

            "prompt_path": str(self.config.prompt_path),
            "selected_households_path": str(self.config.selected_households_path),
            "responses_dir": str(self.config.responses_dir),

            "copying_enabled": False,
            "data_interface": "item_json_paths",
        }


if __name__ == "__main__":
    import sys

    DEFAULT_CONFIG_PATH = "../configs/datasets/rc_vispdat.yaml"

    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path(DEFAULT_CONFIG_PATH)

    print("=" * 80)
    print("Loading dataloader config")
    print("=" * 80)
    print("config_path:", config_path.resolve())

    dataloader = HMISPairwiseDataLoader.from_yaml(config_path)

    dataloader.config.verbose = True
    dataloader.config.progress_every = 500

    dataloader.prepare()

    metadata = dataloader.get_run_metadata()
    items = dataloader.get_items()
    tie_sheet = dataloader.tie_sheet
    item_json_paths = dataloader.get_item_json_paths()

    print("\n" + "=" * 80)
    print("Run metadata")
    print("=" * 80)

    for k, v in metadata.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 80)
    print("Households")
    print("=" * 80)
    print("num_items:", len(items))
    print("first 20 items:")
    print(items[:20])

    print("\n" + "=" * 80)
    print("JSON path mapping preview")
    print("=" * 80)

    for uid in items[:10]:
        print(uid, "->", item_json_paths[uid])

    print("\n" + "=" * 80)
    print("Tie sheet preview")
    print("=" * 80)
    print("num_pairs:", len(tie_sheet))

    print("\nfirst 20 pairs:")
    for pair in tie_sheet[:20]:
        print(pair)

    print("\nlast 20 pairs:")
    for pair in tie_sheet[-20:]:
        print(pair)

    tie_sheet_df = pd.DataFrame(
        tie_sheet,
        columns=["household_1", "household_2"],
    )

    num_self_pairs = int(
        (tie_sheet_df["household_1"] == tie_sheet_df["household_2"]).sum()
    )
    num_duplicate_ordered_pairs = int(tie_sheet_df.duplicated().sum())

    unordered_keys = tie_sheet_df.apply(
        lambda row: tuple(sorted([row["household_1"], row["household_2"]])),
        axis=1,
    )
    num_duplicate_unordered_pairs = int(unordered_keys.duplicated().sum())

    print("\n" + "=" * 80)
    print("Tie sheet checks")
    print("=" * 80)
    print("num_self_pairs:", num_self_pairs)
    print("num_duplicate_ordered_pairs:", num_duplicate_ordered_pairs)
    print("num_duplicate_unordered_pairs:", num_duplicate_unordered_pairs)

    out_path = dataloader.config.responses_dir / "tie_sheet_preview.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tie_sheet_df.to_csv(out_path, index=False)

    path_preview_df = pd.DataFrame(
        [
            {"uid": uid, "json_path": str(path)}
            for uid, path in item_json_paths.items()
        ]
    )

    path_out_path = dataloader.config.responses_dir / "item_json_paths_preview.csv"
    path_preview_df.to_csv(path_out_path, index=False)

    print("\n" + "=" * 80)
    print("Saved")
    print("=" * 80)
    print("tie_sheet csv:", out_path)
    print("item_json_paths csv:", path_out_path)