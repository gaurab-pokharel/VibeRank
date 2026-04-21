from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors

BAND_ORDER = ["low", "medium", "high"]
POSITION_ORDER = ["lower", "middle", "upper"]

BAND_COLORS = {
    "low": "#77f252",      
    "medium": "#f2d552",  
    "high": "#f25252",     
}

NEUTRAL_COLOR = "#ffffff"
BAND_ORDER = ["low", "medium", "high"]
POSITION_ORDER = ["lower", "middle", "upper"]

POSITION_ABBR = {
    "lower": "L",
    "middle": "M",
    "upper": "U",
}

BAND_ABBR = {
    "low": "L",
    "medium": "M",
    "high": "H",
}


def load_selected_households(selected_csv):
    df = pd.read_csv(selected_csv).copy()

    df["Client Uid"] = pd.to_numeric(df["Client Uid"], errors="coerce").astype("Int64")
    df["GRAND TOTAL"] = pd.to_numeric(df["GRAND TOTAL"], errors="coerce")
    df["priority_band"] = df["priority_band"].astype(str).str.lower()
    df["within_band_position"] = df["within_band_position"].astype(str).str.lower()

    df = df.dropna(subset=["Client Uid", "priority_band", "within_band_position"]).copy()
    df["uid"] = df["Client Uid"].astype(int).astype(str)

    df["priority_band"] = pd.Categorical(df["priority_band"], categories=BAND_ORDER, ordered=True)
    df["within_band_position"] = pd.Categorical(
        df["within_band_position"], categories=POSITION_ORDER, ordered=True
    )

    df = df.sort_values(
        by=["priority_band", "within_band_position", "GRAND TOTAL", "uid"]
    ).reset_index(drop=True)

    # Public-safe labels
    df["household_num"] = [f"H{i}" for i in range(1, len(df) + 1)]
    df["band_pos_label"] = [
        f"{BAND_ABBR[str(band)]}{POSITION_ABBR[str(pos)]}"
        for band, pos in zip(df["priority_band"], df["within_band_position"])
    ]
    df["display_label"] = [
        f"{bp} \n ({hn})"
        for bp, hn in zip(df["band_pos_label"], df["household_num"])
    ]

    return df


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_dummy_winner(raw_response):
    """
    Extract winner uid from dummy raw response, e.g.
    'I would prioritize Household 1 (uid=12345) ...'
    """
    if raw_response is None:
        return None

    m = re.search(r"uid=(\d+)", raw_response)
    if m:
        return m.group(1)
    return None

def parse_winner_from_transitional_housing(value):
    """
    Convert transitional_housing_household into winner label "1" or "2".

    Expected inputs:
        "Household 1"
        "Household 2"

    Returns:
        "1" or "2"

    Raises:
        ValueError for invalid values.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "Household 1":
        return "1"
    elif value == "Household 2":
        return "2"

    raise ValueError(f"Could not parse winner from transitional_housing_household={value!r}")


def build_trial_df_from_csv(csv_path, winner_parser=parse_winner_from_transitional_housing):
    df = pd.read_csv(csv_path).copy()

    if df.empty:
        raise ValueError("No rows found in CSV.")

    required_cols = ["transitional_housing_household", "left_item", "right_item"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # winner_parser should return "1" or "2"
    df["winner_side"] = df["transitional_housing_household"].apply(winner_parser)

    df = df.dropna(subset=["left_item", "right_item", "winner_side"]).copy()

    df["left_item"] = df["left_item"].astype(str)
    df["right_item"] = df["right_item"].astype(str)
    df["winner_side"] = df["winner_side"].astype(str)

    # Convert winner side into actual winner UUID
    df["winner_item"] = np.where(
        df["winner_side"] == "1",
        df["left_item"],
        np.where(
            df["winner_side"] == "2",
            df["right_item"],
            np.nan
        )
    )

    df = df.dropna(subset=["winner_item"]).copy()
    df["winner_item"] = df["winner_item"].astype(str)

    df = df[["left_item", "right_item", "winner_item"]].copy()

    print(df.head())
    return df

def _parse_winner(text):
    """
    Returns:
        "1" or "2"

    Looks through the first few non-empty cleaned lines and extracts
    whoever gets Transitional Housing.
    """
    if not text or not str(text).strip():
        raise ValueError("Empty response text.")

    import re

    raw_lines = str(text).splitlines()

    cleaned_lines = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # remove common special/control tokens
        line = re.sub(r"<\|.*?\|>", "", line)
        line = re.sub(r"</?think>", "", line, flags=re.IGNORECASE)
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    if not cleaned_lines:
        raise ValueError("No usable lines found in response text.")

    # Only inspect the first few meaningful lines
    candidate_lines = cleaned_lines[:5]

    patterns = [
        r"Transitional Housing:\s*Household\s*([12])\b",
        r"Transitional Housing:\s*([12])\b",
    ]

    for line in candidate_lines:
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                return match.group(1)

    raise ValueError(
        f"Could not parse winner from first meaningful lines: {candidate_lines!r}"
    )


def build_trial_df_from_jsonl(log_path, winner_parser=_parse_winner):
    records = load_jsonl(log_path)

    response_records = [r for r in records if r.get("event") == "response"]
    df = pd.DataFrame(response_records).copy()

    if df.empty:
        raise ValueError("No response records found in log.")

    # Parse winner side: "1" means left won, "2" means right won
    df["winner_item"] = df["raw_response"].apply(winner_parser)

    # keep only rows where parser succeeded
    df = df.dropna(subset=["winner_item", "left_item", "right_item"]).copy()

    df["left_item"] = df["left_item"].astype(str)
    df["right_item"] = df["right_item"].astype(str)
    df["winner_item"] = df["winner_item"].astype(str)

    # Convert winner side into actual winner UID
    df["winner_item"] = np.where(
        df["winner_item"] == "1",
        df["left_item"],
        np.where(
            df["winner_item"] == "2",
            df["right_item"],
            np.nan
        )
    )

    df = df.dropna(subset=["winner_item"]).copy()
    df = df[["left_item", "right_item", "winner_item"]].copy()

    print(df.head())
    return df




def bernoulli_certainty(p):
    """
    Certainty scaled from Bernoulli variance.
    p = 0.5 -> 0 certainty
    p = 0 or 1 -> 1 certainty
    """
    if pd.isna(p):
        return np.nan
    return 1.0 - 4.0 * p * (1.0 - p)

def parse_more_vulnerable_household(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    if s == "Household 1":
        return "1"
    elif s == "Household 2":
        return "2"
    else:
        return np.nan


def build_trial_df_from_vulnerablecsv(csv_path, winner_parser=parse_more_vulnerable_household):
    df = pd.read_csv(csv_path).copy()

    if df.empty:
        raise ValueError("No rows found in CSV.")

    required_cols = ["more_vulnerable_household", "left_item", "right_item"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # winner_parser should return "1" or "2"
    df["winner_side"] = df["more_vulnerable_household"].apply(winner_parser)

    df = df.dropna(subset=["left_item", "right_item", "winner_side"]).copy()

    df["left_item"] = df["left_item"].astype(str)
    df["right_item"] = df["right_item"].astype(str)
    df["winner_side"] = df["winner_side"].astype(str)

    # Convert winner side into actual winner UUID
    df["winner_item"] = np.where(
        df["winner_side"] == "1",
        df["left_item"],
        np.where(
            df["winner_side"] == "2",
            df["right_item"],
            np.nan
        )
    )

    df = df.dropna(subset=["winner_item"]).copy()
    df["winner_item"] = df["winner_item"].astype(str)

    df = df[["left_item", "right_item", "winner_item"]].copy()

    print(df.head())
    return df

def build_directional_probability_matrix(selected_df, trials_df):
    """
    Directed version.

    prob_matrix[i, j] =
        P(row household beats column household |
          row shown first, column shown second)

    n_matrix[i, j] =
        number of repeats observed for that directed prompt
    """
    items = selected_df["uid"].tolist()
    n = len(items)

    prob_matrix = np.full((n, n), np.nan, dtype=float)
    n_matrix = np.zeros((n, n), dtype=int)

    for i, row_uid in enumerate(items):
        for j, col_uid in enumerate(items):
            #print('gets in innter loop')
            if i == j:
                continue

            directed_trials = trials_df[
                (trials_df["left_item"] == row_uid) &
                (trials_df["right_item"] == col_uid)
            ].copy()

            if len(directed_trials) == 0:
                continue
            
            if row_uid == '588' and col_uid=='51360':
                print('row first')
                print(directed_trials)
            
            if col_uid == '588' and row_uid=='51360':
                print('col fiorst')
                print(directed_trials)

            p_row_beats_col = (directed_trials["winner_item"] == row_uid).mean()
    
            #p_row_beats_col = len(directed_trials)
            prob_matrix[i, j] = p_row_beats_col
            n_matrix[i, j] = len(directed_trials)

    return prob_matrix, n_matrix

def blend_with_white(hex_color, strength, min_strength=0.15):
    """
    strength in [0, 1]
    0 -> almost white
    1 -> full base color
    """
    strength = float(np.clip(strength, 0, 1))
    strength = min_strength + (1 - min_strength) * strength

    base = np.array(mcolors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])

    mixed = white * (1 - strength) + base * strength
    return mixed


def make_color_matrix(selected_df, prob_matrix):
    bands = selected_df["priority_band"].astype(str).tolist()
    n = len(bands)

    color_matrix = np.ones((n, n, 3), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                color_matrix[i, j, :] = blend_with_white(BAND_COLORS[bands[i]], 1.0)
                continue

            p = prob_matrix[i, j]
            if np.isnan(p):
                color_matrix[i, j, :] = mcolors.to_rgb("#FFFFFF")
                continue

            certainty = bernoulli_certainty(p)

            if p > 0.5:
                favored_band = bands[i]
            elif p < 0.5:
                favored_band = bands[j]
            else:
                favored_band = None

            if favored_band is None:
                color_matrix[i, j, :] = mcolors.to_rgb(NEUTRAL_COLOR)
            else:
                color_matrix[i, j, :] = blend_with_white(BAND_COLORS[favored_band], certainty)

    return color_matrix




from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_pairwise_stability_heatmap(
    selected_df,
    prob_matrix,
    n_matrix,
    title=None,
    figsize=(10, 8),
    save_path=None,
    annotate=True,
    include_uid=True,
):
    color_matrix = make_color_matrix(selected_df, prob_matrix)

    if include_uid:
        labels = [
            f"{label}\nUID: {uid}"
            for label, uid in zip(
                selected_df["display_label"].astype(str),
                selected_df["Client Uid"].astype(str),
            )
        ]
    else:
        labels = selected_df["display_label"].astype(str).tolist()

    n = len(labels)

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n):
        for j in range(n):
            rect = Rectangle(
                (j, i), 1, 1,
                facecolor=color_matrix[i, j],
                edgecolor="white",
                linewidth=1.5
            )
            ax.add_patch(rect)

            if annotate:
                if i == j:
                    text = "—"
                elif np.isnan(prob_matrix[i, j]):
                    text = ""
                else:
                    text = f"{prob_matrix[i, j]:.2f}\n(n={n_matrix[i, j]})"

                ax.text(
                    j + 0.5, i + 0.5, text,
                    ha="center", va="center",
                    fontsize=9
                )

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)

    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_yticklabels(labels, rotation=0)

    ax.set_xticks(np.arange(n + 1), minor=True)
    ax.set_yticks(np.arange(n + 1), minor=True)
    ax.grid(which="minor", linestyle="--", alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Strong band boundary lines between low / medium / high
    boundary_positions = [3, 6]
    for pos in boundary_positions:
        ax.axhline(pos, color="black", linewidth=3)
        ax.axvline(pos, color="black", linewidth=3)

    # Outer border
    ax.axhline(0, color="black", linewidth=2)
    ax.axhline(n, color="black", linewidth=2)
    ax.axvline(0, color="black", linewidth=2)
    ax.axvline(n, color="black", linewidth=2)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("Household shown second")
    ax.set_ylabel("Household shown first")

    plt.subplots_adjust(bottom=0.22, left=0.22)

    fig.text(
        0.5, 0.08,
        "Cell value = P(row household beats column household | row shown first, column shown second).\n"
        "Hue = band of the favored household; intensity = higher certainty / lower variance.",
        ha="center",
        va="center",
        fontsize=9
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def make_heatmap_for_run(
    *,
    selected_csv,
    log_path,
    title=None,
    save_path=None,
    winner_parser=_parse_winner,
):
    selected_df = load_selected_households(selected_csv)
    #print(selected_df)
    #trials_df = build_trial_df_from_csv(log_path, winner_parser=parse_winner_from_transitional_housing)#build_trial_df_from_jsonl(log_path, winner_parser=winner_parser)
    #trials_df = build_trial_df_from_jsonl(log_path, winner_parser=winner_parser)
    trials_df = build_trial_df_from_vulnerablecsv(log_path)#build_trial_df_from_jsonl(log_path, winner_parser=winner_parser)
    prob_matrix, n_matrix = build_directional_probability_matrix(
        selected_df, trials_df
    )

    plot_pairwise_stability_heatmap(
        selected_df=selected_df,
        prob_matrix=prob_matrix,
        n_matrix=n_matrix,
        title=title,
        save_path=save_path,
    )

    return selected_df, trials_df, prob_matrix, n_matrix