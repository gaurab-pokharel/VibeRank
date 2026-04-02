from abc import ABC, abstractmethod
from pathlib import Path
import json
import numpy as np


class Comparator(ABC):
    """
    Abstract base class for pairwise comparators.

    Responsibilities of this base class:
    - keep track of items and index mappings
    - manage the win matrix
    - load household JSON files
    - render prompts from a prompt template
    - optionally forward raw prompt/response information to an injected logger

    Assumptions:
    - each item corresponds to a JSON file named <item>.json inside data_folder
    - the prompt template contains a replacement token, by default: <insert block data>
    """

    def __init__(
        self,
        items,
        num_samples=1,
        results_folder="comparison_results",
        data_folder="data_folder",
        prompt_path=None,
        logger=None,
    ):
        self.items = items
        self.num_samples = num_samples
        self.N = len(items)

        self.results_folder = Path(results_folder)
        self.data_folder = Path(data_folder)
        self.prompt_path = Path(prompt_path) if prompt_path is not None else self.data_folder.parent / "prompt.txt"
        self.logger = logger

        if not self.data_folder.exists():
            raise ValueError(f"Data folder not found: {self.data_folder}")

        if not self.prompt_path.exists():
            raise ValueError(f"Prompt file not found: {self.prompt_path}")

        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.index_map = {}
        self.map_index = {}
        for idx, item in enumerate(items):
            self.index_map[item] = idx
            self.map_index[idx] = item

        self.win_matrix = np.zeros((self.N, self.N), dtype=int)
        self.num_comparisons = 0

    # ------------------------------------------------------------------
    # Data / prompt helpers
    # ------------------------------------------------------------------

    def get_item_path(self, item):
        """Return the JSON path for a given item."""
        return self.data_folder / f"{item}.json"

    def load_item_data(self, item):
        """Load a single household JSON record."""
        item_path = self.get_item_path(item)
        if not item_path.exists():
            raise FileNotFoundError(f"Household JSON not found: {item_path}")

        with open(item_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_prompt_template(self):
        """Load the base prompt template from disk."""
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def build_household_block(
        self,
        item_i,
        item_j,
        left_label="Household 1",
        right_label="Household 2",
    ):
        """
        Build the JSON block inserted into the prompt.
        """
        households = {
            left_label: self.load_item_data(item_i),
            right_label: self.load_item_data(item_j),
        }
        return json.dumps(households, indent=4, ensure_ascii=False)

    def get_prompt(
        self,
        item_i,
        item_j,
        replacement_token="<insert block data>",
        left_label="Household 1",
        right_label="Household 2",
    ):
        """
        Render the prompt for comparing two items.
        """
        template = self.load_prompt_template()
        household_block = self.build_household_block(
            item_i=item_i,
            item_j=item_j,
            left_label=left_label,
            right_label=right_label,
        )
        return template.replace(replacement_token, household_block)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def register_pair_view(self, item_i, item_j, order, left_item, right_item, prompt):
        """
        Register one ordered prompt view with the logger, if a logger exists.

        Intended use:
        - call once for forward
        - call once for backward
        The logger itself can decide whether to deduplicate.
        """
        if self.logger is None:
            return

        self.logger.register_pair_view(
            item_a=str(item_i),
            item_b=str(item_j),
            order=order,
            left_item=str(left_item),
            right_item=str(right_item),
            prompt=prompt,
        )

    def log_raw_response(
        self,
        item_i,
        item_j,
        order,
        repeat_index,
        raw_response,
        seed=None,
        left_item=None,
        right_item=None,
        latency_ms=None,
        error=None,
        extra=None,
    ):
        """
        Log one raw LLM response, if a logger exists.
        """
        if self.logger is None:
            return

        self.logger.log_response(
            item_a=str(item_i),
            item_b=str(item_j),
            order=order,
            repeat_index=repeat_index,
            seed=seed,
            raw_response=raw_response,
            left_item=None if left_item is None else str(left_item),
            right_item=None if right_item is None else str(right_item),
            latency_ms=latency_ms,
            error=error,
            extra=extra,
        )

    def flush_logs(self):
        """Flush logger buffer, if a logger exists."""
        if self.logger is not None and hasattr(self.logger, "flush"):
            self.logger.flush()

    def close_logger(self):
        """Close logger, if a logger exists."""
        if self.logger is not None and hasattr(self.logger, "close"):
            self.logger.close()

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def reset_win_matrix(self):
        """Reset the win matrix to all zeros."""
        self.win_matrix = np.zeros((self.N, self.N), dtype=int)
        self.num_comparisons = 0

    def store_win_matrix(self, filename="win_matrix.json"):
        """
        Optional helper to write the current win matrix to disk.
        This is separate from raw-response logging.
        """
        outpath = self.results_folder / filename
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(self.win_matrix.tolist(), f, indent=2)

    def get_item(self, index):
        """Return the item corresponding to an index."""
        return self.map_index[index]

    def get_index(self, item):
        """Return the index corresponding to an item."""
        return self.index_map[item]

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    def compare(self, item_i, item_j):
        """
        Perform a pairwise comparison between two items.

        Concrete subclasses should decide how many repeats to run,
        how to call the LLM, how to parse results, and how to update
        the win matrix if needed.
        """
        pass

    @abstractmethod
    def compare_items(self, tie_sheet):
        """
        Run comparisons over a tie sheet, where tie_sheet is typically
        a list of item pairs or index pairs.
        """
        pass

    @abstractmethod
    def reset_comparator(self):
        """
        Reset any comparator-specific state.
        """
        pass

    @abstractmethod
    def call_llm(self, prompt):
        """
        Execute the LLM call for a rendered prompt and return the raw response.
        """
        pass