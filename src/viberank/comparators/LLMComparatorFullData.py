import time
import json
from pathlib import Path
from typing import Iterable

from viberank.comparators.base import Comparator


class LLMComparator(Comparator):
    """
    LLM-backed comparator with local test mode.

    Real mode:
    - Uses vLLM.

    Local test mode:
    - Does NOT import vLLM.
    - Does NOT initialize any model.
    - Returns deterministic fake responses.
    - Useful for testing dataloader -> comparator plumbing.
    """

    def __init__(
        self,
        items,
        tie_sheet=None,
        item_json_paths=None,
        num_samples=1,
        results_folder="comparison_results",
        data_folder=None,  # kept for backward compatibility
        prompt_path=None,
        logger=None,
        temperature=0.0,
        max_tokens=256,
        timeout=120,
        llm_name="qwen",  # qwen, llama7, deepseek8B, local_test
        rng_seed=10,
        local_test_mode=False,
        batch_size=64
    ):
        # If base Comparator dislikes None data_folder, "." is harmless because
        # no-copy mode overrides get_prompt() and uses item_json_paths.
        safe_data_folder = data_folder if data_folder is not None else "."

        super().__init__(
            items=items,
            num_samples=num_samples,
            results_folder=results_folder,
            data_folder=safe_data_folder,
            prompt_path=prompt_path,
            logger=logger,
        )

        self.tie_sheet = None
        if tie_sheet is not None:
            self.tie_sheet = [(str(a), str(b)) for a, b in tie_sheet]

        self.item_json_paths = None
        if item_json_paths is not None:
            self.item_json_paths = {
                str(uid): Path(path)
                for uid, path in item_json_paths.items()
            }

        self.rng_seed = rng_seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.batch_size = int(batch_size)
        self._prompt_template_cache = None

        self.local_test_mode = bool(local_test_mode) or llm_name in {
            "local_test",
            "mock",
            "dry_run",
        }

        self.llm_name = llm_name
        self.llm = None
        self.SamplingParams = None

        if self.local_test_mode:
            print("initialized LLMComparator in LOCAL TEST MODE: no vLLM/model loaded")
        else:
            self._setup_real_llm(llm_name)

    # ------------------------------------------------------------------
    # Real LLM setup
    # ------------------------------------------------------------------

    def _setup_real_llm(self, llm_name):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams

        if llm_name == "llama7":
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        elif llm_name == "deepseek8B":
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

        elif llm_name == "qwen":
            model_name = "Qwen/Qwen2.5-7B-Instruct"

        else:
            raise ValueError(f"Unknown llm_name: {llm_name}")

        self.llm = LLM(
            model=model_name,
            trust_remote_code=False,
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        print(f"initialized real LLM: {model_name}")

    # ------------------------------------------------------------------
    # No-copy JSON loading
    # ------------------------------------------------------------------

    def _get_item_json_path(self, item):
        item = str(item)

        if self.item_json_paths is not None:
            if item not in self.item_json_paths:
                raise KeyError(f"Missing JSON path for item={item}")

            path = Path(self.item_json_paths[item])

            if not path.exists():
                raise FileNotFoundError(
                    f"JSON file does not exist for item={item}: {path}"
                )

            return path

        # Backward-compatible fallback.
        path = Path(self.data_folder) / f"{item}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"JSON file does not exist for item={item}: {path}"
            )

        return path

    def _load_item_json(self, item):
        path = self._get_item_json_path(item)

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def load_item_data(self, item):
        item = str(item)

        # Preferred full-data mode: use exact JSON paths from dataloader.
        # This is necessary because households live inside high/medium/low folders.
        if self.item_json_paths is not None:
            if item not in self.item_json_paths:
                raise KeyError(
                    f"Missing JSON path for item={item}. "
                    f"Available item_json_paths keys sample: "
                    f"{list(self.item_json_paths.keys())[:10]}"
                )

            item_path = Path(self.item_json_paths[item])

            if not item_path.exists():
                raise FileNotFoundError(f"Household JSON not found: {item_path}")

            with open(item_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Backward-compatible fallback.
        item_path = Path(self.data_folder) / f"{item}.json"

        if not item_path.exists():
            raise FileNotFoundError(f"Household JSON not found: {item_path}")

        with open(item_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
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

    def get_prompt(self, left_item, right_item, replacement_token="<insert block data>", left_label="Household 1", right_label="Household 2"):
        

        if self._prompt_template_cache is None:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self._prompt_template_cache = f.read()
        prompt_template = self._prompt_template_cache

        household_block = self.build_household_block(
            item_i=left_item,
            item_j=right_item,
            left_label=left_label,
            right_label=right_label,
        )

        prompt = prompt_template

        return prompt.replace(replacement_token, household_block)

    # ------------------------------------------------------------------
    # LLM calling / fake local responses
    # ------------------------------------------------------------------

    def _seed_for_call(self, tie_index, repeat_index):
        tie_index = -1 if tie_index is None else int(tie_index)

        return (
            int(self.rng_seed) * 1_000_003
            + tie_index * 9_176
            + int(repeat_index) * 101
        ) % (2**31 - 1)

    def _mock_llm_response(self, tie_index=None, repeat_index=None):
        """
        Returns fake text that your existing _parse_winner() can parse.

        Alternates deterministically between left and right wins.
        """

        seed = self._seed_for_call(tie_index, repeat_index)

        if seed % 2 == 0:
            # Parser interprets this as LEFT winning.
            return (
                "Emergency Shelter Household 2. "
                "Transitional Housing: Household 1."
            )

        else:
            # Parser interprets this as RIGHT winning.
            return (
                "Emergency Shelter Household 1. "
                "Transitional Housing: Household 2."
            )

    def call_llm(self, prompt, tie_index=None, repeat_index=None):
        if self.local_test_mode:
            raw_text = self._mock_llm_response(
                tie_index=tie_index,
                repeat_index=repeat_index,
            )
            print("[LOCAL_TEST_RESPONSE]", raw_text)
            return raw_text

        if self.llm is None or self.SamplingParams is None:
            raise RuntimeError(
                "Real LLM is not initialized. Use local_test_mode=True "
                "or initialize with a valid llm_name."
            )

        seed = self._seed_for_call(tie_index, repeat_index)

        sampling_params = self.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        raw_text = outputs[0].outputs[0].text.strip()

        #print(raw_text)

        return raw_text

    # batched call to llm for speedup

    def call_llm_batch(self, prompts, batch_tasks=None):
        """
        Batched vLLM call.

        Args:
            prompts:
                List of prompt strings.
            batch_tasks:
                Optional list of task dicts. Used only for deterministic
                local_test_mode responses.

        Returns:
            List[str] raw model responses.
        """

        prompts = list(prompts)

        if len(prompts) == 0:
            return []

        if self.local_test_mode:
            raw_texts = []

            if batch_tasks is None:
                batch_tasks = [
                    {"tie_index": None, "repeat_index": i}
                    for i in range(len(prompts))
                ]

            for task in batch_tasks:
                raw_texts.append(
                    self._mock_llm_response(
                        tie_index=task.get("tie_index"),
                        repeat_index=task.get("repeat_index"),
                    )
                )

            return raw_texts

        if self.llm is None or self.SamplingParams is None:
            raise RuntimeError(
                "Real LLM is not initialized. Use local_test_mode=True "
                "or initialize with a valid llm_name."
            )

        sampling_params = self.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        raw_texts = []
        for output in outputs:
            raw_texts.append(output.outputs[0].text.strip())

        return raw_texts



    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_winner(self, text):
        text = " ".join(text.strip().split())

        if "Emergency Shelter Household 1. Transitional Housing: Household 2." in text:
            return "right"

        if "Emergency Shelter Household 2. Transitional Housing: Household 1." in text:
            return "left"

        raise ValueError(f"Could not parse winner from response: {text!r}")

    # ------------------------------------------------------------------
    # Pair comparison
    # ------------------------------------------------------------------

    def compare(self, item_i, item_j, tie_index=None, completed_repeats=None):
        if item_i == item_j:
            raise ValueError("Cannot compare an item to itself.")

        left_item = str(item_i)
        right_item = str(item_j)

        prompt = self.get_prompt(left_item, right_item, left_label='Patient A', right_label='Patient B')

        completed_repeats = completed_repeats or set()

        self.register_pair_view(
            tie_index=tie_index,
            item_i=left_item,
            item_j=right_item,
            order="as_given",
            left_item=left_item,
            right_item=right_item,
            prompt=prompt,
        )

        left_wins_count = 0
        right_wins_count = 0

        for repeat_index in range(self.num_samples):
            if repeat_index in completed_repeats:
                print(
                    f"Skipping already-logged repeat: "
                    f"tie_index={tie_index}, repeat_index={repeat_index}"
                )
                continue

            raw_response = None
            latency_ms = None
            error_msg = None

            try:
                t0 = time.time()

                raw_response = self.call_llm(
                    prompt,
                    tie_index=tie_index,
                    repeat_index=repeat_index,
                )

                latency_ms = int((time.time() - t0) * 1000)

                winner_side = self._parse_winner(raw_response)

                if winner_side == "left":
                    winner_item = left_item
                    loser_item = right_item
                    left_wins_count += 1

                elif winner_side == "right":
                    winner_item = right_item
                    loser_item = left_item
                    right_wins_count += 1

                else:
                    raise ValueError(f"Unexpected winner_side: {winner_side}")

                winner_idx = self.get_index(winner_item)
                loser_idx = self.get_index(loser_item)

                self.win_matrix[winner_idx, loser_idx] += 1
                self.num_comparisons += 1

            except Exception as e:
                error_msg = str(e)

            self.log_raw_response(
                tie_index=tie_index,
                item_i=left_item,
                item_j=right_item,
                order="as_given",
                repeat_index=repeat_index,
                raw_response=raw_response,
                left_item=left_item,
                right_item=right_item,
                latency_ms=latency_ms,
                error=error_msg,
            )

        return {
            "left_item": left_item,
            "right_item": right_item,
            "left_wins": left_wins_count,
            "right_wins": right_wins_count,
            "num_trials": self.num_samples,
        }

    # ------------------------------------------------------------------
    # Tie sheet loop
    # ------------------------------------------------------------------

    def compare_items(self, tie_sheet: Iterable[tuple] | None = None, max_pairs=None):
        """
        Batched tie sheet loop.

        This replaces the old one-prompt-at-a-time loop with batched vLLM calls.
        Existing compare() is kept for backward compatibility, but this function
        now does the fast path directly.

        Args:
            tie_sheet:
                Optional tie sheet. If None, uses self.tie_sheet.
            max_pairs:
                Optional local-test convenience. Runs only first max_pairs pairs.
        """

        if tie_sheet is None:
            if self.tie_sheet is None:
                raise ValueError(
                    "No tie_sheet provided. Either pass tie_sheet to compare_items() "
                    "or initialize LLMComparator with tie_sheet=..."
                )

            tie_sheet = self.tie_sheet

        tie_sheet = list(tie_sheet)

        if max_pairs is not None:
            tie_sheet = tie_sheet[: int(max_pairs)]

        completed = {}
        if self.logger is not None:
            completed = self.logger.load_completed_repeats()

        total_pairs = len(tie_sheet)

        print(f"Preparing batched comparison tasks for {total_pairs} pairs...")
        print(f"Batch size: {self.batch_size}")

        tasks = []

        for tie_index, (a, b) in enumerate(tie_sheet, start=0):
            done_repeats = completed.get(tie_index, set())

            if len(done_repeats) >= self.num_samples:
                print(
                    f"Skipping completed pair "
                    f"tie_index={tie_index} "
                    f"({len(done_repeats)}/{self.num_samples} repeats already logged)"
                )
                continue

            if isinstance(a, int) and isinstance(b, int):
                item_i = self.get_item(a)
                item_j = self.get_item(b)
            else:
                item_i = str(a)
                item_j = str(b)

            left_item = str(item_i)
            right_item = str(item_j)

            prompt = self.get_prompt(left_item, right_item)

            self.register_pair_view(
                tie_index=tie_index,
                item_i=left_item,
                item_j=right_item,
                order="as_given",
                left_item=left_item,
                right_item=right_item,
                prompt=prompt,
            )

            for repeat_index in range(self.num_samples):
                if repeat_index in done_repeats:
                    continue

                tasks.append(
                    {
                        "tie_index": tie_index,
                        "item_i": left_item,
                        "item_j": right_item,
                        "left_item": left_item,
                        "right_item": right_item,
                        "repeat_index": repeat_index,
                        "prompt": prompt,
                    }
                )

        total_tasks = len(tasks)

        print(f"Total unfinished LLM calls to run: {total_tasks}")

        if total_tasks == 0:
            print("Nothing to run. All requested comparisons are already completed.")
            self.flush_logs()
            return self.win_matrix

        batch_size = max(1, int(self.batch_size))

        for start in range(0, total_tasks, batch_size):
            end = min(start + batch_size, total_tasks)
            batch_tasks = tasks[start:end]
            prompts = [task["prompt"] for task in batch_tasks]

            print(
                f"Running batch {start // batch_size + 1} "
                f"({start + 1}-{end}/{total_tasks})"
            )

            batch_t0 = time.time()

            try:
                raw_responses = self.call_llm_batch(
                    prompts,
                    batch_tasks=batch_tasks,
                )
                print("DEBUG len(raw_responses):", len(raw_responses))
                print("DEBUG first raw response:", repr(raw_responses[0]) if raw_responses else "EMPTY")

                batch_latency_ms = int((time.time() - batch_t0) * 1000)
                avg_latency_ms = int(batch_latency_ms / max(1, len(batch_tasks)))

                batch_errors = [None] * len(batch_tasks)

            except Exception as batch_error:
                print(f"Batch failed, falling back to one-by-one calls: {batch_error}")

                raw_responses = []
                batch_errors = []
                avg_latency_ms = None

                for task in batch_tasks:
                    try:
                        single_t0 = time.time()

                        raw_text = self.call_llm(
                            task["prompt"],
                            tie_index=task["tie_index"],
                            repeat_index=task["repeat_index"],
                        )

                        raw_responses.append(raw_text)
                        batch_errors.append(None)

                    except Exception as single_error:
                        raw_responses.append(None)
                        batch_errors.append(str(single_error))

            for task, raw_response, batch_error in zip(
                batch_tasks,
                raw_responses,
                batch_errors,
            ):
                tie_index = task["tie_index"]
                left_item = task["left_item"]
                right_item = task["right_item"]
                repeat_index = task["repeat_index"]

                latency_ms = avg_latency_ms
                error_msg = batch_error

                try:
                    if raw_response is None:
                        raise ValueError(error_msg or "No raw response returned.")

                    winner_side = self._parse_winner(raw_response)

                    if winner_side == "left":
                        winner_item = left_item
                        loser_item = right_item

                    elif winner_side == "right":
                        winner_item = right_item
                        loser_item = left_item

                    else:
                        raise ValueError(f"Unexpected winner_side: {winner_side}")

                    winner_idx = self.get_index(winner_item)
                    loser_idx = self.get_index(loser_item)

                    self.win_matrix[winner_idx, loser_idx] += 1
                    self.num_comparisons += 1

                except Exception as e:
                    error_msg = str(e)

                if self.logger is not None:
                    self.logger.log_response(
                        tie_index=tie_index,
                        item_a=left_item,
                        item_b=right_item,
                        order="as_given",
                        repeat_index=repeat_index,
                        seed=self._seed_for_call(tie_index, repeat_index),
                        raw_response=raw_response,
                        left_item=left_item,
                        right_item=right_item,
                        latency_ms=latency_ms,
                        error=error_msg,
                        extra={"tie_index": tie_index},
                    )
                    self.logger.flush()

        return self.win_matrix

    def reset_comparator(self):
        self.reset_win_matrix()