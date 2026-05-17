import time
from typing import Iterable

from tqdm.auto import tqdm

from viberank.comparators.base import Comparator
from vllm import LLM, SamplingParams


class LLMComparator(Comparator):
    """
    Real LLM-backed comparator using local vLLM with batch prompting.

    Resume behavior:
    - compare_items() scans the JSONL log for completed (tie_index, repeat_index)
    - fully completed tie indices are skipped
    - partially completed tie indices resume only from missing repeats

    Batch behavior:
    - Builds all missing prompts first
    - Sends prompts to vLLM in batches
    - Logs every response separately with its original tie_index and repeat_index
    """

    def __init__(
        self,
        items,
        num_samples=1,
        results_folder="comparison_results",
        data_folder="data_folder",
        prompt_path=None,
        logger=None,
        temperature=0.0,
        max_tokens=256,
        timeout=120,
        llm_name="qwen",   # qwen, llama7, deepseek8B
        rng_seed=10,
        batch_size=32,
    ):
        super().__init__(
            items=items,
            num_samples=num_samples,
            results_folder=results_folder,
            data_folder=data_folder,
            prompt_path=prompt_path,
            logger=logger,
        )

        self.rng_seed = rng_seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.batch_size = int(batch_size)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if llm_name == "llama7":
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        elif llm_name == "deepseek8B":
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

        elif llm_name == "qwen":
            model_name = "Qwen/Qwen2.5-7B-Instruct"

        else:
            raise ValueError(f"Unknown llm_name: {llm_name}")

        self.llm_name = llm_name
        self.model_name = model_name

        self.llm = LLM(
            model=model_name,
            trust_remote_code=False,
        )

        print(f"initialized LLM: {llm_name} -> {model_name}")
        print(f"batch_size={self.batch_size}")

    def _seed_for_call(self, tie_index, repeat_index):
        """
        Deterministic per-call seed.

        Same (tie_index, repeat_index, rng_seed) => same seed every rerun.
        Different repeats => different seeds.
        """
        tie_index = -1 if tie_index is None else int(tie_index)

        return (
            int(self.rng_seed) * 1_000_003
            + tie_index * 9_176
            + int(repeat_index) * 101
        ) % (2**31 - 1)

    def _make_sampling_params(self, tie_index=None, repeat_index=None):
        seed = self._seed_for_call(tie_index, repeat_index)

        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )

    def call_llm(self, prompt, tie_index=None, repeat_index=None):
        """
        Single-prompt call retained for compatibility.
        The main compare_items() now uses batch prompting instead.
        """
        sampling_params = self._make_sampling_params(
            tie_index=tie_index,
            repeat_index=repeat_index,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        raw_text = outputs[0].outputs[0].text.strip()

        print(raw_text)

        return raw_text

    def call_llm_batch(self, jobs):
        """
        Batch vLLM call.

        Each job must contain:
        - prompt
        - tie_index
        - repeat_index

        Returns:
        - list[str] raw responses, same order as jobs
        """
        prompts = [job["prompt"] for job in jobs]

        sampling_params_list = [
            self._make_sampling_params(
                tie_index=job["tie_index"],
                repeat_index=job["repeat_index"],
            )
            for job in jobs
        ]

        outputs = self.llm.generate(
            prompts,
            sampling_params_list,
        )

        raw_texts = [
            output.outputs[0].text.strip()
            for output in outputs
        ]

        return raw_texts

    def _parse_winner(self, text):
        """
        Existing parser retained.
        """
        text = " ".join(text.strip().split())

        if "Emergency Shelter Household 1. Transitional Housing: Household 2." in text:
            return "right"

        if "Emergency Shelter Household 2. Transitional Housing: Household 1." in text:
            return "left"

        raise ValueError(f"Could not parse winner from response: {text!r}")

    def _process_one_response(self, job, raw_response, latency_ms=None, error_msg=None):
        """
        Parse one raw response, update win matrix if parse succeeds,
        and log the response no matter what.
        """
        tie_index = job["tie_index"]
        repeat_index = job["repeat_index"]

        item_i = job["item_i"]
        item_j = job["item_j"]

        left_item = job["left_item"]
        right_item = job["right_item"]

        if error_msg is None:
            try:
                winner_side = self._parse_winner(raw_response)

                if winner_side == "left":
                    winner_item = left_item
                    loser_item = right_item
                else:
                    winner_item = right_item
                    loser_item = left_item

                winner_idx = self.get_index(winner_item)
                loser_idx = self.get_index(loser_item)

                self.win_matrix[winner_idx, loser_idx] += 1
                self.num_comparisons += 1

            except Exception as e:
                error_msg = str(e)

        self.log_raw_response(
            tie_index=tie_index,
            item_i=item_i,
            item_j=item_j,
            order="as_given",
            repeat_index=repeat_index,
            raw_response=raw_response,
            left_item=left_item,
            right_item=right_item,
            latency_ms=latency_ms,
            error=error_msg,
        )

    def compare(self, item_i, item_j, tie_index=None, completed_repeats=None):
        """
        Single-pair comparison retained for compatibility.

        This still uses batched generation internally across repeats.
        """
        if item_i == item_j:
            raise ValueError("Cannot compare an item to itself.")

        completed_repeats = completed_repeats or set()

        left_item = item_i
        right_item = item_j
        prompt = self.get_prompt(left_item, right_item)

        self.register_pair_view(
            tie_index=tie_index,
            item_i=item_i,
            item_j=item_j,
            order="as_given",
            left_item=left_item,
            right_item=right_item,
            prompt=prompt,
        )

        jobs = []

        for repeat_index in range(self.num_samples):
            if repeat_index in completed_repeats:
                print(
                    f"Skipping already-logged repeat: "
                    f"tie_index={tie_index}, repeat_index={repeat_index}"
                )
                continue

            jobs.append(
                {
                    "tie_index": tie_index,
                    "repeat_index": repeat_index,
                    "item_i": item_i,
                    "item_j": item_j,
                    "left_item": left_item,
                    "right_item": right_item,
                    "prompt": prompt,
                }
            )

        left_wins_before = self.win_matrix[self.get_index(left_item), self.get_index(right_item)]
        right_wins_before = self.win_matrix[self.get_index(right_item), self.get_index(left_item)]

        for start in range(0, len(jobs), self.batch_size):
            batch_jobs = jobs[start:start + self.batch_size]

            try:
                t0 = time.time()
                raw_texts = self.call_llm_batch(batch_jobs)
                batch_latency_ms = int((time.time() - t0) * 1000)
                per_response_latency_ms = int(batch_latency_ms / max(len(batch_jobs), 1))

                for job, raw_response in zip(batch_jobs, raw_texts):
                    self._process_one_response(
                        job=job,
                        raw_response=raw_response,
                        latency_ms=per_response_latency_ms,
                        error_msg=None,
                    )

            except Exception as e:
                error_msg = str(e)

                for job in batch_jobs:
                    self._process_one_response(
                        job=job,
                        raw_response=None,
                        latency_ms=None,
                        error_msg=error_msg,
                    )

            self.flush_logs()

        left_wins_after = self.win_matrix[self.get_index(left_item), self.get_index(right_item)]
        right_wins_after = self.win_matrix[self.get_index(right_item), self.get_index(left_item)]

        return {
            "left_item": left_item,
            "right_item": right_item,
            "left_wins": int(left_wins_after - left_wins_before),
            "right_wins": int(right_wins_after - right_wins_before),
            "num_trials": self.num_samples,
        }

    def _build_missing_jobs(self, tie_sheet):
        """
        Build a flat list of missing LLM calls.

        One job = one prompt for one tie_index and one repeat_index.
        """
        completed = {}

        if self.logger is not None:
            completed = self.logger.load_completed_repeats()

        jobs = []
        total = len(tie_sheet)

        for tie_index, pair in enumerate(tie_sheet, start=0):
            a, b = pair

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
                item_i = a
                item_j = b

            if item_i == item_j:
                raise ValueError(f"Cannot compare an item to itself: {item_i}")

            left_item = item_i
            right_item = item_j

            prompt = self.get_prompt(left_item, right_item)

            self.register_pair_view(
                tie_index=tie_index,
                item_i=item_i,
                item_j=item_j,
                order="as_given",
                left_item=left_item,
                right_item=right_item,
                prompt=prompt,
            )

            missing_repeats = [
                repeat_index
                for repeat_index in range(self.num_samples)
                if repeat_index not in done_repeats
            ]

            print(
                f"Queued tie_index={tie_index} "
                f"with {len(done_repeats)}/{self.num_samples} repeats already logged "
                f"and {len(missing_repeats)} missing "
                f"({tie_index + 1}/{total})"
            )

            for repeat_index in missing_repeats:
                jobs.append(
                    {
                        "tie_index": tie_index,
                        "repeat_index": repeat_index,
                        "item_i": item_i,
                        "item_j": item_j,
                        "left_item": left_item,
                        "right_item": right_item,
                        "prompt": prompt,
                    }
                )

        return jobs

    def compare_items(self, tie_sheet: Iterable[tuple]):
        """
        Main batch-prompting version.
        """
        tie_sheet = list(tie_sheet)

        jobs = self._build_missing_jobs(tie_sheet)

        print(f"Total missing LLM calls to run: {len(jobs)}")
        print(f"Batch size: {self.batch_size}")

        if len(jobs) == 0:
            print("Nothing to run. All repeats already completed.")
            self.flush_logs()
            return self.win_matrix

        for start in tqdm(
            range(0, len(jobs), self.batch_size),
            desc="Running LLM batches",
        ):
            batch_jobs = jobs[start:start + self.batch_size]

            try:
                t0 = time.time()
                raw_texts = self.call_llm_batch(batch_jobs)
                batch_latency_ms = int((time.time() - t0) * 1000)
                per_response_latency_ms = int(batch_latency_ms / max(len(batch_jobs), 1))

                for job, raw_response in zip(batch_jobs, raw_texts):
                    self._process_one_response(
                        job=job,
                        raw_response=raw_response,
                        latency_ms=per_response_latency_ms,
                        error_msg=None,
                    )

            except Exception as e:
                error_msg = str(e)

                print(f"Batch failed: {error_msg}")

                for job in batch_jobs:
                    self._process_one_response(
                        job=job,
                        raw_response=None,
                        latency_ms=None,
                        error_msg=error_msg,
                    )

            self.flush_logs()

        self.flush_logs()
        return self.win_matrix

    def reset_comparator(self):
        self.reset_win_matrix()