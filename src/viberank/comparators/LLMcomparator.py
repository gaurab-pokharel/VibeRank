import time
from typing import Iterable

from viberank.comparators.base import Comparator
from vllm import LLM, SamplingParams


class LLMComparator(Comparator):
    """
    Real LLM-backed comparator using local vLLM.

    Resume behavior:
    - compare_items() scans the JSONL log for completed (tie_index, repeat_index)
    - fully completed tie indices are skipped
    - partially completed tie indices resume only from missing repeats
    """

    def __init__(
        self,
        items,
        num_samples=1,
        results_folder='comparison_results',
        data_folder="data_folder",
        prompt_path=None,
        logger=None,
        temperature=0.0,
        max_tokens=256,
        timeout=120,
        llm_name='qwen',   # qwen, llama
        rng_seed = 10
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

        if llm_name == 'llama7':
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            self.llm = LLM(
                model=model_name,
                trust_remote_code=False,
            )
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        elif llm_name == 'deepseek8B':
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
            self.llm = LLM(
                model=model_name,
                trust_remote_code=False,
            )
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        elif llm_name == 'qwen':
            model_name = "Qwen/Qwen2.5-7B-Instruct"
            self.llm = LLM(
                model=model_name,
                trust_remote_code=False,
            )
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
   
        else:
            raise ValueError(f"Unknown llm_name: {llm_name}")

        print('initialized LLM')

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
    
    def call_llm(self, prompt, tie_index=None, repeat_index=None):
        seed = self._seed_for_call(tie_index, repeat_index)
        sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed = seed
            )
        outputs = self.llm.generate([prompt], sampling_params)
        raw_text = outputs[0].outputs[0].text.strip()

        # Debug print
        print(raw_text)

        return raw_text

   

    def _parse_winner(self, text):
        """
        Existing parser retained here because your current compare()
        still updates the win matrix. If later you fully switch to
        raw-output-only logging, this can be removed.
        """
        text = " ".join(text.strip().split())

        if "Emergency Shelter Household 1. Transitional Housing: Household 2." in text:
            return "right"

        if "Emergency Shelter Household 2. Transitional Housing: Household 1." in text:
            return "left"

        raise ValueError(f"Could not parse winner from response: {text!r}")

    def compare(self, item_i, item_j, tie_index=None, completed_repeats=None):
        if item_i == item_j:
            raise ValueError("Cannot compare an item to itself.")

        left_item = item_i
        right_item = item_j
        prompt = self.get_prompt(left_item, right_item)

        completed_repeats = completed_repeats or set()

        self.register_pair_view(
            tie_index=tie_index,
            item_i=item_i,
            item_j=item_j,
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
                raw_response = self.call_llm(prompt)
                latency_ms = int((time.time() - t0) * 1000)

                winner_side = self._parse_winner(raw_response)

                if winner_side == "left":
                    winner_item = left_item
                    loser_item = right_item
                    left_wins_count += 1
                else:
                    winner_item = right_item
                    loser_item = left_item
                    right_wins_count += 1

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

        return {
            "left_item": left_item,
            "right_item": right_item,
            "left_wins": left_wins_count,
            "right_wins": right_wins_count,
            "num_trials": self.num_samples,
        }

    def compare_items(self, tie_sheet: Iterable[tuple]):
        tie_sheet = list(tie_sheet)

        completed = {}
        if self.logger is not None:
            completed = self.logger.load_completed_repeats()

        total = len(tie_sheet)

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
                item_i = a
                item_j = b

            print(
                f"Processing tie_index={tie_index} "
                f"with {len(done_repeats)}/{self.num_samples} repeats already logged "
                f"({tie_index + 1}/{total})"
            )

            self.compare(
                item_i,
                item_j,
                tie_index=tie_index,
                completed_repeats=done_repeats,
            )

        self.flush_logs()
        return self.win_matrix

    def reset_comparator(self):
        self.reset_win_matrix()