import time
import re
import requests
from typing import Iterable

from viberank.comparators.base import Comparator

class LLMComparator(Comparator):
    """_summary_

    Real LLM-backed comparator using a local vLLM/OpenAI-style server.
    """

    def __init__(
            self,
            items,
            num_samples = 1,
            results_folder = 'comparison_results',
            data_folder = "data_folder",
            prompt_path = None,
            logger = None,
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.0,
            max_tokens=256,
            timeout=120,
    ):
        super().__init__(
            items=items,
            num_samples=num_samples,
            results_folder=results_folder,
            data_folder=data_folder,
            prompt_path=prompt_path,
            logger=logger,
        )

        self.api_base_url = self.api_base_url.restrip('/')
        self.model_name = model_name
        self.temperature = temperature
        self.map_tokens = max_tokens
        self.timeout = timeout


    def call_llm(self, prompt):
        url = f"{self.api_base_url}/completions"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["text"]
    
    def _parse_winner(self, text):
        """
        Will have to reimplement with a second level LLM call 
        """


        text = " ".join(text.strip().split())

        if "Emergency Shelter Household 1. Transitional Housing: Household 2." in text:
            return "right"

        if "Emergency Shelter Household 2. Transitional Housing: Household 1." in text:
            return "left"

        raise ValueError(f"Could not parse winner from response: {text!r}")
 

    def compare(self, item_i, item_j):
        if item_i == item_j:
            raise ValueError("Cannot compare an item to itself.")

        left_item = item_i
        right_item = item_j

        prompt = self.get_prompt(left_item, right_item)

        self.register_pair_view(
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
        for a, b in tie_sheet:
            if isinstance(a, int) and isinstance(b, int):
                item_i = self.get_item(a)
                item_j = self.get_item(b)
            else:
                item_i = a
                item_j = b

            self.compare(item_i, item_j)

        self.flush_logs()
        return self.win_matrix
    
    def reset_comparator(self):
        self.reset_win_matrix()
