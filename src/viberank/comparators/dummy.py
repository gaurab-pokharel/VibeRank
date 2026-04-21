# from viberank.comparators.base import Comparator
# import random
# from typing import Iterable, Optional


# class DummyComparator(Comparator):
#     """
#     Dummy comparator for testing the pairwise-comparison pipeline.

#     This version assumes the tie sheet itself determines order.
#     So if you want both prompt orders, include both (A, B) and (B, A)
#     in the tie sheet.

#     Recommended setup:
#         - full ordered matrix tie sheet
#         - num_samples = 5 per ordered pair
#     """

#     def __init__(
#         self,
#         items,
#         num_samples=1,
#         results_folder="comparison_results",
#         data_folder="data_folder",
#         prompt_path=None,
#         logger=None,
#         true_ranking=None,
#         rng_seed: Optional[int] = None,
#     ):
#         super().__init__(
#             items=items,
#             num_samples=num_samples,
#             results_folder=results_folder,
#             data_folder=data_folder,
#             prompt_path=prompt_path,
#             logger=logger,
#         )

#         self.rng = random.Random(rng_seed)

#         if true_ranking is None:
#             self.true_ranking = list(items)
#             self.rng.shuffle(self.true_ranking)
#         else:
#             self.true_ranking = list(true_ranking)

#         if len(set(self.true_ranking)) != len(items):
#             raise ValueError("true_ranking contains duplicates.")

#         if set(self.true_ranking) != set(items):
#             raise ValueError("true_ranking must contain exactly the same items as `items`.")

#         self.true_position = {
#             item: idx for idx, item in enumerate(self.true_ranking)
#         }

#         self.prob_dict = self._build_probability_table()

#     def _build_probability_table(self):
#         """
#         Build P(item_i beats item_j) from the latent ranking.

#         Better-ranked items beat worse-ranked items with higher probability.
#         """
#         prob_dict = {}

#         if self.N <= 1:
#             return prob_dict

#         for item_i in self.items:
#             for item_j in self.items:
#                 if item_i == item_j:
#                     continue

#                 pos_i = self.true_position[item_i]
#                 pos_j = self.true_position[item_j]
#                 diff = abs(pos_i - pos_j)

#                 if pos_i < pos_j:
#                     p = 0.5 + 0.45 * (diff / (self.N - 1))
#                 else:
#                     p = 0.5 - 0.45 * (diff / (self.N - 1))

#                 prob_dict[(item_i, item_j)] = p

#         return prob_dict

#     def _sample_outcome(self, left_item, right_item):
#         """
#         Sample whether left_item beats right_item.
#         """
#         p_left_wins = self.prob_dict[(left_item, right_item)]
#         return self.rng.random() < p_left_wins

#     def call_llm(self, prompt, preferred_side=None, left_item=None, right_item=None):
#         """
#         Simulate a raw LLM response string.
#         """
#         rationales = [
#             "based on overall vulnerability indicators",
#             "based on housing instability and support needs",
#             "based on the relative severity reflected in the questionnaire",
#             "based on the comparative urgency of the two households",
#             "based on the broader pattern of need in the assessment",
#         ]
#         rationale = self.rng.choice(rationales)

#         if preferred_side == "left":
#             chosen_label = "Household 1"
#             chosen_item = left_item
#         elif preferred_side == "right":
#             chosen_label = "Household 2"
#             chosen_item = right_item
#         else:
#             chosen_label = self.rng.choice(["Household 1", "Household 2"])
#             chosen_item = left_item if chosen_label == "Household 1" else right_item

#         return (
#             f"I would prioritize {chosen_label} "
#             f"(uid={chosen_item}) for the more intensive intervention, "
#             f"{rationale}."
#         )

#     def compare(self, item_i, item_j):
#         """
#         Compare one ordered pair exactly as given.

#         That means:
#             compare(A, B) uses prompt order A then B
#             compare(B, A) uses prompt order B then A

#         Returns:
#             dict with basic counts for convenience.
#         """
#         if item_i == item_j:
#             raise ValueError("Cannot compare an item to itself.")

#         left_item = item_i
#         right_item = item_j

#         prompt = self.get_prompt(left_item, right_item)

#         self.register_pair_view(
#             item_i=item_i,
#             item_j=item_j,
#             order="as_given",
#             left_item=left_item,
#             right_item=right_item,
#             prompt=prompt,
#         )

#         left_wins_count = 0
#         right_wins_count = 0

#         for repeat_index in range(self.num_samples):
#             left_wins = self._sample_outcome(left_item, right_item)
#             preferred_side = "left" if left_wins else "right"

#             raw_response = self.call_llm(
#                 prompt,
#                 preferred_side=preferred_side,
#                 left_item=left_item,
#                 right_item=right_item,
#             )

#             self.log_raw_response(
#                 item_i=item_i,
#                 item_j=item_j,
#                 order="as_given",
#                 repeat_index=repeat_index,
#                 seed=None,
#                 raw_response=raw_response,
#                 left_item=left_item,
#                 right_item=right_item,
#             )

#             winner_item = left_item if left_wins else right_item
#             loser_item = right_item if left_wins else left_item

#             winner_idx = self.get_index(winner_item)
#             loser_idx = self.get_index(loser_item)

#             self.win_matrix[winner_idx, loser_idx] += 1
#             self.num_comparisons += 1

#             if winner_item == left_item:
#                 left_wins_count += 1
#             else:
#                 right_wins_count += 1

#         return {
#             "left_item": left_item,
#             "right_item": right_item,
#             "left_wins": left_wins_count,
#             "right_wins": right_wins_count,
#             "num_trials": self.num_samples,
#         }

#     def compare_items(self, tie_sheet: Iterable[tuple]):
#         """
#         Run comparisons over a tie sheet.

#         Supports either:
#         - pairs of indices, e.g. (0, 1)
#         - pairs of item ids, e.g. ("123", "456")
#         """
#         for a, b in tie_sheet:
#             if isinstance(a, int) and isinstance(b, int):
#                 item_i = self.get_item(a)
#                 item_j = self.get_item(b)
#             else:
#                 item_i = a
#                 item_j = b

#             self.compare(item_i, item_j)

#         self.flush_logs()
#         return self.win_matrix



#     def reset_comparator(self):
#         """
#         Reset observed comparison state, but keep the same latent true ranking.
#         """
#         self.reset_win_matrix()


from viberank.comparators.base import Comparator
import random
from typing import Iterable, Optional


class DummyComparator(Comparator):
    """
    Dummy comparator for testing the pairwise-comparison pipeline.

    Simulates LLM calls using a latent true ranking with
    probabilistic outcomes.
    """

    def __init__(
        self,
        items,
        num_samples=1,
        results_folder="comparison_results",
        data_folder="data_folder",
        prompt_path=None,
        logger=None,
        true_ranking=None,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(
            items=items,
            num_samples=num_samples,
            results_folder=results_folder,
            data_folder=data_folder,
            prompt_path=prompt_path,
            logger=logger,
        )

        self.rng = random.Random(rng_seed)

        if true_ranking is None:
            self.true_ranking = list(items)
            self.rng.shuffle(self.true_ranking)
        else:
            self.true_ranking = list(true_ranking)

        if len(set(self.true_ranking)) != len(items):
            raise ValueError("true_ranking contains duplicates.")

        if set(self.true_ranking) != set(items):
            raise ValueError("true_ranking must contain exactly the same items as `items`.")

        self.true_position = {
            item: idx for idx, item in enumerate(self.true_ranking)
        }

        self.prob_dict = self._build_probability_table()

    def _build_probability_table(self):
        prob_dict = {}

        if self.N <= 1:
            return prob_dict

        for item_i in self.items:
            for item_j in self.items:
                if item_i == item_j:
                    continue

                pos_i = self.true_position[item_i]
                pos_j = self.true_position[item_j]
                diff = abs(pos_i - pos_j)

                if pos_i < pos_j:
                    p = 0.5 + 0.45 * (diff / (self.N - 1))
                else:
                    p = 0.5 - 0.45 * (diff / (self.N - 1))

                prob_dict[(item_i, item_j)] = p

        return prob_dict

    def _sample_outcome(self, left_item, right_item):
        p_left_wins = self.prob_dict[(left_item, right_item)]
        return self.rng.random() < p_left_wins

    def call_llm(self, prompt, preferred_side=None, left_item=None, right_item=None):
        rationales = [
            "based on overall vulnerability indicators",
            "based on housing instability and support needs",
            "based on the relative severity reflected in the questionnaire",
            "based on the comparative urgency of the two households",
            "based on the broader pattern of need in the assessment",
        ]
        rationale = self.rng.choice(rationales)

        if preferred_side == "left":
            chosen_label = "Household 1"
            chosen_item = left_item
        elif preferred_side == "right":
            chosen_label = "Household 2"
            chosen_item = right_item
        else:
            chosen_label = self.rng.choice(["Household 1", "Household 2"])
            chosen_item = left_item if chosen_label == "Household 1" else right_item

        return (
            f"I would prioritize {chosen_label} "
            f"(uid={chosen_item}) for the more intensive intervention, "
            f"{rationale}."
        )

    def compare(self, item_i, item_j, tie_index=0):
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
            tie_index=tie_index,
        )

        left_wins_count = 0
        right_wins_count = 0

        for repeat_index in range(self.num_samples):
            left_wins = self._sample_outcome(left_item, right_item)
            preferred_side = "left" if left_wins else "right"

            raw_response = self.call_llm(
                prompt,
                preferred_side=preferred_side,
                left_item=left_item,
                right_item=right_item,
            )

            self.log_raw_response(
                item_i=item_i,
                item_j=item_j,
                order="as_given",
                repeat_index=repeat_index,
                seed=None,
                raw_response=raw_response,
                left_item=left_item,
                right_item=right_item,
                tie_index=tie_index, 
            )

            winner_item = left_item if left_wins else right_item
            loser_item = right_item if left_wins else left_item

            winner_idx = self.get_index(winner_item)
            loser_idx = self.get_index(loser_item)

            self.win_matrix[winner_idx, loser_idx] += 1
            self.num_comparisons += 1

            if winner_item == left_item:
                left_wins_count += 1
            else:
                right_wins_count += 1

        return {
            "left_item": left_item,
            "right_item": right_item,
            "left_wins": left_wins_count,
            "right_wins": right_wins_count,
            "num_trials": self.num_samples,
        }

    def compare_items(self, tie_sheet: Iterable[tuple]):
        for tie_index, (a, b) in enumerate(tie_sheet):
            if isinstance(a, int) and isinstance(b, int):
                item_i = self.get_item(a)
                item_j = self.get_item(b)
            else:
                item_i = a
                item_j = b

            self.compare(item_i, item_j, tie_index=tie_index)

        self.flush_logs()
        return self.win_matrix.copy()

    def reset_comparator(self):
        self.reset_win_matrix()