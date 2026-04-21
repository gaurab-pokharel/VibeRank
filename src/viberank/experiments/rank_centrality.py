from __future__ import annotations

from typing import Any


class RankCentralityExperimentRunner:
    """
    Runs a Rank Centrality pairwise-comparison experiment using:
    - a prepared RankCentralityDataLoader
    - a logger
    - an initialized comparator
    """

    def __init__(
        self,
        *,
        dataloader,
        logger,
        comparator,
        run_id: str,
        model_name: str,
        prompt_version: str | None = None,
        extra_run_metadata: dict[str, Any] | None = None,
    ):
        self.dataloader = dataloader
        self.logger = logger
        self.comparator = comparator
        self.run_id = run_id
        self.model_name = model_name
        self.prompt_version = prompt_version
        self.extra_run_metadata = extra_run_metadata or {}

    def _validate(self) -> None:
        loader_items = list(self.dataloader.get_items())
        comparator_items = list(self.comparator.items)
        if loader_items != comparator_items:
            raise ValueError(
                "Comparator items do not match dataloader items in order/content."
            )

        if getattr(self.comparator, "logger", None) is not self.logger:
            raise ValueError(
                "Comparator.logger must be the same logger passed to the runner."
            )

    def run(self):
        self.dataloader.prepare()
        self._validate()

        run_meta = self.dataloader.get_run_metadata()
        run_meta.update(self.extra_run_metadata)

        self.logger.log_run_start(
            run_id=self.run_id,
            dataset_name=self.dataloader.config.dataset_name,
            model_name=self.model_name,
            prompt_version=self.prompt_version,
            extra=run_meta,
        )

        result = self.comparator.compare_items(self.dataloader.pairs)

        self.logger.log_run_end(
            run_id=self.run_id,
            extra={
                "num_items": self.dataloader.num_items,
                "num_pairs": self.dataloader.num_pairs,
                "num_logged_comparisons": int(
                    getattr(self.comparator, "num_comparisons", 0)
                ),
            },
        )

        self.logger.close()
        return result