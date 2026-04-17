import sys
from pathlib import Path
sys.path.insert(0, "/projects/simlai1/Viberank/VibeRank/src")
from datetime import datetime
from pathlib import Path

from viberank.datasets.hmls_dataloader import HMISPairwiseDataLoader
from viberank.experiments.pairwise_comparisons import PairwiseExperimentRunner
from viberank.utils.logging import JSONLResponseLogger
from viberank.comparators.dummy import DummyComparator
from viberank.comparators.LLMcomparator import LLMComparator


""""

This file sets up the model, sends prompts and logs raw responses.
Ported over to a py file to fascilitate running on slurm

"""





config_path = Path("/projects/simlai1/Viberank/VibeRank/configs/datasets/hmls.yaml")

dataloader = HMISPairwiseDataLoader.from_yaml(config_path)
dataloader.prepare()

run_id = "Vispdat_qwen_withvulnerability" #datetime.now().strftime("dummy_vispdat_%Y%m%d_%H%M%S")
log_path = dataloader.config.responses_dir / f"{run_id}.jsonl"

logger = JSONLResponseLogger(
    log_path=log_path,
    flush_every=1,
    store_prompts=True,
)

comp = LLMComparator(
    **dataloader.get_comparator_kwargs(),
    num_samples=dataloader.config.run_settings.get("repeats_per_ordered_pair", 10),
    logger = logger,
    rng_seed=42, 
    llm_name = 'qwen',
    timeout= 120,
    max_tokens = 256,
    temperature = 0.1,
    prompt_path='prompt_vulnerability.txt'
)


runner = PairwiseExperimentRunner(
    dataloader=dataloader,
    logger=logger,
    comparator=comp,
    run_id=run_id,
    model_name="qwen",
    prompt_version=dataloader.config.prompt_filename,
)

win_matrix = runner.run()


# we need to take the logfile, parse, and then generate the heatmap seperately. 
