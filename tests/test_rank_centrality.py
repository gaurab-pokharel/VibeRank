# %%


# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0] / "src"))

# %%


# %%
from datetime import datetime
from pathlib import Path

from viberank.datasets.hmls_rc_dataloader import RankCentralityDataLoader
from viberank.experiments.rank_centrality import RankCentralityExperimentRunner
from viberank.utils.logging import JSONLResponseLogger
from viberank.comparators.dummy import DummyComparator


# %%
config_path = Path("/projects/simlai1/Viberank/VibeRank/configs/datasets/rc_vispdat.yaml")

dataloader = RankCentralityDataLoader.from_yaml(config_path)
dataloader.prepare()

run_id = datetime.now().strftime("dummy_vispdat_%Y%m%d_%H%M%S")
log_path = dataloader.config.responses_dir / f"{run_id}.jsonl"

logger = JSONLResponseLogger(
    log_path=log_path,
    flush_every=1,
    store_prompts=True,
)

# %%
from viberank.comparators.LLMcomparator import LLMComparator

# %%




comp_kwargs = dataloader.get_comparator_kwargs()
"""comparator = DummyComparator(
    items=comp_kwargs["items"],
    data_folder=comp_kwargs["data_folder"],
    prompt_path=comp_kwargs["prompt_path"],
    results_folder=comp_kwargs["results_folder"],
    logger=logger,
    num_samples=1,
    rng_seed=42,
    # true_ranking=None means it generates a random one
)"""


# %%
comp_kwargs["prompt_path"] = "/projects/simlai1/Viberank/data/raw/hmls/prompt_vulnerability.txt"

# %%
pairs = comp_kwargs.pop("pairs", None)
comp = LLMComparator(
    **comp_kwargs,
    num_samples=dataloader.config.run_settings.get("repeats_per_ordered_pair", 10),
    logger = logger,
    rng_seed=42, 
    llm_name = 'qwen', # deepseek8B / llama7 / qwen
    timeout= 120,
    max_tokens = 256,
    temperature = 0.1,
    #prompt_path='prompt_vulnerability.txt'
)

# %%
runner = RankCentralityExperimentRunner(
    dataloader=dataloader,
    logger=logger,
    comparator=comp,
    run_id="rc_vispdat_qwen_001",
    model_name="qwen",
    prompt_version="v1",
)


result = runner.run()

# %%
print(result)

# %%



