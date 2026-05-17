# %%


# %%


# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[0] / "src"))
from datetime import datetime
from pathlib import Path

from viberank.datasets.hmls_rc_dataloader import RankCentralityDataLoader
from viberank.experiments.rank_centrality import RankCentralityExperimentRunner
from viberank.utils.logging import JSONLResponseLogger
from viberank.comparators.dummy import DummyComparator

# %%
from viberank.datasets.mimic_rc_dataloader import MIMICRankCentralityDataLoader

# %%
from viberank.comparators.LLMComparatorFullData import LLMComparator

# %%
#config_path = Path("../configs/datasets/mimic_500.yaml")
config_path = Path("/projects/simlai1/Viberank/VibeRank/configs/datasets/mimic_500.yaml")
dataloader = MIMICRankCentralityDataLoader.from_yaml(config_path)
dataloader.set_fraction_pairs(fraction_pairs=0.4)
dataloader.prepare(seed=10) # change this to change tournament


run_id = datetime.now().strftime("MIMIC_500_LLAMA7_tournament1")
log_path = dataloader.config.responses_dir / f"{run_id}.jsonl"

logger = JSONLResponseLogger(
    log_path=log_path,
    flush_every=1,
    store_prompts=True,
)

# %%
len(dataloader._pairs)

# %%
comp_kwargs = dataloader.get_comparator_kwargs()
pairs = comp_kwargs.pop("pairs", None)
comp_kwargs["prompt_path"] = "/projects/simlai1/Viberank/data/raw/mimic/prompt_medical.txt"
# %%
comp = LLMComparator(
    **comp_kwargs,
    num_samples=dataloader.config.run_settings.get("repeats_per_ordered_pair", 2),
    logger = logger,
    rng_seed=43, 
    llm_name = 'llama7', # deepseek8B / llama7 / qwen
    timeout= 120,
    max_tokens = 64,
    temperature = 0.1,
    local_test_mode = False,
    batch_size=64
    #prompt_path='prompt_vulnerability.txt'
)

# %%
runner = RankCentralityExperimentRunner(
    dataloader=dataloader,
    logger=logger,
    comparator=comp,
    run_id="MIMIC_500_LLAMA7_tournament1",
    model_name="deepseek8B",
    prompt_version="v1",
)

# %%
result = runner.run()

# %%
print(result)


