

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0] / "src"))

# %%
from datetime import datetime
from pathlib import Path

from viberank.datasets.hmls_rc_dataloader import RankCentralityDataLoader
from viberank.experiments.rank_centrality import RankCentralityExperimentRunner
from viberank.utils.logging import JSONLResponseLogger
from viberank.comparators.dummy import DummyComparator
from viberank.comparators.LLMComparatorFullData import LLMComparator

from viberank.datasets.hmls_dataloader_all import HMISPairwiseDataLoader


# %%
config_path = Path("/projects/simlai1/Viberank/VibeRank/configs/datasets/rc_full_data.yaml")



# %%
dataloader = HMISPairwiseDataLoader.from_yaml(config_path)
dataloader.prepare()

run_id = datetime.now().strftime("AIES_vispdat_%Y%m%d_%H%M%S")
log_path = dataloader.config.responses_dir / f"{run_id}.jsonl"

logger = JSONLResponseLogger(
    log_path=log_path,
    flush_every=1,
    store_prompts=True,
)

# %%
len(dataloader.tie_sheet)

# %%

comp_kwargs = dataloader.get_comparator_kwargs()

# %%
comp_kwargs["prompt_path"] = "/projects/simlai1/Viberank/data/raw/hmls/prompt_vulnerability.txt"

# %%
pairs = comp_kwargs.pop("pairs", None)
comp = LLMComparator(
    **comp_kwargs,
    num_samples=dataloader.config.run_settings.get("repeats_per_ordered_pair", 2),
    logger = logger,
    rng_seed=42, 
    llm_name = 'qwen', # deepseek8B / llama7 / qwen
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
    run_id="aies_vifspdat_qwen_001",
    model_name="qwen",
    prompt_version="v1",
)

# %%
result = runner.run()

# %%
print(result)


