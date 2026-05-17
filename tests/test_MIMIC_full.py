# %%
import sys
import gc
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path.cwd().parents[0] / "src"))

from viberank.datasets.mimic_rc_dataloader import MIMICRankCentralityDataLoader
from viberank.experiments.rank_centrality import RankCentralityExperimentRunner
from viberank.utils.logging import JSONLResponseLogger
from viberank.comparators.LLMComparatorFullData import LLMComparator

# %%
config_path = Path("/projects/simlai1/Viberank/VibeRank/configs/datasets/mimic_500.yaml")

dataloader = MIMICRankCentralityDataLoader.from_yaml(config_path)
dataloader.set_fraction_pairs(fraction_pairs=0.4)
dataloader.prepare(seed=10)  # change this to change tournament

print("Number of pairs:", len(dataloader._pairs))

# %%
prompt_path = "/projects/simlai1/Viberank/data/raw/mimic/prompt_medical.txt"

models_to_run = [
    {
        "llm_name": "qwen",
        "run_label": "QWEN",
    },
    {
        "llm_name": "llama7",
        "run_label": "LLAMA7",
    },
    {
        "llm_name": "deepseek8B",
        "run_label": "DEEPSEEK",
    },
]

results = {}

# %%
for model_cfg in models_to_run:
    llm_name = model_cfg["llm_name"]
    run_label = model_cfg["run_label"]

    run_id = f"MIMIC_500_{run_label}_tournament1_seq_run"
    log_path = dataloader.config.responses_dir / f"{run_id}.jsonl"

    print("=" * 80)
    print(f"Starting run: {run_id}")
    print(f"LLM name: {llm_name}")
    print(f"Log path: {log_path}")
    print("=" * 80)

    logger = JSONLResponseLogger(
        log_path=log_path,
        flush_every=1,
        store_prompts=True,
    )

    comp_kwargs = dataloader.get_comparator_kwargs()
    pairs = comp_kwargs.pop("pairs", None)
    comp_kwargs["prompt_path"] = prompt_path

    comp = LLMComparator(
        **comp_kwargs,
        num_samples=dataloader.config.run_settings.get("repeats_per_ordered_pair", 2),
        logger=logger,
        rng_seed=43,
        llm_name=llm_name,  # qwen / llama7 / deepseek8B
        timeout=120,
        max_tokens=64,
        temperature=0.1,
        local_test_mode=False,
        batch_size=64,
    )

    runner = RankCentralityExperimentRunner(
        dataloader=dataloader,
        logger=logger,
        comparator=comp,
        run_id=run_id,
        model_name=llm_name,
        prompt_version="v1",
    )

    result = runner.run()
    results[llm_name] = result

    print(f"Finished run: {run_id}")
    print(result)

    # Cleanup before loading the next model
    del runner
    del comp
    del logger
    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

# %%
print("=" * 80)
print("All runs finished.")
print("=" * 80)

for model_name, result in results.items():
    print(f"\nModel: {model_name}")
    print(result)
