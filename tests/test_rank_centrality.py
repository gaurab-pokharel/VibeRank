# %%

import sys
from pathlib import Path
from datetime import datetime
import gc

sys.path.insert(0, str(Path.cwd().parents[0] / "src"))

from viberank.datasets.hmls_rc_dataloader import RankCentralityDataLoader
from viberank.experiments.rank_centrality import RankCentralityExperimentRunner
from viberank.utils.logging import JSONLResponseLogger
from viberank.comparators.LLMcomparator import LLMComparator

try:
    import torch
except ImportError:
    torch = None


# %%
config_path = Path("/projects/simlai1/Viberank/VibeRank/configs/datasets/rc_vispdat.yaml")

dataloader = RankCentralityDataLoader.from_yaml(config_path)
dataloader.prepare()


# %%
prompt_path = "/projects/simlai1/Viberank/data/raw/hmls/prompt_vulnerability.txt"

models_to_run = [
    {
        "llm_name": "qwen",
        "model_name": "qwen",
        "run_prefix": "QWEN_TAYVIFPDAT_30x30VulProp",
    },
    {
        "llm_name": "llama7",
        "model_name": "llama7",
        "run_prefix": "LLAMA7_TAYVIFPDAT_30x30VulProp",
    },
    {
        "llm_name": "deepseek8B",
        "model_name": "deepseek8B",
        "run_prefix": "DEEPSEEK8B_TAYVIFPDAT_30x30VulProp",
    },
]


# %%
all_results = {}

for model_cfg in models_to_run:
    llm_name = model_cfg["llm_name"]
    model_name = model_cfg["model_name"]
    run_prefix = model_cfg["run_prefix"]

    print("\n" + "=" * 80)
    print(f"Starting model: {llm_name}")
    print("=" * 80)

    run_id = datetime.now().strftime(f"{run_prefix}_%Y%m%d_%H%M%S")
    log_path = dataloader.config.responses_dir / f"{run_id}.jsonl"

    logger = JSONLResponseLogger(
        log_path=log_path,
        flush_every=1,
        store_prompts=True,
    )

    comp_kwargs = dataloader.get_comparator_kwargs()
    comp_kwargs["prompt_path"] = prompt_path

    # Important: remove pairs if comparator does not accept it
    comp_kwargs.pop("pairs", None)

    comp = LLMComparator(
        **comp_kwargs,
        num_samples=dataloader.config.run_settings.get("repeats_per_ordered_pair", 10),
        logger=logger,
        rng_seed=42,
        llm_name=llm_name,  # qwen / llama7 / deepseek8B
        timeout=120,
        max_tokens=256,
        temperature=0.1,
        batch_size=64,     # use this only if your updated LLMComparator has batch_size
    )

    runner = RankCentralityExperimentRunner(
        dataloader=dataloader,
        logger=logger,
        comparator=comp,
        run_id=run_id,
        model_name=model_name,
        prompt_version="v1",
    )

    result = runner.run()
    all_results[model_name] = result

    print("\nFinished model:", llm_name)
    print("Log saved to:", log_path)
    print("Result:")
    print(result)

    # Cleanup before loading next vLLM model
    try:
        comp.flush_logs()
    except Exception:
        pass

    del runner
    del comp
    del logger

    gc.collect()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# %%
print("\n" + "=" * 80)
print("All models completed")
print("=" * 80)

for model_name, result in all_results.items():
    print("\nModel:", model_name)
    print(result)