import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# Config
# =========================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_JSONL = Path(r"\projects\simlai1\Viberank\data\VibeRank\raw\hmls\VISPDAT\responses\Vispdat_qwen_run.jsonl")
OUTPUT_CSV = Path(r"\projects\simlai1\Viberank\data\VibeRank\raw\hmls\VISPDAT\responses\Vispdat_llama_parsed.csv")

MAX_INPUT_CHARS = 4000
MAX_NEW_TOKENS = 8


# =========================
# Load model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================
# Data loading
# =========================
def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON at line {line_num}")
    return pd.DataFrame(records)


# =========================
# Prompting
# =========================
def preprocess_text(raw_response: str) -> str:
    text = str(raw_response).strip()

    # Remove obvious trailing special tokens if present
    text = text.replace("<|eot_id|>", "").strip()

    # Keep size manageable
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    return text


def build_messages(raw_response: str):
    text = preprocess_text(raw_response)

    return [
        {
            "role": "system",
            "content": (
                "You are a deterministic information extractor. "
                "Return only one label from the allowed set."
            ),
        },
        {
            "role": "user",
            "content": (
                "You are performing extraction only.\n\n"
                "Return exactly one label:\n"
                "1\n"
                "2\n"
                "U\n\n"
                "Extraction rule:\n"
                "- If the text explicitly says Emergency Shelter was given to Household 1, return 1.\n"
                "- If the text explicitly says Emergency Shelter was given to Household 2, return 2.\n"
                "- Otherwise return U.\n\n"
                "Do not explain.\n"
                "Do not analyze.\n"
                "Do not infer.\n"
                "Use only the explicit assignment stated in the text.\n\n"
                f"Text:\n{text}\n\n"
                "Label:"
            ),
        },
    ]


def parse_one(raw_response: str) -> dict:
    messages = build_messages(raw_response)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    generated_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    normalized = generated_text.upper().strip()

    if normalized.startswith("1"):
        label = "1"
    elif normalized.startswith("2"):
        label = "2"
    else:
        label = "U"

    return {
        "parsed_label": label,
        "generated_text": generated_text,
        "prompt": prompt,
    }


# =========================
# Main
# =========================
df = load_jsonl(INPUT_JSONL)

responses_df = df[df["event"] == "response"].copy()
responses_df = responses_df[responses_df["raw_response"].notna()].copy()

print(f"Loaded {len(responses_df)} response rows")

# Test a few first
for x in responses_df["raw_response"].head(5):
    result = parse_one(x)
    print("=" * 100)
    print("ORIGINAL TEXT:")
    print(x[:1000])
    print()
    print("LABEL:", result["parsed_label"])
    print("GENERATED TEXT:", repr(result["generated_text"]))
    print()

# Full run
parsed = responses_df["raw_response"].apply(parse_one)

responses_df["parsed_label"] = parsed.apply(lambda x: x["parsed_label"])
responses_df["generated_text"] = parsed.apply(lambda x: x["generated_text"])

responses_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved parsed results to: {OUTPUT_CSV}")