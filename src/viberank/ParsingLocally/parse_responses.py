import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# Config
# =========================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
INPUT_JSONL = Path("/projects/simlai1/Viberank/data/VibeRank/raw/hmls/VISPDAT/responses/Vispdat_qwen_run.jsonl")
OUTPUT_CSV = Path("/projects/simlai1/Viberank/data/VibeRank/raw/hmls/VISPDAT/responses/Vispdat_qwen_parsed.csv")

MAX_INPUT_CHARS = 4000
MAX_NEW_TOKENS = 32


# =========================
# Load model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
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

    # Remove obvious special tokens / junk
    text = text.replace("<|eot_id|>", " ")
    text = text.replace("<|end_of_text|>", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    return text


def build_messages(raw_response: str):
    text = preprocess_text(raw_response)

    return [
        {
            "role": "system",
            "content": (
                "You are a deterministic information extraction system. "
                "Your task is to read messy text and extract only the final determination "
                "about which household is more vulnerable. "
                "Return only the required output format. "
                "Do not explain, summarize, justify, or add extra words."
            ),
        },
        {
            "role": "user",
            "content": (
                "Extract the final vulnerability decision from the text below.\n\n"
                "There are exactly two households:\n"
                "1. Household 1\n"
                "2. Household 2\n\n"
                "Determine which household is ultimately identified as more vulnerable.\n\n"
                "DECISION RULES:\n"
                "- The text may include reasoning, repetition, corrections, or multiple mentions.\n"
                "- Use the final explicit decision in the text if one is present.\n"
                "- If the text only gives reasoning, infer the final decision from that reasoning.\n"
                "- Ignore irrelevant text and special tokens such as <|eot_id|>.\n"
                "- If no clear final decision can be determined, output 'Household indeterminate is deemed more vulnerable'.\n\n"
                "OUTPUT RULES:\n"
                "- Output exactly one line.\n"
                "- Output exactly in one of these forms:\n"
                "Household 1 is deemed more vulnerable\n"
                "Household 2 is deemed more vulnerable\n"
                "Household indeterminate is deemed more vulnerable\n"
                "- Do not output anything else.\n"
                "- Do not use markdown.\n"
                "- Do not use bullet points.\n"
                "- Do not use quotes.\n"
                "- Do not explain your answer.\n\n"
                "TEXT TO PARSE:\n"
                f"<<<\n{text}\n>>>"
            ),
        },
    ]


# =========================
# Output parsing
# =========================
def extract_household_assignments(generated_text: str) -> dict:
    """
    Expected model output:
    Emergency Shelter: Household 1; Transitional Housing: Household 2
    """
    result = {
        "emergency_shelter_household": None,
        "transitional_housing_household": None,
        "parse_status": "failed",
    }

    text = generated_text.strip()

    es_match = re.search(
        r"Emergency Shelter\s*:\s*Household\s*([12])",
        text,
        flags=re.IGNORECASE,
    )
    th_match = re.search(
        r"Transitional Housing\s*:\s*Household\s*([12])",
        text,
        flags=re.IGNORECASE,
    )

    if es_match:
        result["emergency_shelter_household"] = f"Household {es_match.group(1)}"
    if th_match:
        result["transitional_housing_household"] = f"Household {th_match.group(1)}"

    if es_match and th_match:
        result["parse_status"] = "ok"
    elif es_match or th_match:
        result["parse_status"] = "partial"

    return result


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
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    generated_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    extracted = extract_household_assignments(generated_text)

    return {
        "generated_text": generated_text,
        "emergency_shelter_household": extracted["emergency_shelter_household"],
        "transitional_housing_household": extracted["transitional_housing_household"],
        "parse_status": extracted["parse_status"],
        "prompt": prompt,
    }


# =========================
# Main
# =========================
df = load_jsonl(INPUT_JSONL)

responses_df = df[df["event"] == "response"].copy()
responses_df = responses_df[responses_df["raw_response"].notna()].copy()

print(f"Loaded {len(responses_df)} response rows")



# Full run
parsed = responses_df["raw_response"].apply(parse_one)

responses_df["generated_text"] = parsed.apply(lambda x: x["generated_text"])
responses_df["emergency_shelter_household"] = parsed.apply(lambda x: x["emergency_shelter_household"])
responses_df["transitional_housing_household"] = parsed.apply(lambda x: x["transitional_housing_household"])
responses_df["parse_status"] = parsed.apply(lambda x: x["parse_status"])

responses_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved parsed results to: {OUTPUT_CSV}")