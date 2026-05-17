import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 32


# -------------------------
# Load model once
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()


def parse_mimic_priority_jsonl_to_csv(
    input_jsonl_path,
    output_csv_path,
    max_input_chars=4000,
    max_new_tokens=32,
    batch_size=BATCH_SIZE,
    save_parser_prompt=False,
):
    print("\nParsing")
    print(input_jsonl_path)

    input_jsonl_path = Path(input_jsonl_path)
    output_csv_path = Path(output_csv_path)

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

    def preprocess_text(raw_response: str) -> str:
        text = str(raw_response).strip()

        text = text.replace("<|eot_id|>", " ")
        text = text.replace("<|end_of_text|>", " ")
        text = re.sub(r"\s+", " ", text).strip()

        # Keep the end because final decisions often appear near the end.
        if len(text) > max_input_chars:
            text = text[-max_input_chars:]

        return text

    def build_messages(raw_response: str):
        text = preprocess_text(raw_response)

        return [
            {
                "role": "system",
                "content": (
                    "You are a deterministic information extraction system. "
                    "Your task is to read messy text and extract only the final decision "
                    "about which patient should be prioritized to be seen first in the emergency department. "
                    "Return only the required output format. "
                    "Do not explain, summarize, justify, or add extra words."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract the final prioritization decision from the text below.\n\n"
                    "There are exactly two patients:\n"
                    "1. Patient A\n"
                    "2. Patient B\n\n"
                    "Determine which patient is ultimately prioritized more.\n\n"
                    "DECISION RULES:\n"
                    "- The text may include reasoning, repetition, corrections, or multiple mentions.\n"
                    "- Use the final explicit decision in the text if one is present.\n"
                    "- Patient A and Patient B's ESI levels may be mentioned; lower ESI means higher priority.\n"
                    "- Only use ESI level if a clear final decision is not explicitly provided.\n"
                    "- If the text only gives reasoning, infer the final decision from that reasoning.\n"
                    "- Ignore irrelevant text and special tokens such as <|eot_id|>.\n"
                    "- If no clear final decision can be determined, output:\n"
                    "Patient indeterminate is prioritized\n\n"
                    "OUTPUT RULES:\n"
                    "- Output exactly one line.\n"
                    "- Output exactly one of these three forms:\n"
                    "Patient A is prioritized\n"
                    "Patient B is prioritized\n"
                    "Patient indeterminate is prioritized\n"
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

    def extract_priority_decision(generated_text: str) -> dict:
        result = {
            "prioritized_patient": None,
            "parse_status": "failed",
        }

        text = str(generated_text).strip()

        pattern = re.compile(
            r"Patient\s*(A|B|indeterminate)\s+is\s+prioritized",
            flags=re.IGNORECASE,
        )

        matches = list(pattern.finditer(text))

        if not matches:
            return result

        # If the model accidentally emits multiple decisions, keep the last one.
        final_match = matches[-1]
        label = final_match.group(1).lower()

        if label == "a":
            result["prioritized_patient"] = "Patient A"
        elif label == "b":
            result["prioritized_patient"] = "Patient B"
        elif label == "indeterminate":
            result["prioritized_patient"] = "indeterminate"

        result["parse_status"] = "ok"
        return result

    def parse_many_batched(raw_responses):
        results = []
        n = len(raw_responses)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_raw = raw_responses[start:end]

            prompts = []

            for raw_response in batch_raw:
                messages = build_messages(raw_response)

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                prompts.append(prompt)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Important:
            # Since we use left padding, do NOT slice using attention_mask.sum(dim=1).
            # All generated tokens begin after the padded prompt length.
            prompt_len = inputs["input_ids"].shape[1]

            for i in range(len(batch_raw)):
                new_tokens = outputs[i][prompt_len:]

                generated_text = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                ).strip()

                extracted = extract_priority_decision(generated_text)

                row_result = {
                    "generated_text": generated_text,
                    "prioritized_patient": extracted["prioritized_patient"],
                    "parse_status": extracted["parse_status"],
                }

                if save_parser_prompt:
                    row_result["parser_prompt"] = prompts[i]

                results.append(row_result)

            print(f"Parsed {end}/{n} rows")

        return results

    df = load_jsonl(input_jsonl_path)

    if df.empty:
        raise ValueError(f"No valid JSONL records found in {input_jsonl_path}")

    if "event" in df.columns:
        responses_df = df[df["event"] == "response"].copy()
    else:
        responses_df = df.copy()

    if "raw_response" not in responses_df.columns:
        raise ValueError("Input JSONL must contain a 'raw_response' column.")

    responses_df = responses_df[responses_df["raw_response"].notna()].copy()

    print(f"Loaded {len(responses_df)} response rows")

    raw_responses = responses_df["raw_response"].tolist()
    parsed = parse_many_batched(raw_responses)

    responses_df["generated_text"] = [x["generated_text"] for x in parsed]
    responses_df["prioritized_patient"] = [x["prioritized_patient"] for x in parsed]
    responses_df["parse_status"] = [x["parse_status"] for x in parsed]

    if save_parser_prompt:
        responses_df["parser_prompt"] = [x["parser_prompt"] for x in parsed]

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    responses_df.to_csv(output_csv_path, index=False)

    print(f"Saved parsed results to: {output_csv_path}")

    num_failed = int((responses_df["parse_status"] != "ok").sum())
    print(f"Failed parses: {num_failed}/{len(responses_df)}")

    return responses_df


inputs = [
    "/projects/simlai1/Viberank/data/VibeRank/raw/mimic/mimic_triage/rc_responses/MIMIC_500_DEEPSEEK_tournament1_seq_run.jsonl",
    "/projects/simlai1/Viberank/data/VibeRank/raw/mimic/mimic_triage/rc_responses/MIMIC_500_LLAMA7_tournament1_seq_run.jsonl",
    "/projects/simlai1/Viberank/data/VibeRank/raw/mimic/mimic_triage/rc_responses/MIMIC_500_QWEN_tournament1_seq_run.jsonl",
]

for input_path in inputs:
    input_path = Path(input_path)
    output_path = input_path.with_name(f"{input_path.stem}_parsed.csv")

    print(f"\nIN : {input_path}")
    print(f"OUT: {output_path}")

    parse_mimic_priority_jsonl_to_csv(
        input_jsonl_path=input_path,
        output_csv_path=output_path,
        batch_size=32,
        save_parser_prompt=False,
    )