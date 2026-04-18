import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_vulnerability_jsonl_to_csv(
    input_jsonl_path,
    output_csv_path,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_input_chars=4000,
    max_new_tokens=32,
):
    """
    Load a JSONL file of model responses, parse the final vulnerability decision
    from each response using an instruction-tuned LLM, and save results to CSV.

    Parameters
    ----------
    input_jsonl_path : str or Path
        Path to the input JSONL file.
    output_csv_path : str or Path
        Path where the parsed CSV should be saved.
    model_name : str, optional
        Hugging Face model name.
    max_input_chars : int, optional
        Maximum number of characters from the raw response to include in the prompt.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate for each parse.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the parsed results.
    """
    print('parsing')
    print(input_jsonl_path)



    input_jsonl_path = Path(input_jsonl_path)
    output_csv_path = Path(output_csv_path)

    # -------------------------
    # Helpers
    # -------------------------
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

        if len(text) > max_input_chars:
            text = text[:max_input_chars]

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
                    "- If no clear final decision can be determined, output "
                    "'Household indeterminate is deemed more vulnerable'.\n\n"
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

    def extract_vulnerability_decision(generated_text: str) -> dict:
        result = {
            "more_vulnerable_household": None,
            "parse_status": "failed",
        }

        text = generated_text.strip()

        match_1 = re.search(
            r"Household\s*1\s+is\s+deemed\s+more\s+vulnerable",
            text,
            flags=re.IGNORECASE,
        )
        match_2 = re.search(
            r"Household\s*2\s+is\s+deemed\s+more\s+vulnerable",
            text,
            flags=re.IGNORECASE,
        )
        match_ind = re.search(
            r"Household\s*indeterminate\s+is\s+deemed\s+more\s+vulnerable|Indeterminate",
            text,
            flags=re.IGNORECASE,
        )

        if match_1:
            result["more_vulnerable_household"] = "Household 1"
            result["parse_status"] = "ok"
        elif match_2:
            result["more_vulnerable_household"] = "Household 2"
            result["parse_status"] = "ok"
        elif match_ind:
            result["more_vulnerable_household"] = "indeterminate"
            result["parse_status"] = "ok"

        return result

    # -------------------------
    # Load model once
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

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

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
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

        extracted = extract_vulnerability_decision(generated_text)

        return {
            "generated_text": generated_text,
            "more_vulnerable_household": extracted["more_vulnerable_household"],
            "parse_status": extracted["parse_status"],
            "prompt": prompt,
        }

    # -------------------------
    # Main run
    # -------------------------
    df = load_jsonl(input_jsonl_path)

    if "event" in df.columns:
        responses_df = df[df["event"] == "response"].copy()
    else:
        responses_df = df.copy()

    if "raw_response" not in responses_df.columns:
        raise ValueError("Input JSONL must contain a 'raw_response' column.")

    responses_df = responses_df[responses_df["raw_response"].notna()].copy()

    print(f"Loaded {len(responses_df)} response rows")

    parsed = responses_df["raw_response"].apply(parse_one)

    responses_df["generated_text"] = parsed.apply(lambda x: x["generated_text"])
    responses_df["more_vulnerable_household"] = parsed.apply(
        lambda x: x["more_vulnerable_household"]
    )
    responses_df["parse_status"] = parsed.apply(lambda x: x["parse_status"])

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    responses_df.to_csv(output_csv_path, index=False)

    print(f"Saved parsed results to: {output_csv_path}")
    return responses_df


# lets parse all 3 together in same session

parse_vulnerability_jsonl_to_csv(
    input_jsonl_path="/projects/simlai1/Viberank/data/VibeRank/raw/hmls/VISPDAT/responses/Vispdat_qwen_withvulnerability.jsonl",
    output_csv_path="/projects/simlai1/Viberank/data/VibeRank/raw/hmls/VISPDAT/responses/Vispdat_qwen_withvulnerability_parsed.csv",
)

parse_vulnerability_jsonl_to_csv(
    input_jsonl_path="/projects/simlai1/Viberank/data/VibeRank/raw/hmls/VISPDAT/responses/Vispdat_llama7_withvulnerability.jsonl",
    output_csv_path="/projects/simlai1/Viberank/data/VibeRank/raw/hmls/VISPDAT/responses/Vispdat_llama7_withvulnerability_parsed.csv",
)

parse_vulnerability_jsonl_to_csv(
    input_jsonl_path="/projects/simlai1/Viberank/data/VibeRank/raw/hmls/VISPDAT/responses/Vispdat_deepseek8B_withvulnerability.jsonl",
    output_csv_path="/projects/simlai1/Viberank/data/VibeRank/raw/hmls/VISPDAT/responses/Vispdat_deepseek8B_withvulnerability_parsed.csv",
)

