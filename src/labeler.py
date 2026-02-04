import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
RAW_FILE = "data/raw/raw.json"
OUTPUT_FILE = "data/synthetic/synthetic.json"

BATCH_SIZE = 4
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PROMPT_TEMPLATE = """You are a Named Entity Recognition (NER) expert.

Label the following sentence using IOB format.
Use only these entity types:
PER, ORG, LOC, MISC

Tokens:
{tokens}

Return exactly one NER tag per token, in order.
Output format:
NER: tag1 tag2 tag3 ...
"""


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model


def build_prompt(tokens: List[str]) -> str:
    return PROMPT_TEMPLATE.format(tokens=" ".join(tokens))


def parse_ner_output(text: str, num_tokens: int) -> List[str]:
    """
    Extracts NER tags from model output.
    Falls back to 'O' if alignment fails.
    """
    if "NER:" not in text:
        return ["O"] * num_tokens

    ner_part = text.split("NER:")[-1].strip()
    tags = ner_part.split()

    if len(tags) != num_tokens:
        return ["O"] * num_tokens

    return tags


def main():
    tokenizer, model = load_model()

    with open(RAW_FILE, "r") as f:
        raw_data = json.load(f)

    synthetic_data = []

    for i in range(0, len(raw_data), BATCH_SIZE):
        batch = raw_data[i:i + BATCH_SIZE]

        prompts = [build_prompt(item["tokens"]) for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for item, output_text in zip(batch, decoded):
            ner_tags = parse_ner_output(output_text, len(item["tokens"]))

            synthetic_data.append({
                "id": item["id"],
                "tokens": item["tokens"],
                "ner_tags": ner_tags
            })

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(synthetic_data, f, indent=2)

    print(f"Generated {len(synthetic_data)} synthetic samples → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()