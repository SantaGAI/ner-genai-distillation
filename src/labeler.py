import json
from pathlib import Path
from typing import List
from tqdm import tqdm
import re
import time
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
RAW_FILE = "data/raw/raw.json"
OUTPUT_FILE = "data/synthetic/synthetic.json"

BATCH_SIZE = 4
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = """You are a highly accurate Named Entity Recognition (NER) system.

Use ONLY the following IOB entity tags:
B-PER, I-PER,
B-ORG, I-ORG,
B-LOC, I-LOC,
B-MISC, I-MISC,
O

Each tag definition includes a brief example.
1. B-PER: Beginning of a person’s name (e.g., John in “John Smith”)
2. I-PER: Continuation of the same person’s name (e.g., Smith in “John Smith”)
3. B-ORG: Beginning of an organization name (e.g., Google in “Google Inc.”)
4. I-ORG: Continuation of the same organization name (e.g., Inc. in “Google Inc.”)
5. B-LOC: Beginning of a location name (e.g., Paris in “Paris, France”)
6. I-LOC: Continuation of the same location name (e.g., France in “Paris, France”)
7. B-MISC: Beginning of a miscellaneous named entity not covered above (e.g., Olympics, iPhone)
8. I-MISC: Continuation of the same miscellaneous entity
9. O: Token that does not belong to any named entity (example: common words, numbers, punctuation, symbols)

Task:
1. Identify named entity spans in the sentence.
2. Convert them into token-level IOB tags.

IOB tagging rules (MANDATORY):
- B-XXX marks the first token of an entity.
- I-XXX marks continuation of the SAME entity (must follow B-XXX of same type).
- I-XXX must NEVER appear without a preceding B-XXX of the same type.
- NEVER use I-XXX without a preceding B-XXX of the same type.
- Entities must be contiguous multi-word spans where applicable.
- Label punctuation, brackets, symbols as O.
- Label numbers/dates as O unless part of a clear named entity (e.g., "Flight 123" → B-MISC for "Flight" if entity-like).
- Treat hyphenated words or contractions as single tokens unless split.
- Numbers should be labeled O unless clearly part of a named entity.
- If ambiguous or uncertain, label as O.

Input tokens (space-separated):
{tokens}

Output requirements:
- Return EXACTLY one tag per input token.
- Tags must be space-separated.
- Use ONLY the allowed tags listed above.
- Do NOT include explanations or extra text.

Output format:
NER: tag1 tag2 tag3 ...
"""

VALID_TAGS = {
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
    "O"
}

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # IMPORTANT: Set pad token for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Tell model about padding token
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def build_prompt(tokens, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "You are an expert Named Entity Recognition system."
        },
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(tokens=" ".join(tokens))
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def parse_ner_output(text: str, num_tokens: int) -> List[str]:
    if "NER:" not in text:
        return ["O"] * num_tokens

    ner_part = text.split("NER:", 1)[1]

    # extract only valid IOB tags
    tags = re.findall(
        r"\b(B-PER|I-PER|B-ORG|I-ORG|B-LOC|I-LOC|B-MISC|I-MISC|O)\b",
        ner_part
    )

    if len(tags) < num_tokens:
        # pad if model under-produces
        tags = tags + ["O"] * (num_tokens - len(tags))
    else:
        tags = tags[:num_tokens]

    return tags


def main():
    tokenizer, model = load_model()

    with open(RAW_FILE, "r") as f:
        raw_data = json.load(f)

    synthetic_data = []

    num_batches = (len(raw_data) + BATCH_SIZE - 1) // BATCH_SIZE
    latencies = []  # seconds

    for i in tqdm(
        range(0, len(raw_data), BATCH_SIZE),
        total=num_batches,
        desc="LLM NER labeling",
        unit="batch"
    ):
        batch = raw_data[i:i + BATCH_SIZE]

        prompts = [build_prompt(item["tokens"], tokenizer) for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)

        # IMPORTANT: input length for slicing generated tokens
        input_len = inputs["input_ids"].shape[1]

        # ---- latency start ----
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        torch.cuda.synchronize()
        end = time.perf_counter()
        # ---- latency end ----

        latencies.append(end - start)

        decoded = tokenizer.batch_decode(
            outputs[:, input_len:],  
            skip_special_tokens=True
        )

        for item, output_text in zip(batch, decoded):
            ner_tags = parse_ner_output(output_text, len(item["tokens"]))

            synthetic_data.append({
                "id": item["id"],
                "tokens": item["tokens"],
                "ner_tags": ner_tags
            })
    
    lat = np.array(latencies)

    with open("latency_mistral_llm.json", "w") as f:
        json.dump(
            {
                "unit": "seconds_per_batch",
                "min": float(lat.min()),
                "max": float(lat.max()),
                "avg": float(lat.mean()),
                "p95": float(np.percentile(lat, 95)),
                "batches": len(latencies),
                "batch_size": BATCH_SIZE
            },
            f,
            indent=2
        )
    
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(synthetic_data, f, indent=2)
    
    print(f"Generated {len(synthetic_data)} synthetic samples → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()