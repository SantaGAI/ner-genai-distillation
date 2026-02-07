import json
from pathlib import Path
from typing import List
from tqdm import tqdm
import re
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ----------------

# Instruction-tuned LLM used as offline NER teacher
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Input: tokenized sentences without labels (from CoNLL-2003)
RAW_FILE = "data/raw/raw.json"

# Output: synthetic token-level NER labels
OUTPUT_FILE = "data/synthetic/synthetic.json"

# Batch size kept small due to LLM memory footprint
BATCH_SIZE = 4
MAX_NEW_TOKENS = 256

# Use GPU if available (expected for 7B model)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- PROMPT TEMPLATE ----------------
# Strongly constrained prompt to:
# - Preserve token alignment
# - Enforce valid CoNLL IOB tagging
# - Prevent free-form explanations
PROMPT_TEMPLATE = """You are a highly accurate Named Entity Recognition (NER) system.
...
NER: tag1 tag2 tag3 ...
"""

# Allowed IOB tags (used for strict parsing)
VALID_TAGS = {
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
    "O"
}

# ---------Model & Tokenizer Loading ----------

def load_model():
    """
    Loads tokenizer and causal LM.
    Handles padding explicitly for decoder-only models.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Decoder-only models often lack pad token → map to EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,   # Reduced precision for memory efficiency
        device_map="auto"            # Automatically shard across available GPUs
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


# ------------- Prompt Construction -----------------

def build_prompt(tokens, tokenizer):
    """
    Builds a chat-style prompt while preserving token order.
    """
    messages = [
        {"role": "system", "content": "You are an expert Named Entity Recognition system."},
        {"role": "user", "content": PROMPT_TEMPLATE.format(tokens=" ".join(tokens))}
    ]

    # apply_chat_template ensures correct formatting for instruction models
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# ------------ Output Parsing & Alignment --------------

def parse_ner_output(text: str, num_tokens: int) -> List[str]:
    """
    Extracts valid IOB tags from LLM output.
    Ensures exactly one tag per input token.
    """
    if "NER:" not in text:
        # Fallback for malformed outputs
        return ["O"] * num_tokens

    ner_part = text.split("NER:", 1)[1]

    # Strict regex prevents hallucinated labels
    tags = re.findall(
        r"\b(B-PER|I-PER|B-ORG|I-ORG|B-LOC|I-LOC|B-MISC|I-MISC|O)\b",
        ner_part
    )

    # Pad or trim to maintain token-level alignment
    if len(tags) < num_tokens:
        tags += ["O"] * (num_tokens - len(tags))
    else:
        tags = tags[:num_tokens]

    return tags


# ----------- Main Synthetic Labeling Loop -------------

def main():
    tokenizer, model = load_model()

    # Load tokenized but unlabeled sentences
    with open(RAW_FILE, "r") as f:
        raw_data = json.load(f)

    synthetic_data = []

    num_batches = (len(raw_data) + BATCH_SIZE - 1) // BATCH_SIZE
    latencies = []  # Track batch-level inference latency

    for i in tqdm(
        range(0, len(raw_data), BATCH_SIZE),
        total=num_batches,
        desc="LLM NER labeling",
        unit="batch"
    ):
        batch = raw_data[i:i + BATCH_SIZE]

        # Build prompts per sentence
        prompts = [build_prompt(item["tokens"], tokenizer) for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)

        # Used to strip prompt tokens from generated output
        input_len = inputs["input_ids"].shape[1]

        # ---- latency measurement ----
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():  # Inference-only mode
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False   # Greedy decoding for determinism
            )

        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

        # Decode only generated tokens (exclude prompt)
        decoded = tokenizer.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True
        )

        # Parse and store aligned NER tags
        for item, output_text in zip(batch, decoded):
            ner_tags = parse_ner_output(output_text, len(item["tokens"]))
            synthetic_data.append({
                "id": item["id"],
                "tokens": item["tokens"],
                "ner_tags": ner_tags
            })

    # ---------- Latency Statistics (for ops) ---------------

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

    # ----------- Save Synthetic Dataset --------------
    
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(synthetic_data, f, indent=2)

    print(f"Generated {len(synthetic_data)} synthetic samples → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()