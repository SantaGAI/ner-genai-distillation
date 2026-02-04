import json
import random
from pathlib import Path


INPUT_FILE = "/Users/santanu/Projects/ner-genai-distillation/data/source/eng.train"
OUTPUT_GOLD = "/Users/santanu/Projects/ner-genai-distillation/data/gold/gold.json"
OUTPUT_RAW = "/Users/santanu/Projects/ner-genai-distillation/data/raw/raw.json"

NUM_GOLD = 200
NUM_RAW = 1200
RANDOM_SEED = 42


def parse_conll_file(filepath):
    """
    Parses a CoNLL-2003 formatted file into a list of sentences.
    Each sentence is a dict with tokens and ner_tags.
    """
    sentences = []
    tokens = []
    ner_tags = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Sentence boundary
            if not line:
                if tokens:
                    sentences.append({
                        "tokens": tokens,
                        "ner_tags": ner_tags
                    })
                    tokens = []
                    ner_tags = []
                continue

            # Ignore document markers
            if line.startswith("-DOCSTART-"):
                continue

            parts = line.split()
            if len(parts) < 4:
                # Malformed line, skip safely
                continue

            token = parts[0]
            ner = parts[-1]

            tokens.append(token)
            ner_tags.append(ner)

    # Catch last sentence if file doesn't end with newline
    if tokens:
        sentences.append({
            "tokens": tokens,
            "ner_tags": ner_tags
        })

    return sentences


def main():
    random.seed(RANDOM_SEED)

    sentences = parse_conll_file(INPUT_FILE)

    if len(sentences) < NUM_GOLD + NUM_RAW:
        raise ValueError(
            f"Not enough sentences in dataset: found {len(sentences)}"
        )

    random.shuffle(sentences)

    gold_sentences = sentences[:NUM_GOLD]
    raw_sentences = sentences[NUM_GOLD:NUM_GOLD + NUM_RAW]

    gold_data = [
        {
            "id": f"gold_{i:04d}",
            "tokens": s["tokens"],
            "ner_tags": s["ner_tags"]
        }
        for i, s in enumerate(gold_sentences)
    ]

    raw_data = [
        {
            "id": f"raw_{i:04d}",
            "tokens": s["tokens"]
        }
        for i, s in enumerate(raw_sentences)
    ]

    Path(OUTPUT_GOLD).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_RAW).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_GOLD, "w", encoding="utf-8") as f:
        json.dump(gold_data, f, indent=2)

    with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    print(f"Saved {len(gold_data)} gold sentences → {OUTPUT_GOLD}")
    print(f"Saved {len(raw_data)} raw sentences → {OUTPUT_RAW}")


if __name__ == "__main__":
    main()