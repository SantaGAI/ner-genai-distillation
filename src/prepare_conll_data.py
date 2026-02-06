import json
from pathlib import Path

INPUT_FILE = "/Users/santanu/Projects/ner-genai-distillation/data/source/eng.train"
OUTPUT_GOLD = "/Users/santanu/Projects/ner-genai-distillation/data/gold/gold.json"
OUTPUT_RAW = "/Users/santanu/Projects/ner-genai-distillation/data/raw/raw.json"

NUM_SAMPLES = 1000


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
                continue

            token = parts[0]
            ner = parts[-1]

            tokens.append(token)
            ner_tags.append(ner)

    if tokens:
        sentences.append({
            "tokens": tokens,
            "ner_tags": ner_tags
        })

    return sentences


def main():
    sentences = parse_conll_file(INPUT_FILE)

    if len(sentences) < NUM_SAMPLES:
        raise ValueError(
            f"Not enough sentences in dataset: found {len(sentences)}"
        )

    selected = sentences[:NUM_SAMPLES]

    gold_data = []
    raw_data = []

    for i, s in enumerate(selected):
        gold_data.append({
            "id": f"sample_{i:04d}",
            "tokens": s["tokens"],
            "ner_tags": s["ner_tags"]
        })

        raw_data.append({
            "id": f"sample_{i:04d}",
            "tokens": s["tokens"]
        })

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