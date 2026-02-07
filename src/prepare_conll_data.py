import json
from pathlib import Path

# ---------------- CONFIG ----------------
# Path to the original CoNLL-2003 training file
INPUT_FILE = "/ner-genai-distillation/data/source/eng.train"

# Output paths:
# 1) Gold data → tokens + ground-truth NER tags
# 2) Raw data  → tokens only (used for LLM labeling / augmentation)
OUTPUT_GOLD = "/ner-genai-distillation/data/gold/gold.json"
OUTPUT_RAW = "/ner-genai-distillation/data/raw/raw.json"

# Number of sentences to extract from the dataset
NUM_SAMPLES = 1000


def parse_conll_file(filepath):
    """
    Parse a CoNLL-2003 formatted file into sentence-level samples.

    Each sentence is returned as:
    {
        "tokens": [...],
        "ner_tags": [...]
    }

    Notes:
    - Sentence boundaries are empty lines
    - Document markers (-DOCSTART-) are ignored
    - Only token and NER tag columns are used
    """
    sentences = []
    tokens = []
    ner_tags = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Empty line indicates end of a sentence
            if not line:
                if tokens:
                    sentences.append({
                        "tokens": tokens,
                        "ner_tags": ner_tags
                    })
                    tokens = []
                    ner_tags = []
                continue

            # Skip document boundary markers
            if line.startswith("-DOCSTART-"):
                continue

            parts = line.split()
            if len(parts) < 4:
                # Skip malformed lines
                continue

            token = parts[0]
            ner = parts[-1]

            tokens.append(token)
            ner_tags.append(ner)

    # Handle final sentence if file does not end with a blank line
    if tokens:
        sentences.append({
            "tokens": tokens,
            "ner_tags": ner_tags
        })

    return sentences


def main():
    # Parse CoNLL-2003 file into sentence-level structures
    sentences = parse_conll_file(INPUT_FILE)

    # Ensure dataset contains enough samples
    if len(sentences) < NUM_SAMPLES:
        raise ValueError(
            f"Not enough sentences in dataset: found {len(sentences)}"
        )

    # Select a fixed subset for reproducibility
    selected = sentences[:NUM_SAMPLES]

    gold_data = []
    raw_data = []

    # Create aligned gold and raw datasets
    for i, s in enumerate(selected):
        # Gold data retains ground-truth NER tags
        gold_data.append({
            "id": f"sample_{i:04d}",
            "tokens": s["tokens"],
            "ner_tags": s["ner_tags"]
        })

        # Raw data removes labels (used for LLM-based annotation)
        raw_data.append({
            "id": f"sample_{i:04d}",
            "tokens": s["tokens"]
        })

    # Ensure output directories exist
    Path(OUTPUT_GOLD).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_RAW).parent.mkdir(parents=True, exist_ok=True)

    # Save gold dataset
    with open(OUTPUT_GOLD, "w", encoding="utf-8") as f:
        json.dump(gold_data, f, indent=2)

    # Save raw (unlabeled) dataset
    with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    print(f"Saved {len(gold_data)} gold sentences → {OUTPUT_GOLD}")
    print(f"Saved {len(raw_data)} raw sentences → {OUTPUT_RAW}")


if __name__ == "__main__":
    main()