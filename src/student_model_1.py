import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import f1_score, precision_score, recall_score

# ---------------- CONFIG ----------------

MODEL_NAME = "distilbert-base-cased"
OUTPUT_DIR = "models/student-ner"

TRAIN_FILE = "data/source/eng.train"

MAX_LENGTH = 128
TRAIN_SPLIT = 0.9  # 90% train, 10% validation
RANDOM_SEED = 42

LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]

label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for i, l in enumerate(LABEL_LIST)}

# ---------------- DATA LOADING ----------------

def parse_conll(filepath: str) -> List[Dict]:
    """
    Parse CoNLL-2003 file into HuggingFace-ready samples
    """
    samples = []
    tokens, tags = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if tokens:
                    samples.append({
                        "tokens": tokens,
                        "ner_tags": tags
                    })
                    tokens, tags = [], []
                continue

            if line.startswith("-DOCSTART-"):
                continue

            parts = line.split()
            token = parts[0]
            ner = parts[-1]

            tokens.append(token)
            tags.append(ner)

    if tokens:
        samples.append({"tokens": tokens, "ner_tags": tags})

    return samples

# ---------------- TOKENIZATION ----------------

def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    labels = []
    for i, ner_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[ner_tags[word_id]])
            else:
                label_ids.append(-100)
            prev_word_id = word_id

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

# ---------------- METRICS ----------------

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    true_preds = []

    for pred, lab in zip(predictions, labels):
        seq_preds = []
        seq_labels = []
        for p, l in zip(pred, lab):
            if l != -100:
                seq_preds.append(id2label[p])
                seq_labels.append(id2label[l])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds)
    }

# ---------------- MAIN ----------------

def main():
    np.random.seed(RANDOM_SEED)

    # Load and split data
    samples = parse_conll(TRAIN_FILE)
    dataset = Dataset.from_list(samples)

    dataset = dataset.shuffle(seed=RANDOM_SEED)
    split = dataset.train_test_split(train_size=TRAIN_SPLIT)

    train_ds = split["train"]
    val_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )

    val_ds = val_ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=50,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("\n Final Validation Metrics")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n Student model saved to {OUTPUT_DIR}")

# ---------------- RUN ----------------

if __name__ == "__main__":
    main()