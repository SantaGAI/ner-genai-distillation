import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
GOLD_FILE = "data/gold/gold.json"
SYNTH_FILE = "data/synthetic/synthetic.json"
MODELS_DIR = Path("models/distillation")
METRICS_DIR = Path("metrics/distillation")

MODEL_STUDENT = "distilbert/distilbert-base-cased"  # 66M params
MODEL_TEACHER = "dbmdz/bert-large-cased-finetuned-conll03-english"  # Pretrained NER teacher

MAX_LENGTH = 128
SEED = 42
LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for l, i in label2id.items()}

MODELS_DIR.mkdir(exist_ok=True, parents=True)
METRICS_DIR.mkdir(exist_ok=True, parents=True)

# ---------------- UTILITIES ----------------
def load_json(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)

def save_metrics(path: Path, metrics: Dict):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

# ---------------- PHASE 1: TEACHER-GUIDED DISTILLATION ----------------
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, alpha=0.5, temperature=4.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.alpha = alpha
        self.T = temperature
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        
        # Student forward
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Teacher forward (frozen)
        with torch.no_grad():
            teacher_inputs = {k: v.to(self.teacher.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            teacher_outputs = self.teacher(**teacher_inputs)
            teacher_logits = teacher_outputs.logits.to(student_logits.device)
        
        # Hard labels loss (gold + synthetic)
        hard_loss = None
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        # Soft distillation loss (KL divergence)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction='batchmean'
        )
        
        # Combined loss
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss if hard_loss is not None else soft_loss
        return (loss, student_outputs) if return_outputs else loss

def tokenize_and_align(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"], is_split_into_words=True,
        truncation=True, padding=True, max_length=MAX_LENGTH
    )
    
    labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[tags[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized["labels"] = labels
    return tokenized

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # ROBUST ARRAY NORMALIZATION (handles ALL Trainer formats)
    def safe_concat(arr):
        if isinstance(arr, np.ndarray):
            return arr
        if isinstance(arr, (list, tuple)):
            # Handle mixed dimensions safely
            arrays = []
            for item in arr:
                if isinstance(item, np.ndarray) and item.size > 0:
                    arrays.append(item)
            if arrays:
                return np.concatenate(arrays, axis=0)
        return np.asarray(arr)
    
    logits = safe_concat(logits)
    labels = safe_concat(labels)
    
    predictions = np.argmax(logits, axis=-1)  # [B, L]
    
    # CORRECT ITERATION: predictions=[B,L], labels=[B,L]
    true_labels = []
    true_preds = []
    
    for p_seq, l_seq in zip(predictions, labels):
        for p, l in zip(p_seq, l_seq):  # p=int, l=int
            if l != -100:  # Valid token
                true_labels.append(int(l))
                true_preds.append(int(p))
    
    if not true_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_preds, average="micro", zero_division=0
    )
    
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def confidence_filter(samples: List[Dict], min_conf: float = 0.85) -> List[Dict]:
    """Filter synthetic data by LLM confidence"""
    return [s for s in samples if s.get("sample_conf", 0) >= min_conf]

def run_phase1():
    print(" PHASE 1: Teacher-Guided Distillation")
    
    # Load data
    gold = load_json(GOLD_FILE)
    synth = load_json(SYNTH_FILE)
    
    # Split gold
    np.random.seed(SEED)
    np.random.shuffle(gold)
    split = int(0.8 * len(gold))
    train_gold, val_gold = gold[split:], gold[:split]
    
    # Filter high-confidence synthetic
    synth_filtered = confidence_filter(synth, min_conf=0.82)
    train_data = train_gold + synth_filtered[:len(train_gold)*3]  # 1:3 ratio
    
    print(f" Train: {len(train_gold)} gold + {len(train_data)-len(train_gold)} synth")
    print(f" Val: {len(val_gold)} gold")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_STUDENT)
    student = AutoModelForTokenClassification.from_pretrained(
        MODEL_STUDENT, num_labels=len(LABEL_LIST),
        id2label=id2label, label2id=label2id
    )
    teacher = AutoModelForTokenClassification.from_pretrained(MODEL_TEACHER)
    
    # Tokenize
    def tokenize_wrapper(examples):
        return tokenize_and_align(examples, tokenizer)

    train_ds = Dataset.from_list(train_data).map(tokenize_wrapper, batched=True)
    val_ds = Dataset.from_list(val_gold).map(tokenize_wrapper, batched=True)
    
    collator = DataCollatorForTokenClassification(tokenizer)
    
    # Hyperparameters (tuned)
    args = TrainingArguments(
        output_dir=str(MODELS_DIR / "phase1"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        logging_steps=50,
        save_total_limit=2
    )
    
    trainer = DistillationTrainer(
        model=student,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        teacher_model=teacher,
        alpha=0.5,
        temperature=4.0
    )
    
    trainer.train()
    metrics = trainer.evaluate()
    save_metrics(METRICS_DIR / "phase1.json", metrics)
    trainer.save_model(MODELS_DIR / "phase1")
    
    print(f"Phase 1 F1: {metrics['eval_f1']:.3f}")
    return str(MODELS_DIR / "phase1")

# ---------------- PHASE 2: CONTRASTIVE REPRESENTATION DISTILLATION ----------------
class ContrastiveDistillationTrainer(Trainer):
    def __init__(self, teacher_model, lambda_contrast=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.lambda_contrast = lambda_contrast
        # PERSISTENT PROJECTION LAYER - created ONCE
        self.proj = nn.Linear(1024, 768)
        self.proj.to(next(self.model.parameters()).device)
        self.proj.eval()  # Freeze projection weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        
        # Get hidden states from both models
        student_outputs = model(**inputs, output_hidden_states=True)
        student_hidden = student_outputs.hidden_states[-1][:, 1:-1, :]  # [B, L, 768]
        
        with torch.no_grad():
            teacher_inputs = {k: v.to(self.teacher.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            teacher_outputs = self.teacher(**teacher_inputs, output_hidden_states=True)
            teacher_hidden = teacher_outputs.hidden_states[-1][:, 1:-1, :].to(student_hidden.device)  # [B, L, 1024]
        
        # Pool representations
        teacher_pooled = teacher_hidden.mean(dim=1)  # [B, 1024]
        student_pooled = student_hidden.mean(dim=1)  # [B, 768]
        
        # REUSE PERSISTENT PROJECTION (ensures device consistency)
        teacher_hidden_proj = self.proj(teacher_pooled)  # [B, 768]
        
        # Normalize representations  
        student_norm = F.normalize(student_pooled, dim=-1)
        teacher_norm = F.normalize(teacher_hidden_proj, dim=-1)
        
        # InfoNCE contrastive loss
        logits = torch.matmul(student_norm, teacher_norm.T) / 0.07
        labels_contrast = torch.arange(student_norm.size(0)).to(student_norm.device)
        contrast_loss = F.cross_entropy(logits, labels_contrast)
        
        # Classification loss
        cls_loss = F.cross_entropy(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        loss = cls_loss + self.lambda_contrast * contrast_loss
        return (loss, student_outputs) if return_outputs else loss




def run_phase2(phase1_model_path: str):
    print(" PHASE 2: Contrastive Representation Distillation")
    
    gold = load_json(GOLD_FILE)
    synth = load_json(SYNTH_FILE)
    
    np.random.seed(SEED)
    np.random.shuffle(gold)
    split = int(0.8 * len(gold))
    train_gold, val_gold = gold[split:], gold[:split]
    
    synth_filtered = confidence_filter(synth, min_conf=0.85)
    train_data = train_gold + synth_filtered[:len(train_gold)*4]
    
    tokenizer = AutoTokenizer.from_pretrained(phase1_model_path)
    student = AutoModelForTokenClassification.from_pretrained(phase1_model_path)
    teacher = AutoModelForTokenClassification.from_pretrained(MODEL_TEACHER)
    
    def tokenize_wrapper(examples):
        return tokenize_and_align(examples, tokenizer)

    train_ds = Dataset.from_list(train_data).map(tokenize_wrapper, batched=True)
    val_ds = Dataset.from_list(val_gold).map(tokenize_wrapper, batched=True)
    
    collator = DataCollatorForTokenClassification(tokenizer)
    
    args = TrainingArguments(
        output_dir=str(MODELS_DIR / "phase2"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=8e-6,  # Lower LR for fine-tuning
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        report_to="none"
    )
    
    trainer = ContrastiveDistillationTrainer(
        model=student,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        teacher_model=teacher,
        lambda_contrast=0.3
    )
    
    trainer.train()
    metrics = trainer.evaluate()
    save_metrics(METRICS_DIR / "phase2.json", metrics)
    trainer.save_model(MODELS_DIR / "phase2")
    
    print(f" Phase 2 F1: {metrics['eval_f1']:.3f}")
    return str(MODELS_DIR / "phase2")

# ---------------- PHASE 3: ITERATIVE SELF-TRAINING ----------------
def generate_pseudo_labels(model_path: str, unlabeled_data: List[Dict], min_conf: float = 0.9) -> List[Dict]:
    """Generate pseudo-labels with confidence filtering"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    
    pseudo_labeled = []
    device = next(model.parameters()).device
    
    for item in tqdm(unlabeled_data, desc="Pseudo-labeling"):
        inputs = tokenizer(item["tokens"], is_split_into_words=True, 
                          truncation=True, padding=True, max_length=MAX_LENGTH, 
                          return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
            # Word-level alignment
            word_ids = inputs.word_ids()
            pseudo_tags = []
            word_confs = []
            
            prev_word = None
            for i, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != prev_word:
                    tag = tokenizer.decode([predictions[0, i].item()], skip_special_tokens=True)
                    tag_id = label2id.get(tag, 0)
                    conf = probs[0, i, tag_id].item()
                    
                    pseudo_tags.append(id2label[tag_id])
                    word_confs.append(conf)
                    prev_word = word_idx
            
            # Filter by confidence + entropy
            entropy = -np.sum([p * np.log(p + 1e-8) for p in probs[0, :len(word_ids)].cpu()])
            
            if np.mean(word_confs) > min_conf and entropy < 0.5:
                item["ner_tags"] = pseudo_tags
                item["token_confs"] = word_confs
                item["sample_conf"] = float(np.mean(word_confs))
                pseudo_labeled.append(item)
    
    return pseudo_labeled

def dynamic_mix_ratio(old_size: int, new_size: int) -> float:
    """Dynamic gold:synthetic ratio 1:4 → 1:1"""
    return min(4.0, max(1.0, old_size / max(new_size, 1)))

def run_phase3(phase2_model_path: str):
    print(" PHASE 3: Iterative Self-Training (3 cycles)")
    
    gold = load_json(GOLD_FILE)
    synth = load_json(SYNTH_FILE)
    
    np.random.seed(SEED)
    np.random.shuffle(gold)
    split = int(0.8 * len(gold))
    train_gold, val_gold = gold[split:], gold[:split]
    
    # Use remaining synthetic as unlabeled pool
    unlabeled_pool = [s for s in synth if s["sample_conf"] < 0.82][:5000]
    
    best_model = phase2_model_path
    best_f1 = 0
    
    for cycle in range(3):
        print(f"\n Cycle {cycle+1}/3")
        
        # Generate pseudo-labels
        pseudo = generate_pseudo_labels(best_model, unlabeled_pool, min_conf=0.9)
        print(f" Generated {len(pseudo)} high-conf pseudo samples")
        
        # Dynamic mixing
        mix_ratio = dynamic_mix_ratio(len(train_gold), len(pseudo))
        train_size = min(len(pseudo), int(len(train_gold) * mix_ratio))
        train_data = train_gold + pseudo[:train_size]
        
        tokenizer = AutoTokenizer.from_pretrained(best_model)
        student = AutoModelForTokenClassification.from_pretrained(best_model)
        
        def tokenize_wrapper(examples):
            return tokenize_and_align(examples, tokenizer)

        train_ds = Dataset.from_list(train_data).map(tokenize_wrapper, batched=True)
        val_ds = Dataset.from_list(val_gold).map(tokenize_wrapper, batched=True)
        
        collator = DataCollatorForTokenClassification(tokenizer)
        
        args = TrainingArguments(
            output_dir=str(MODELS_DIR / f"phase3_cycle{cycle}"),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-6,  # Very low for self-training
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=5,
            weight_decay=0.01,
            warmup_steps=20,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=SEED,
            report_to="none"
        )
        
        trainer = Trainer(
            model=student,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        metrics = trainer.evaluate()
        save_metrics(METRICS_DIR / f"phase3_cycle{cycle}.json", metrics)
        trainer.save_model(MODELS_DIR / f"phase3_cycle{cycle}")
        
        print(f"Cycle {cycle+1} F1: {metrics['eval_f1']:.3f} | Mix ratio: 1:{mix_ratio:.1f}")
        
        if metrics['eval_f1'] > best_f1:
            best_f1 = metrics['eval_f1']
            best_model = str(MODELS_DIR / f"phase3_cycle{cycle}")
    
    print(f"\nFINAL BEST F1: {best_f1:.3f}")
    print(f"Best model: {best_model}")
    
    return best_model

# ---------------- MAIN PIPELINE ----------------
def main():
    print("NER DISTILLATION PIPELINE")
    print("=" * 50)
    
    # Phase 1: Teacher-guided distillation
    phase1_model = run_phase1()
    
    # Phase 2: Contrastive representation matching  
    phase2_model = run_phase2(phase1_model)
    
    # Phase 3: Iterative self-training
    final_model = run_phase3(phase2_model)
    
    print(f"\nCOMPLETE! Final model: {final_model}")
    print(f" All metrics: {METRICS_DIR}")
    print(f" All models: {MODELS_DIR}")

if __name__ == "__main__":
    main()
