METHODOLOGY

CoNLL-2003 Dataset
A widely used benchmark for Named Entity Recognition (NER) built from Reuters news articles. It provides token-level IOB annotations for four entity types (PER, ORG, LOC, MISC) and includes predefined train, validation, and test splits, making it ideal for training, evaluating, and comparing NER models under standardized settings.

Synthetic Data Generation (CoNLL-2003 → LLM)

Generated synthetic NER annotations from the CoNLL-2003 dataset using a large instruction-tuned language model to support downstream distillation and robustness training.

Method
	•	Input: Pre-tokenized CoNLL-2003 sentences
	•	Model: mistralai/Mistral-7B-Instruct-v0.2
	•	Task: Strict CoNLL-2003 NER tagging (PER / ORG / LOC / MISC)

Key Design Choices
	•	Token-preserving prompts ensure exact alignment between input tokens and output IOB tags.
	•	Constraint-driven prompting enforces valid CoNLL-style IOB rules and prevents free-form output.
	•	Deterministic inference (greedy decoding) guarantees reproducibility.
	•	Strict parsing & padding ensure one tag per token, defaulting to O when uncertain.

Usage

The resulting synthetic dataset is treated as noisy supervision and is used only for data augmentation and knowledge distillation, while gold labels remain the primary training anchor.

⸻

Student Model 1: Supervised + Weakly-Augmented NER (Baseline Distillation)

Model(s):
	•	Student: distilbert-base-cased

Methodology:
Student Model 1 follows a classical NER fine-tuning approach where a lightweight transformer is trained using:
	•	Gold CoNLL-2003 labels as the primary supervision
	•	LLM-generated synthetic labels as additional noisy training data

Training is performed with standard cross-entropy loss on token-level IOB tags, treating gold and synthetic samples equally (or with simple filtering).

Why this was tried:
	•	Establishes a strong baseline for comparison
	•	Tests whether LLM-generated labels alone can boost performance via data augmentation
	•	Simple, reproducible, and widely used in industry NER pipelines

Limitation:
	•	Assumes LLM labels are correct
	•	No explicit mechanism to handle label noise, teacher uncertainty, or representation mismatch

⸻

Student Model 2: Multi-Phase Knowledge Distillation with Teacher Guidance

Model(s):
	•	Teacher (Phase 0): dbmdz/bert-large-cased-finetuned-conll03-english (frozen, external)
	•	Student: distilbert-base-cased

Methodology:
Student Model 2 implements a research-grade distillation pipeline inspired by recent SOTA NER and KD literature:

Phase 0 – Teacher Signal Generation
	•	Teacher generates token-level labels + confidence/logits
	•	Produces weak but informative supervision, not just hard labels

Phase 1 – Teacher-Guided Distillation
	•	Student trained with dual loss:
	•	Cross-Entropy on gold labels
	•	KL-divergence between teacher and student logits (temperature τ=4)
	•	Allows student to learn soft decision boundaries

Phase 2 – Contrastive Representation Distillation (CERND)
	•	Aligns hidden representations of student and teacher
	•	Uses contrastive (InfoNCE) loss to transfer semantic structure, not just labels

Phase 3 – Iterative Self-Training
	•	Student generates pseudo-labels on unlabeled data
	•	Filters by high confidence + low entropy
	•	Gradually increases reliance on student predictions over multiple cycles

Why this was tried:
	•	Explicitly addresses LLM label noise
	•	Transfers both knowledge and representations
	•	Mimics industry-grade distillation pipelines used for production NLP models
	•	Aims for robust generalization, not just benchmark accuracy

Summary Comparison (One-liner)
	•	Student Model 1 tests whether LLM labels help at all (baseline).
	•	Student Model 2 tests how far distillation can go when uncertainty, representations, and self-training are explicitly modeled.

This makes Model 1 a control experiment, and Model 2 the research-driven, production-oriented solution.

⸻

COMPARISON METRICS


### Synthetic Data Generation Metrics

| Category                 | Metric                     | Value |
|--------------------------|----------------------------|-------|
| **Generation Performance** | Unit                       | seconds_per_batch |
|                          | Mean Time (Inference)      | 4.1 s |
|                          | P95 Time                   | 4.6 s |
|                          | Total Batches              | 250   |
| **Data Quality**         | Total Samples              | 1000  |
|                          | Avg Entities / Sample      | 1.8   |
|                          | Entity Density             | 14%   |
|                          | Avg Confidence             | 94%   |
|                          | P90 Confidence             | 100%  |

