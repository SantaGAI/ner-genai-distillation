# Hybrid NER & GenAI Distillation

## Dataset

**CoNLL-2003** is a widely used benchmark dataset for Named Entity Recognition (NER) built from Reuters news articles. It provides token-level *IOB annotations* for four entity types:

* **PER** – Person
* **ORG** – Organization  
* **LOC** – Location
* **MISC** – Miscellaneous

The dataset includes predefined train, validation, and test splits, making it a standard choice for training, evaluation, and fair comparison of NER models.

---

## Methodology

### **Synthetic Data Generation (CoNLL-2003 → LLM)**

Synthetic NER annotations are generated from the CoNLL-2003 dataset using a large instruction-tuned language model to support downstream data augmentation and knowledge distillation.

#### **Method**
* **Input**: Pre-tokenized CoNLL-2003 sentences
* **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
* **Task**: Strict CoNLL-2003 NER tagging (PER / ORG / LOC / MISC)

#### **Key Design Choices**
* Token-preserving prompts ensure exact alignment between input tokens and output IOB tags
* Constraint-driven prompting enforces valid CoNLL IOB rules and prevents free-form outputs
* Deterministic inference (*greedy decoding*) guarantees reproducibility
* Strict parsing & padding ensure exactly one tag per token, defaulting to `O` when uncertain

#### **Usage**
The resulting synthetic dataset is treated as *noisy supervision* and is used only for data augmentation and knowledge distillation.  
**Gold labels** remain the primary training anchor.

---

### **Student Model 1: Supervised + Weakly-Augmented NER (Baseline)**

#### **Model**
* **Student**: `distilbert-base-cased`

#### **Methodology**
Student Model 1 follows a classical NER fine-tuning pipeline, where the student model is trained using:
* Gold CoNLL-2003 labels as primary supervision
* LLM-generated synthetic labels as additional noisy data

Training uses standard token-level cross-entropy loss, treating gold and synthetic samples equally (or with minimal filtering).

#### **Why This Was Tried**
* Establishes a strong baseline for comparison
* Tests whether LLM-generated labels alone improve performance
* Simple, reproducible, and commonly used in industry NER pipelines

#### **Limitation**
* Assumes LLM labels are correct
* No explicit handling of label noise, teacher uncertainty, or representation mismatch

---

### **Student Model 2: Multi-Phase Knowledge Distillation with Teacher Guidance**

#### **Models**
* **Teacher (Phase 0)**: `dbmdz/bert-large-cased-finetuned-conll03-english` (*frozen*)
* **Student**: `distilbert-base-cased`

#### **Methodology**
Student Model 2 implements a *research-grade, multi-phase distillation pipeline* inspired by recent SOTA NER and KD literature.

##### **Phase 0 – Teacher Signal Generation**
* Teacher produces token-level labels and confidence/logits
* Generates *soft, informative supervision*, not just hard labels

##### **Phase 1 – Teacher-Guided Distillation**
* Student trained with a dual loss:
  * Cross-Entropy on gold labels
  * KL-Divergence between teacher and student logits
* **Temperature**: τ = 4
* Enables learning of soft decision boundaries

##### **Phase 2 – Contrastive Representation Distillation (CERND)**
* Aligns hidden representations of teacher and student
* Uses contrastive (*InfoNCE*) loss
* Transfers semantic structure, not just labels

##### **Phase 3 – Iterative Self-Training**
* Student generates pseudo-labels on unlabeled data
* Filters by high confidence and low entropy
* Gradually increases reliance on student predictions over multiple cycles

#### **Why This Was Tried**
* Explicitly addresses label noise and uncertainty
* Transfers both knowledge and representations
* Reflects industry-grade distillation pipelines used in production NLP
* Targets robust generalization, not just benchmark accuracy

---

### **Summary Comparison**
* **Student Model 1** → Tests whether LLM labels help at all (*baseline*)
* **Student Model 2** → Tests how far distillation can go when uncertainty, representations, and self-training are explicitly modeled

**Model 1** acts as the control experiment, while **Model 2** represents the research-driven, production-oriented solution.

---


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

