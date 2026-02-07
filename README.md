# Hybrid NER & GenAI Distillation

## **Dataset**

**CoNLL-2003** is a widely used benchmark dataset for Named Entity Recognition (NER) built from Reuters news articles. It provides token-level *IOB annotations* for four entity types:

* **PER** – Person
* **ORG** – Organization  
* **LOC** – Location
* **MISC** – Miscellaneous

The dataset includes predefined train, validation, and test splits, making it a standard choice for training, evaluation, and fair comparison of NER models. However, considering the task completion timeline and compute resources availability, only first 1,000 sentences and 12,057 tokens are taken for the experiments from eng.train data. Gold and Raw data created subsequently with and without true labels. 

---

## **Methodology**

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

## **Comparison Metrics**

### **Synthetic Data Generation Metrics**

| **Category** | **Metric** | **Value** |
|--------------|------------|-----------|
| **Generation Performance** | Unit | `seconds_per_batch` |
| | **Mean Time** | **4.1s** |
| | P95 Time | **4.6s** |
| | **Total Batches** | **250** |
| **Data Quality** | **Total Samples** | **1,000** |
| | Avg Entities/Sample | **1.8** |
| | **Entity Density** | **14%** |
| | **Avg Confidence** | **94%** |
| | P90 Confidence | **100%** |

---

### **Student Model 1: Supervised + LLM Baseline**

| **Model** | **Precision** | **Recall** | **F1 Score** |
|-----------|---------------|------------|--------------|
| **LLM**<br>*Mistral-7B-Instruct* | **4.5%** | **5.1%** | **4.8%** |
| **Student Model 1**<br>*distilbert-base-cased* | **99.0%** | **99.0%** | **99.0%** |

---

### **Student Model 2: Multi-Phase Knowledge Distillation**

| **Phase** | **Best Epoch F1** | **Final F1** | **Val Loss** | **Improvement** |
|-----------|-------------------|--------------|--------------|-----------------|
| **Phase 1**<br>*Teacher Distillation* | **77.3%** (E3) | **77.3%** | **3.5** | **Baseline** |
| **Phase 2**<br>*Contrastive Distillation* | **81.8%** (E7) | **80.8%** | **1.12** | **+4.5%** |
| **Phase 3 Cycle 1**<br>*Self-Training* | **92.1%** (E5) | **92.1%** | **0.27** | **+14.8%** |
| **Phase 3 Cycle 2**<br>*Self-Training* | **93.3%** (E5) | **93.3%** | **0.23** | **+16.0%** |
| **Phase 3 Cycle 3**<br>*Self-Training* | **94.0%** (E5) | **94.0%** | **0.2** | **+16.7%** |
| **FINAL BEST** | | **94.0%** | **0.2** | **+16.7%** |

---

## ⚙️ **Training & Inference Environment**

| **Category** | **Details** |
|--------------|-------------|
| **Cloud Platform** | **HyperStack VM** |
| **VM Configuration** | **L40 GPU**<br>*28 Core CPU, 58 GB RAM, 100 GB Disk* |
| **Training Time** | **~1 hour**<br>*Student Models 1 & 2* |
| **Inference Time** | **25 minutes**<br>*LLM NER (250 batches/1K records)* |
