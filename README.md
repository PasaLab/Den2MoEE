# Den2MoEE

Official code for **Den2MoEE: Reconstructing Dense LLMs into Expert-Specialized Mixture-of-Experts for Efficient Embedding Models**.

Den2MoEE is a unified Dense-to-MoE reconstruction framework that transforms dense LLM embedding models into efficient MoE embedding architectures through domain-aware expert specialization and routing-aware adaptation, enabling strong embedding performance with substantially reduced activated computation.

---

## 📖 Introduction

Training large-scale MoE embedding models from scratch is often prohibitively expensive. Den2MoEE addresses this challenge by reconstructing dense LLM embedding models into efficient MoE embedding models while preserving near-dense representation quality.

The framework consists of three stages:

* **Dense-to-MoE Conversion:** Construct semantically coherent experts from dense embedding models.
* **Recovery Training:** Efficiently adapt reconstructed MoE models through lightweight retraining.
* **Embedding Evaluation:** Evaluate the resulting models on standard embedding benchmarks.

Please refer to our paper for more details and experimental results.

---

## 🚀 Quick Start

### ⚙️ Preparation

Clone the repository:

```bash
git clone https://github.com/[YOUR_REPO]/Den2MoEE.git
cd Den2MoEE
```

Create and activate the environment:

```bash
conda create -n den2moee python=3.10 -y
conda activate den2moee
```

Install dependencies for recovery training:

```bash
cd ms-swift
pip install -e .
cd ..
```

Install dependencies for Dense-to-MoE conversion and evaluation:

```bash
cd dense2moe
pip install -r requirements.txt
cd ..
```

---

## 🏋️ Usage

### Dense-to-MoE Conversion

Conversion code is available in:

```text
dense2moe/
```

Please refer to:

```text
dense2moe/README.md
```

for detailed conversion instructions.

---

### Recovery Training

Training is built upon the `ms-swift` framework.

Training scripts are located in:

```text
ms-swift/scripts/den2moee/
```

Example:

```bash
bash ms-swift/scripts/den2moee/[SCRIPT_NAME].sh
```

---

### Evaluation

Evaluation scripts are provided in:

```text
dense2moe/evaluation/
```

Run MTEB evaluation with:

```bash
bash run_mteb.sh \
    --benchmark [XXX] \
    --model-path [XXX] \
    --batch-size [XX]
```

---

## 🔧 Code Structure

```text
Den2MoEE/
├── dense2moe/                     # Dense-to-MoE reconstruction framework
│   ├── evaluation/               # Evaluation scripts and benchmarks
│   └── ...                       # Conversion and expert construction modules
│
├── ms-swift/                     # Recovery training framework
│   └── scripts/den2moee/         # Training scripts for Den2MoEE
│
└── README.md
```



---

## 🤝 Acknowledgments

This project builds upon several excellent open-source efforts. We sincerely thank the authors for their valuable contributions.

* **ms-swift** for the training and adaptation framework.
* **MTEB** for providing a comprehensive benchmark for text embedding evaluation.


