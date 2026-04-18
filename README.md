# Bias Auditor — CV Fairness Project

A computer vision bias auditing system that audits three models (CLIP, DeepFace/ArcFace, Amazon Rekognition) across three demographic datasets (CelebA, FairFace, UTKFace), grounded in ethics paper embeddings from the S2ORC safety corpus.

## Quick Start

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configure Credentials
```bash
cp .env.example .env
# Edit .env with your AWS credentials and HuggingFace token
```

### 3. Download Datasets
```bash
python scripts/make_dataset.py
```

### 4. Extract & Cache Embeddings
```bash
python scripts/build_features.py
```

### 5. Train Auditors
```bash
python scripts/train.py
```

### 6. Evaluate & Generate Results
```bash
python scripts/evaluate.py
```

## Architecture

**Three Auditor Tiers:**
1. **Naive Auditor** — baseline statistical model
2. **Classical ML Auditor** — SVM classifier
3. **Deep Learning Auditor** — late-fusion transformer model

**Fairness Metrics:**
- Demographic parity difference
- Equalized odds difference
- Equal opportunity difference

## Project Structure
See [CLAUDE.md](CLAUDE.md) for full details.

```
bias_auditor/
├── data/
│   ├── raw/               # Downloaded datasets
│   ├── processed/         # Extracted embeddings & features
│   └── outputs/           # Audit results & plots
├── scripts/
│   ├── make_dataset.py    # Download + split datasets
│   ├── build_features.py  # Extract embeddings
│   ├── model.py           # Auditor classes
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Fairness metrics
│   ├── audit_blackbox.py  # Rekognition auditing
│   └── experiment.py      # Distribution shift
├── models/                # Saved checkpoints
└── notebooks/             # EDA & exploration
```

## Key Constraints
- All three auditor tiers must be implemented
- Code is modularized into classes and functions
- No loose executable code outside `if __name__ == "__main__"` blocks
- Git workflow: feature branches → PRs → main
- app.py inference only (no training)
