# unBiasFace — Visual Embedding Bias Auditor

A bias auditing pipeline for visual embedding models, with an interactive web application for exploring results. Audits CLIP ViT-L/14 and DeepFace ArcFace across gender, race, and age prediction tasks using CelebA, FairFace, and UTKFace datasets.

**Live app:** [biasauditor-production.up.railway.app](https://biasauditor-production.up.railway.app)

---

## What this project does

Most computer vision systems are built on top of large pretrained visual encoders — models that were never designed to predict demographic attributes, yet encode them with surprising fidelity. This pipeline audits two widely-used encoders by measuring how much demographic information is recoverable from their embeddings, how consistently that information is encoded across demographic subgroups, and how bias scores shift when the evaluation dataset changes.

The central finding: **bias scores are not a stable property of the model alone.** CLIP race prediction accuracy swings by 17 percentage points depending solely on which dataset is used for evaluation, with no change to the model.

---

## Repository structure

```
bias_auditor/
├── app/                          # Web application
│   ├── main.py                   # FastAPI backend
│   ├── requirements.txt          # App dependencies only
│   ├── templates/
│   │   └── index.html            # Frontend (vanilla HTML/CSS/JS)
│   └── static/
│       └── app_data.json         # Pre-computed predictions + heatmaps
├── data/
│   ├── raw/                      # Downloaded face images
│   ├── processed/
│   │   ├── embeddings/           # Cached .npy embedding files
│   │   └── labels/               # Per-dataset label CSVs
│   └── outputs/
│       ├── audit_results/        # Metrics, confusion matrices, XAI figures
│       └── experiment_results/   # Cross-dataset transfer results
├── models/                       # Saved auditor checkpoints (.pt, .pkl)
├── scripts/
│   ├── make_dataset.py           # Download + split datasets
│   ├── build_features.py         # Extract CLIP + DeepFace embeddings
│   ├── model.py                  # NaiveAuditor, ClassicalAuditor, DeepAuditor
│   ├── train.py                  # Train all auditor tiers
│   ├── tune.py                   # Hyperparameter grid search
│   ├── evaluate.py               # Fairness metrics, confusion matrices, XAI
│   ├── experiment.py             # Cross-dataset distribution shift
│   ├── generate_gradcam.py       # EigenCAM heatmaps on face images
│   ├── resave_checkpoints.py     # Patches num_classes into existing checkpoints
│   └── precompute_app_data.py    # Pre-computes all app data
├── Procfile                      # Railway deployment
├── railway.json                  # Railway config
└── results_analysis.ipynb        # Results walkthrough notebook
```

---

## Architecture

**Three Auditor Tiers:**
1. **Naive Auditor** — baseline statistical model
2. **Classical ML Auditor** — SVM classifier
3. **Deep Learning Auditor** — late-fusion transformer model

**Fairness Metrics:**
- Demographic parity difference
- Equalized odds difference
- Equal opportunity difference

---

## Models

### Embedding sources
- **CLIP ViT-L/14** — 768-dim embeddings, trained on 400M image-text pairs
- **DeepFace ArcFace** — 512-dim embeddings, trained for face identity matching

### Auditor tiers
- **Naive baseline** — majority-class prediction, sets the accuracy floor
- **Classical (SVM)** — LinearSVC + CalibratedClassifierCV on pre-extracted embeddings, with SHAP explainability
- **Deep auditor** — late-fusion two-tower architecture combining visual embeddings with ethics-grounded S2ORC text representations

### Deep auditor architecture
```
Vision tower:  (B, d) → Linear(d, 512) → LayerNorm → GELU → Linear(512, 256) → 256-dim
Text tower:    (B, 384) → Linear(384, 512) → LayerNorm → GELU → Linear(512, 256) → 256-dim
Fusion:        concat [v; t] → 512-dim → Linear(512, 256) → LayerNorm → GELU → Dropout → Linear(256, C)
```
Both encoders are frozen. Only the projection MLPs and classifier head are trained.

---

## Datasets

| Dataset | Images | Labels | Used for |
|---------|--------|--------|----------|
| CelebA | 202,599 | gender | gender (secondary) |
| FairFace | 108,501 | gender, race (7 classes), age (9 groups) | all tasks (primary) |
| UTKFace | 20,000+ | gender, race (4 classes), age (9 groups) | distribution shift |

**S2ORC safety corpus** — ethics and AI safety papers encoded with all-MiniLM-L6-v2 to produce the text tower's fixed 384-dim input.

---

## Results summary

| Source | Task | Naive | SVM | Deep |
|--------|------|-------|-----|------|
| CLIP | gender | 55.6% | 96.4% | 97.2% |
| CLIP | race | 33.4% | 75.0% | 81.4% |
| CLIP | age | 21.4% | 64.1% | 68.3% |
| DeepFace | gender | 55.6% | 86.2% | 86.6% |
| DeepFace | race | 33.4% | 51.3% | 54.2% |
| DeepFace | age | 21.4% | 49.6% | 49.9% |

Key fairness finding: CLIP gender parity gap across racial subgroups is 0.060, but balloons to 0.287 across age subgroups — high aggregate accuracy coexists with substantial subgroup disparity.

---

## Setup

```bash
git clone https://github.com/dvm14/bias_auditor.git
cd bias_auditor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the pipeline

### 1. Download datasets
```bash
python scripts/make_dataset.py
```

### 2. Extract embeddings
```bash
python scripts/build_features.py
```

### 3. Train auditors
```bash
python scripts/train.py
```

### 4. Hyperparameter tuning
```bash
python scripts/tune.py
```

### 5. Evaluate
```bash
python scripts/evaluate.py
```
Outputs: `data/outputs/audit_results/audit_results.csv`, confusion matrices, SHAP plots, EigenCAM heatmaps.

### 6. Distribution shift experiment
```bash
python scripts/experiment.py
```
Outputs: `data/outputs/experiment_results/experiment_results.csv`

### 7. Generate Grad-CAM heatmaps
```bash
pip install grad-cam open_clip_torch
python scripts/generate_gradcam.py
```

### 8. Pre-compute app data
```bash
python scripts/precompute_app_data.py
```
Outputs: `app/static/app_data.json` — required before running the app.

---

## Running the app locally

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit `http://localhost:8000`

---

## Deployment

The app is deployed on Railway from the `app` branch with root directory set to `app/`.

```
Procfile: web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

No ML inference runs at runtime — all predictions, EigenCAM heatmaps, and fairness metrics are pre-computed and stored in `app/static/app_data.json`.

---

## Explainability

- **SHAP** (KernelExplainer) — applied to the SVM auditor to identify which embedding dimensions drive demographic predictions
- **EigenCAM** — applied to the CLIP ViT-L/14 backbone to produce spatial attention heatmaps showing which face regions the model attends to. EigenCAM was chosen over Grad-CAM because ViT produces sequence-shaped activations incompatible with standard Grad-CAM. Not available for DeepFace (pre-extracted embeddings, no image backbone access).

---

## Acknowledgements

Datasets: CelebA (Liu et al., 2015), FairFace (Kärkkäinen & Joo, 2021), UTKFace (Zhang et al., 2017)  
Encoders: CLIP (Radford et al., 2021), ArcFace (Deng et al., 2019)  
Text corpus: S2ORC (Lo et al., 2020)  

Duke University AIPI540 — Diya Mirji
