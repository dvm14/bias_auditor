"""
Evaluate all trained auditors and compute fairness metrics.

For each (source × label_col) combination:
  - Loads test embeddings and labels from cached .npy files
  - Loads trained model checkpoints
  - Computes accuracy + three fairness metrics broken down by subgroup axis
  - Saves audit_results.csv and confusion matrix PNGs
  - Runs Grad-CAM (captum IntegratedGradients) on DeepAuditor
  - Runs SHAP on ClassicalAuditor
  - Saves XAI heatmaps to data/outputs/audit_results/xai/

Rekognition is NOT called here — handled by audit_blackbox.py.

NOTE: Expects checkpoints named {type}_{source}_{label_col}.ext, e.g.:
  models/naive_clip_gender.pkl
  models/svm_clip_gender.pkl
  models/deep_auditor_clip_gender.pt
  Train.py must save with this naming convention.
"""

import logging
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from model import DeepAuditor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASETS = ["celeba", "fairface", "utkface"]
LABEL_COLS = ["gender", "race", "age"]

# For each label_col, break down predictions by the OTHER demographic axes.
SUBGROUP_AXES = {
    "gender": ["race", "age"],
    "race":   ["gender", "age"],
    "age":    ["gender", "race"],
}

SOURCES = {
    "clip":     {"embed_suffix": "clip"},
    "deepface": {"embed_suffix": "deepface"},
}

XAI_SAMPLES = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_embeddings_and_labels(
    data_dir: Path, embed_suffix: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load test .npy embeddings + full labels df (all demographic cols) across all datasets.

    Rows in X and df are aligned by construction (same order, same datasets).
    Returns (None, None) if no files found.
    """
    embeddings_dir = data_dir / "processed" / "embeddings"
    labels_dir = data_dir / "processed" / "labels"

    X_list, df_list = [], []
    for dataset in DATASETS:
        emb_path = embeddings_dir / f"{dataset}_test_{embed_suffix}.npy"
        labels_path = labels_dir / f"{dataset}_test_labels.csv"
        if not emb_path.exists():
            logger.warning(f"Embedding not found, skipping: {emb_path.name}")
            continue
        if not labels_path.exists():
            logger.warning(f"Labels not found, skipping: {labels_path.name}")
            continue
        X_list.append(np.load(emb_path).astype(np.float32))
        df_list.append(pd.read_csv(labels_path))

    if not X_list:
        return None, None

    return (
        np.concatenate(X_list, axis=0),
        pd.concat(df_list, axis=0, ignore_index=True),
    )


def load_train_embeddings(data_dir: Path, embed_suffix: str) -> np.ndarray:
    """Load train embeddings across all datasets (used as SHAP background)."""
    embeddings_dir = data_dir / "processed" / "embeddings"
    arrays = []
    for dataset in DATASETS:
        path = embeddings_dir / f"{dataset}_train_{embed_suffix}.npy"
        if path.exists():
            arrays.append(np.load(path).astype(np.float32))
    return np.concatenate(arrays, axis=0) if arrays else None


def load_text_embedding(data_dir: Path) -> np.ndarray:
    """Return mean S2ORC sentence embedding (1, 384). None if not found."""
    path = data_dir / "processed" / "embeddings" / "s2orc_safety_sentence.npy"
    if not path.exists():
        logger.warning(f"S2ORC embedding not found: {path}")
        return None
    emb = np.load(path).astype(np.float32)
    mean_emb = emb.mean(axis=0, keepdims=True)  # (1, 384)
    logger.info(f"Loaded S2ORC text embedding: mean of {emb.shape} → {mean_emb.shape}")
    return mean_emb


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_naive(models_dir: Path, source: str, label_col: str):
    path = models_dir / f"naive_{source}_{label_col}.pkl"
    if not path.exists():
        logger.warning(f"NaiveAuditor not found: {path.name}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_classical(models_dir: Path, source: str, label_col: str):
    path = models_dir / f"svm_{source}_{label_col}.pkl"
    if not path.exists():
        logger.warning(f"ClassicalAuditor not found: {path.name}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_deep(models_dir: Path, source: str, label_col: str) -> DeepAuditor:
    """Load DeepAuditor — vision_dim is inferred automatically from checkpoint."""
    path = models_dir / f"deep_auditor_{source}_{label_col}.pt"
    if not path.exists():
        logger.warning(f"DeepAuditor not found: {path.name}")
        return None
    return DeepAuditor.load(str(path))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_deep(
    model: DeepAuditor,
    X: np.ndarray,
    text_emb: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Batch inference for DeepAuditor. Returns integer predictions."""
    model.eval()
    text_tensor = torch.from_numpy(text_emb).to(device)
    preds = []
    for start in range(0, len(X), batch_size):
        X_batch = torch.from_numpy(X[start : start + batch_size]).to(device)
        t_batch = text_tensor.expand(X_batch.size(0), -1)
        with torch.no_grad():
            logits = model(X_batch, t_batch)
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------------

def _group_rates(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict:
    """Per-group positive_rate, TPR, FPR for binary y_true/y_pred."""
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        pos, neg = yt == 1, yt == 0
        result[g] = {
            "pos_rate": float(yp.mean()),
            "tpr": float(yp[pos].mean()) if pos.any() else np.nan,
            "fpr": float(yp[neg].mean()) if neg.any() else np.nan,
        }
    return result


def fairness_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> dict:
    """Compute demographic parity, equal opportunity, equalized odds differences.

    Binary labels: standard binary definitions.
    Multi-class labels: macro-average over one-vs-rest binary decompositions.
    """
    classes = np.unique(y_true)

    def _metrics_binary(yt, yp, g):
        rates = _group_rates(yt, yp, g)
        pos_rates = [v["pos_rate"] for v in rates.values()]
        tprs = [v["tpr"] for v in rates.values() if not np.isnan(v["tpr"])]
        fprs = [v["fpr"] for v in rates.values() if not np.isnan(v["fpr"])]
        dp = max(pos_rates) - min(pos_rates)
        eo = (max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
        eod = ((max(tprs) - min(tprs)) + (max(fprs) - min(fprs))) / 2 if len(fprs) >= 2 else eo
        return dp, eo, eod

    if len(classes) == 2:
        dp, eo, eod = _metrics_binary(y_true, y_pred, groups)
    else:
        # Macro-average over one-vs-rest
        dp_list, eo_list, eod_list = [], [], []
        for c in classes:
            yt_bin = (y_true == c).astype(int)
            yp_bin = (y_pred == c).astype(int)
            _dp, _eo, _eod = _metrics_binary(yt_bin, yp_bin, groups)
            dp_list.append(_dp)
            eo_list.append(_eo)
            eod_list.append(_eod)
        dp  = float(np.mean(dp_list))
        eo  = float(np.mean(eo_list))
        eod = float(np.mean(eod_list))

    return {
        "demographic_parity": dp,
        "equal_opportunity": eo,
        "equalized_odds": eod,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source: str,
    label_col: str,
    auditor_name: str,
    out_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
    ax.set_title(f"{auditor_name} | {source} | {label_col}")
    plt.tight_layout()
    path = out_dir / f"cm_{source}_{label_col}_{auditor_name}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    logger.info(f"    Saved confusion matrix: {path.name}")


# ---------------------------------------------------------------------------
# XAI
# ---------------------------------------------------------------------------

def run_gradcam_xai(
    model: DeepAuditor,
    X_samples: np.ndarray,
    text_emb: np.ndarray,
    source: str,
    label_col: str,
    xai_dir: Path,
    device: str,
) -> None:
    """Feature attribution over vision embedding dims via captum IntegratedGradients.

    Saves a bar chart of the top-50 most attributed embedding dimensions.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        logger.warning("captum not installed — skipping Grad-CAM XAI.")
        return

    model.eval()
    text_tensor = torch.from_numpy(text_emb).to(device)

    def _forward(vision_input: torch.Tensor) -> torch.Tensor:
        t = text_tensor.expand(vision_input.size(0), -1)
        return model(vision_input, t)

    ig = IntegratedGradients(_forward)
    vision_tensor = torch.from_numpy(X_samples).float().to(device)

    attrs = ig.attribute(vision_tensor, target=1)  # target class 1 (positive)
    mean_attr = attrs.detach().cpu().numpy().__abs__().mean(axis=0)  # (vision_dim,)

    top_k = min(50, len(mean_attr))
    top_idx = np.argsort(mean_attr)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_k), mean_attr[top_idx])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"dim_{i}" for i in top_idx], fontsize=6)
    ax.set_xlabel("Mean |Attribution|")
    ax.set_title(f"Integrated Gradients (vision) — {source} / {label_col}")
    ax.invert_yaxis()
    plt.tight_layout()
    path = xai_dir / f"gradcam_{source}_{label_col}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    logger.info(f"    Saved Grad-CAM: {path.name}")


def run_shap_xai(
    auditor,
    X_train: np.ndarray,
    X_samples: np.ndarray,
    source: str,
    label_col: str,
    xai_dir: Path,
) -> None:
    """SHAP feature importance for ClassicalAuditor. Saves a top-20 bar chart."""
    try:
        import shap  # noqa: F401
    except ImportError:
        logger.warning("shap not installed — skipping SHAP XAI.")
        return

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train), min(100, len(X_train)), replace=False)
    X_background = X_train[bg_idx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = auditor.explain(X_background, X_samples)

    ## Handle both binary and multiclass SHAP output
    if isinstance(shap_values, list):
        # multiclass: list of (n_samples, n_features) arrays — average across classes
        vals = np.abs(np.stack(shap_values, axis=0)).mean(axis=0)  # (n_samples, n_features)
    elif shap_values.ndim == 3:
        # sometimes KernelExplainer returns (n_samples, n_features, n_classes)
        vals = np.abs(shap_values).mean(axis=-1)  # (n_samples, n_features)
    else:
        vals = shap_values  # binary, already (n_samples, n_features)

    mean_shap = np.abs(vals).mean(axis=0)  # (n_features,)

    mean_shap = np.abs(vals).mean(axis=0)
    top_k = min(20, len(mean_shap))
    top_idx = np.argsort(mean_shap)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_k), mean_shap[top_idx])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"dim_{i}" for i in top_idx], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP — {source} / {label_col}")
    ax.invert_yaxis()
    plt.tight_layout()
    path = xai_dir / f"shap_{source}_{label_col}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    logger.info(f"    Saved SHAP: {path.name}")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_auditor(
    auditor_name: str,
    auditor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_test: pd.DataFrame,
    label_col: str,
    source: str,
    text_emb: np.ndarray,
    device: str,
    out_dir: Path,
    results: list,
) -> None:
    """Evaluate one auditor: accuracy + fairness metrics per subgroup axis."""
    if auditor_name == "deep":
        y_pred = predict_deep(auditor, X_test, text_emb, device)
    else:
        y_pred = auditor.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)
    logger.info(f"      [{auditor_name}] overall acc: {overall_acc:.3f}")

    save_confusion_matrix(y_test, y_pred, source, label_col, auditor_name, out_dir)

    # Overall row — no subgroup breakdown, fairness metrics not applicable.
    results.append({
        "source":             source,
        "label_col":          label_col,
        "subgroup_col":       "overall",
        "subgroup_val":       -1,
        "auditor":            auditor_name,
        "accuracy":           float(overall_acc),
        "demographic_parity": None,
        "equalized_odds":     None,
        "equal_opportunity":  None,
    })

    for subgroup_col in SUBGROUP_AXES.get(label_col, []):
        if subgroup_col not in df_test.columns:
            continue

        valid_mask = ~df_test[subgroup_col].isna().values  # numpy bool array
        if not valid_mask.any():
            continue

        groups = df_test[subgroup_col].values[valid_mask].astype(int)
        yt = y_test[valid_mask]
        yp = y_pred[valid_mask]

        metrics = fairness_metrics(yt, yp, groups)

        for g_val in np.unique(groups):
            mask = groups == g_val
            results.append({
                "source":              source,
                "label_col":           label_col,
                "subgroup_col":        subgroup_col,
                "subgroup_val":        int(g_val),
                "auditor":             auditor_name,
                "accuracy":            float(accuracy_score(yt[mask], yp[mask])),
                "demographic_parity":  metrics["demographic_parity"],
                "equalized_odds":      metrics["equalized_odds"],
                "equal_opportunity":   metrics["equal_opportunity"],
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: str = "data", models_dir: str = "models") -> None:
    data_path = Path(data_dir)
    models_path = Path(models_dir)
    out_dir = data_path / "outputs" / "audit_results"
    xai_dir = out_dir / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    text_emb = load_text_embedding(data_path)
    results = []
    rng = np.random.default_rng(42)

    for source, cfg in SOURCES.items():
        logger.info("=" * 60)
        logger.info(f"Source: {source}")
        logger.info("=" * 60)

        X_test, df_test = load_test_embeddings_and_labels(data_path, cfg["embed_suffix"])
        if X_test is None:
            logger.warning(f"No test data for '{source}', skipping.")
            continue

        X_train_all = load_train_embeddings(data_path, cfg["embed_suffix"])

        for label_col in LABEL_COLS:
            if label_col not in df_test.columns:
                logger.warning(f"  Label '{label_col}' not in test data, skipping.")
                continue

            valid = ~df_test[label_col].isna()
            X = X_test[valid.values]
            y = df_test.loc[valid, label_col].values.astype(int)
            df_sub = df_test[valid].reset_index(drop=True)

            if len(X) == 0:
                continue

            logger.info(f"  label_col={label_col}  n_test={len(X)}")

            naive          = load_naive(models_path, source, label_col)
            classical      = load_classical(models_path, source, label_col)
            deep           = load_deep(models_path, source, label_col)
            deep_on_device = deep.to(device) if deep is not None else None

            for auditor_name, auditor in [("naive", naive), ("svm", classical), ("deep", deep_on_device)]:
                if auditor is None:
                    continue
                if auditor_name == "deep" and text_emb is None:
                    logger.warning("S2ORC embedding missing — skipping DeepAuditor.")
                    continue

                evaluate_auditor(
                    auditor_name, auditor, X, y, df_sub,
                    label_col, source, text_emb, device, out_dir, results,
                )

            # XAI — 20 random test samples
            sample_idx = rng.choice(len(X), min(XAI_SAMPLES, len(X)), replace=False)
            X_samples = X[sample_idx]

            if deep_on_device is not None and text_emb is not None:
                run_gradcam_xai(
                    deep_on_device, X_samples, text_emb,
                    source, label_col, xai_dir, device,
                )

            if classical is not None and X_train_all is not None:
                run_shap_xai(
                    classical, X_train_all, X_samples,
                    source, label_col, xai_dir,
                )

    # Save results CSV
    if results:
        df_results = pd.DataFrame(results)
        csv_path = out_dir / "audit_results.csv"
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df_results)} rows → {csv_path}")
    else:
        logger.warning("No results collected — check that model checkpoints exist.")

    logger.info("=" * 60)
    logger.info("Evaluation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
