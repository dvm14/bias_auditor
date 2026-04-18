"""
Train all three auditor tiers for each embedding source and each demographic label.

Saves:
  models/naive_{source}_{label}.pkl
  models/svm_{source}_{label}.pkl
  models/deep_auditor_{source}_{label}.pt   (skipped for rekognition — no embeddings)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import NaiveAuditor, ClassicalAuditor, DeepAuditor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASETS = ["celeba", "fairface", "utkface"]
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3

# CelebA only has gender — race and age are skipped for that dataset inside
# load_npy_split via the NaN filter. FairFace and UTKFace have all three.
LABEL_COLS = ["gender", "race", "age"]

SOURCES = {
    "clip":        {"vision_dim": 768,  "embed_suffix": "clip"},
    "deepface":    {"vision_dim": 512,  "embed_suffix": "deepface"},
    "rekognition": {"vision_dim": None, "embed_suffix": "rekognition"},
}


def load_npy_split(
    data_dir: Path, embed_suffix: str, split: str, label_col: str = "gender"
) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate .npy embeddings + demographic labels across all datasets.

    Rows where label_col is missing (NaN) are silently dropped, so CelebA rows
    are automatically excluded when label_col is 'race' or 'age'.

    Returns (X, y) or (None, None) if no files are found.
    """
    embeddings_dir = data_dir / "processed" / "embeddings"
    labels_dir = data_dir / "processed" / "labels"

    X_list, y_list = [], []
    for dataset in DATASETS:
        emb_path = embeddings_dir / f"{dataset}_{split}_{embed_suffix}.npy"
        labels_path = labels_dir / f"{dataset}_{split}_labels.csv"

        if not emb_path.exists():
            logger.warning(f"Embedding not found, skipping: {emb_path.name}")
            continue
        if not labels_path.exists():
            logger.warning(f"Labels not found, skipping: {labels_path.name}")
            continue

        X = np.load(emb_path).astype(np.float32)
        df = pd.read_csv(labels_path)

        if label_col not in df.columns:
            logger.warning(
                f"Column '{label_col}' not in {labels_path.name}, skipping {dataset}/{split}."
            )
            continue

        y = df[label_col].values.astype(float)
        valid = ~np.isnan(y)

        if valid.sum() == 0:
            logger.warning(
                f"No valid rows for '{label_col}' in {dataset}/{split}, skipping."
            )
            continue

        X_list.append(X[valid])
        y_list.append(y[valid].astype(int))
        logger.info(
            f"  Loaded {valid.sum()} rows from {dataset}/{split} [{label_col}]"
        )

    if not X_list:
        return None, None

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def load_rekognition_split(
    data_dir: Path, split: str, label_col: str = "gender"
) -> tuple[np.ndarray, np.ndarray]:
    """Load Rekognition confidence score features + demographic labels.

    Rekognition scores are saved as {dataset}_rekognition_scores.csv by
    audit_blackbox.py. Joins on image_id to align rows correctly.
    Returns (None, None) if files are not yet available.
    """
    outputs_dir = data_dir / "outputs" / "audit_results"
    labels_dir = data_dir / "processed" / "labels"

    X_list, y_list = [], []
    for dataset in DATASETS:
        scores_path = outputs_dir / f"{dataset}_rekognition_scores.csv"
        labels_path = labels_dir / f"{dataset}_{split}_labels.csv"

        if not scores_path.exists():
            logger.warning(
                f"Rekognition scores not found, skipping: {scores_path.name}"
            )
            continue
        if not labels_path.exists():
            logger.warning(f"Labels not found, skipping: {labels_path.name}")
            continue

        scores_df = pd.read_csv(scores_path)
        labels_df = pd.read_csv(labels_path)

        # Join on image_id so rows are aligned regardless of CSV lengths.
        merged = labels_df.merge(scores_df, on="image_id", how="inner")

        if label_col not in merged.columns:
            logger.warning(
                f"Column '{label_col}' not in merged data for {dataset}, skipping."
            )
            continue

        feature_cols = scores_df.select_dtypes(include=[np.number]).columns.tolist()
        X = merged[feature_cols].values.astype(np.float32)
        y = merged[label_col].values.astype(float)

        valid = ~np.isnan(y)
        if valid.sum() == 0:
            continue

        X_list.append(X[valid])
        y_list.append(y[valid].astype(int))
        logger.info(
            f"  Loaded {valid.sum()} rows from {dataset}/{split} (rekognition) [{label_col}]"
        )

    if not X_list:
        return None, None

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def load_text_embedding(data_dir: Path) -> np.ndarray:
    """Return the mean S2ORC sentence embedding as a (1, 384) array.

    Used as a constant text context broadcast across all samples in DeepAuditor.
    Returns None if the file doesn't exist.
    """
    path = data_dir / "processed" / "embeddings" / "s2orc_safety_sentence.npy"
    if not path.exists():
        logger.warning(f"S2ORC embedding not found: {path}")
        return None
    emb = np.load(path).astype(np.float32)
    mean_emb = emb.mean(axis=0, keepdims=True)  # (1, 384)
    logger.info(
        f"Loaded S2ORC text embedding: mean of {emb.shape} → {mean_emb.shape}"
    )
    return mean_emb


def train_naive(
    X_train: np.ndarray,
    y_train: np.ndarray,
    source: str,
    label_col: str,
    models_dir: Path,
) -> None:
    """Train NaiveAuditor and save as models/naive_{source}_{label}.pkl."""
    out_path = models_dir / f"naive_{source}_{label_col}.pkl"
    auditor = NaiveAuditor()
    auditor.fit(X_train, y_train)
    with open(out_path, "wb") as f:
        pickle.dump(auditor, f)
    logger.info(f"Saved NaiveAuditor → {out_path}")


def train_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    source: str,
    label_col: str,
    models_dir: Path,
) -> None:
    """Train ClassicalAuditor (SVM) and save as models/svm_{source}_{label}.pkl."""
    out_path = models_dir / f"svm_{source}_{label_col}.pkl"
    auditor = ClassicalAuditor()
    auditor.fit(X_train, y_train)
    with open(out_path, "wb") as f:
        pickle.dump(auditor, f)
    logger.info(f"Saved ClassicalAuditor → {out_path}")


def train_deep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    text_emb: np.ndarray,
    vision_dim: int,
    source: str,
    label_col: str,
    models_dir: Path,
    device: str,
) -> None:
    """Train DeepAuditor and save as models/deep_auditor_{source}_{label}.pt.

    text_emb is the mean S2ORC embedding (1, 384), broadcast across every
    batch — a constant ethics/safety grounding signal.
    """
    out_path = models_dir / f"deep_auditor_{source}_{label_col}.pt"

    # num_classes inferred from unique labels in training set.
    num_classes = int(y_train.max()) + 1
    model = DeepAuditor(vision_dim=vision_dim, num_classes=num_classes).to(device)

    # Build text tensor: (1, 384) → stays constant for every batch.
    text_tensor = torch.from_numpy(text_emb).to(device)  # (1, 384)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss, train_correct = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            t_batch = text_tensor.expand(X_batch.size(0), -1)  # (B, 384)
            logits = model(X_batch, t_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()

        # --- val ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                t_batch = text_tensor.expand(X_batch.size(0), -1)
                logits = model(X_batch, t_batch)
                val_loss += criterion(logits, y_batch).item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()

        # Early stopping: save best checkpoint at epoch level.
        epoch_val_loss = val_loss / len(y_val)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model.save(str(out_path))
            logger.info(f"  New best model saved (val loss: {best_val_loss:.4f})")

        logger.info(
            f"  Epoch {epoch}/{EPOCHS} — "
            f"train loss: {train_loss/len(y_train):.4f}, "
            f"train acc: {train_correct/len(y_train):.3f} | "
            f"val loss: {epoch_val_loss:.4f}, "
            f"val acc: {val_correct/len(y_val):.3f}"
        )


def main(data_dir: str = "data", models_dir: str = "models") -> None:
    """Train one full auditor set per source per demographic label."""
    data_path = Path(data_dir)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    text_emb = load_text_embedding(data_path)

    for label_col in LABEL_COLS:
        for source, cfg in SOURCES.items():
            logger.info("=" * 60)
            logger.info(f"Training source: {source} | label: {label_col}")
            logger.info("=" * 60)

            # Load train and val splits.
            if source == "rekognition":
                X_train, y_train = load_rekognition_split(
                    data_path, "train", label_col=label_col
                )
                X_val, y_val = load_rekognition_split(
                    data_path, "val", label_col=label_col
                )
            else:
                X_train, y_train = load_npy_split(
                    data_path, cfg["embed_suffix"], "train", label_col=label_col
                )
                X_val, y_val = load_npy_split(
                    data_path, cfg["embed_suffix"], "val", label_col=label_col
                )

            if X_train is None:
                logger.warning(
                    f"No training data for source='{source}' label='{label_col}', skipping."
                )
                continue

            logger.info(
                f"Train: {X_train.shape}, Val: {X_val.shape if X_val is not None else 'N/A'}"
            )

            train_naive(X_train, y_train, source, label_col, models_path)
            train_classical(X_train, y_train, source, label_col, models_path)

            # DeepAuditor is skipped for rekognition — black-box, no embeddings.
            if cfg["vision_dim"] is None:
                logger.info(
                    f"Skipping DeepAuditor for source='{source}' (black-box)."
                )
                continue

            if text_emb is None:
                logger.warning("S2ORC text embedding missing — skipping DeepAuditor.")
                continue

            if X_val is None:
                logger.warning(
                    f"No val data for source='{source}' label='{label_col}' — skipping DeepAuditor."
                )
                continue

            train_deep(
                X_train, y_train,
                X_val, y_val,
                text_emb,
                vision_dim=cfg["vision_dim"],
                source=source,
                label_col=label_col,
                models_dir=models_path,
                device=device,
            )

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()