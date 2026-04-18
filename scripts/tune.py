# scripts/tune.py
"""
Hyperparameter tuning for DeepAuditor via grid search.
Trains each config for a reduced number of epochs, picks best val loss,
saves best checkpoint per source/label combination.

Run on Colab GPU — ~30-45 min total.
"""

import sys, os, logging, itertools, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from model import DeepAuditor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent.parent
EMB_DIR    = BASE_DIR / "data" / "processed" / "embeddings"
LABEL_DIR  = BASE_DIR / "data" / "processed" / "labels"
MODELS_DIR = BASE_DIR / "models"
S2ORC_PATH = EMB_DIR / "s2orc_safety_sentence.npy"

SOURCES = {"clip": 768, "deepface": 512}
LABELS  = {"gender": 2, "race": 7, "age": 9}

# ── grid ─────────────────────────────────────────────────────────────────────
GRID = {
    "lr":      [1e-3, 3e-4, 1e-4],
    "dropout": [0.1, 0.3, 0.5],
}
TUNE_EPOCHS  = 10   # shorter runs for search
BATCH_SIZE   = 256
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── data helpers ─────────────────────────────────────────────────────────────
import pandas as pd

def load_split(source, label_col, split):
    emb_key = source  # "clip" or "deepface"
    datasets = ["celeba", "fairface", "utkface"]
    Xs, ys = [], []
    for ds in datasets:
        ep = EMB_DIR / f"{ds}_{split}_{emb_key}.npy"
        lp = LABEL_DIR / f"{ds}_{split}_labels.csv"
        if not ep.exists() or not lp.exists():
            continue
        df = pd.read_csv(lp)
        if label_col not in df.columns:
            continue
        mask = df[label_col].notna()
        y = df.loc[mask, label_col].values.astype(int)
        X = np.load(ep, mmap_mode="r")[:len(df)][mask]
        n = min(len(X), len(y))
        Xs.append(X[:n]); ys.append(y[:n])
    if not Xs:
        return None, None
    return np.concatenate(Xs), np.concatenate(ys)


def make_loader(X, y, text_emb, shuffle=True):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    tt = torch.tensor(text_emb, dtype=torch.float32).mean(0, keepdim=True).expand(len(X), -1)
    ds = TensorDataset(Xt, tt, yt)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


# ── train one config ──────────────────────────────────────────────────────────
def train_config(vision_dim, num_classes, lr, dropout,
                 train_loader, val_loader):
    model = DeepAuditor(vision_dim=vision_dim, num_classes=num_classes,
                        dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(
        list(model.vision_proj.parameters()) +
        list(model.text_proj.parameters()) +
        list(model.classifier.parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(TUNE_EPOCHS):
        model.train()
        for Xb, tb, yb in train_loader:
            Xb, tb, yb = Xb.to(DEVICE), tb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(Xb, tb), yb)
            loss.backward()
            opt.step()

        # val
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for Xb, tb, yb in val_loader:
                Xb, tb, yb = Xb.to(DEVICE), tb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb, tb), yb).item() * len(yb)
                n += len(yb)
        val_loss /= n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())

    return best_val_loss, best_state


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    text_emb = np.load(S2ORC_PATH, mmap_mode="r")
    results  = []

    for source, vision_dim in SOURCES.items():
        for label_col, num_classes in LABELS.items():

            X_tr, y_tr = load_split(source, label_col, "train")
            X_va, y_va = load_split(source, label_col, "val")
            if X_tr is None or X_va is None:
                logger.warning(f"Skipping {source}/{label_col} — missing data")
                continue

            logger.info(f"\n{'='*55}")
            logger.info(f"Tuning: {source} / {label_col}  "
                        f"(train={len(X_tr)}, val={len(X_va)})")

            train_loader = make_loader(X_tr, y_tr, text_emb, shuffle=True)
            val_loader   = make_loader(X_va, y_va, text_emb, shuffle=False)

            best_loss   = float("inf")
            best_cfg    = None
            best_state  = None

            configs = list(itertools.product(GRID["lr"], GRID["dropout"]))
            for lr, dropout in configs:
                logger.info(f"  lr={lr}  dropout={dropout}")
                val_loss, state = train_config(
                    vision_dim, num_classes, lr, dropout,
                    train_loader, val_loader
                )
                logger.info(f"    → val_loss={val_loss:.4f}")
                results.append({
                    "source": source, "label_col": label_col,
                    "lr": lr, "dropout": dropout, "val_loss": val_loss
                })
                if val_loss < best_loss:
                    best_loss  = val_loss
                    best_cfg   = {"lr": lr, "dropout": dropout}
                    best_state = state

            logger.info(f"  BEST: {best_cfg}  val_loss={best_loss:.4f}")

            # save best model (overwrites the original checkpoint)
            model = DeepAuditor(vision_dim=vision_dim, num_classes=num_classes,
                                dropout=best_cfg["dropout"]).to("cpu")
            model.load_state_dict(best_state)
            model.eval()
            out_path = MODELS_DIR / f"deep_auditor_{source}_{label_col}.pt"
            model.save(str(out_path))
            logger.info(f"  Saved → {out_path}")

    # print summary table
    import pandas as pd
    df = pd.DataFrame(results)
    print("\n── Tuning results ──")
    print(df.to_string(index=False))

    best_per = df.loc[df.groupby(["source", "label_col"])["val_loss"].idxmin()]
    print("\n── Best config per task ──")
    print(best_per[["source","label_col","lr","dropout","val_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()