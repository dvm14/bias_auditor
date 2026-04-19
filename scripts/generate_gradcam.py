# This work was done with the help of Claude (https://claude.ai/share/8c1843e5-e882-409c-aae5-b0a144a43d20)
# scripts/generate_gradcam.py
"""
Generate Grad-CAM heatmaps overlaid on face images using the CLIP ViT backbone.
Produces one figure per label_col showing a grid of correct (green) and incorrect
(red) predictions with heatmaps overlaid on the original face image.

Requirements:
    pip install grad-cam open_clip_torch

Usage:
    python scripts/generate_gradcam.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import DeepAuditor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
EMB_DIR    = BASE_DIR / "data" / "processed" / "embeddings"
LABEL_DIR  = BASE_DIR / "data" / "processed" / "labels"
MODELS_DIR = BASE_DIR / "models"
OUT_DIR    = BASE_DIR / "data" / "outputs" / "audit_results" / "xai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S2ORC_PATH = EMB_DIR / "s2orc_safety_sentence.npy"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

N_CORRECT   = 4
N_INCORRECT = 4
SCAN_LIMIT  = 500

LABEL_NAMES = {
    "gender": {0: "Female", 1: "Male"},
    "race": {
        0: "East Asian",
        1: "Indian",
        2: "Black",
        3: "White",
        4: "Middle Eastern",
        5: "Latino",
        6: "Southeast Asian",
    },
    "age": {
        0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29",
        4: "30-39", 5: "40-49", 6: "50-59",
        7: "60-69", 8: "70+",
    },
}

LABEL_DATASET = {
    "gender": "celeba",
    "race":   "fairface",
    "age":    "fairface",
}


# ── ViT reshape transform ─────────────────────────────────────────────────────
def reshape_transform_vit(tensor, height=16, width=16):
    result = tensor[:, 1:, :]
    result = result.reshape(result.size(0), height, width, result.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


# ── CLIP wrapper ──────────────────────────────────────────────────────────────
class CLIPVisionWrapper(torch.nn.Module):
    def __init__(self, clip_model, auditor, text_emb_mean):
        super().__init__()
        self.clip     = clip_model
        self.auditor  = auditor
        self.text_emb = text_emb_mean

    def forward(self, pixel_values):
        vis = self.clip.encode_image(pixel_values).float()
        vis = vis / vis.norm(dim=-1, keepdim=True)
        txt = self.text_emb.expand(vis.shape[0], -1)
        return self.auditor(vis, txt)


# ── data loading ──────────────────────────────────────────────────────────────
def load_images_and_labels(dataset_name, label_col, split="test", limit=SCAN_LIMIT):
    import pandas as pd

    label_path = LABEL_DIR / f"{dataset_name}_{split}_labels.csv"
    if not label_path.exists():
        logger.warning(f"Label file not found: {label_path}")
        return []

    df = pd.read_csv(label_path)
    if label_col not in df.columns:
        logger.warning(f"Column {label_col} not in {label_path.name}")
        return []

    df = df.dropna(subset=[label_col]).head(limit)

    results = []
    for _, row in df.iterrows():
        # Use image_path column directly from the CSV
        img_path = BASE_DIR / row["image_path"]
        if not img_path.exists():
            continue
        try:
            img   = Image.open(img_path).convert("RGB")
            label = int(row[label_col])
            results.append((img, label))
        except Exception:
            continue

    logger.info(f"Loaded {len(results)} images from {dataset_name}/{label_col}")
    return results


# ── main generation function ──────────────────────────────────────────────────
def generate_gradcam_figure(label_col):
    import open_clip

    source = "clip"
    logger.info(f"Generating Grad-CAM: {source} / {label_col}")

    # Load CLIP
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.to(DEVICE).eval()

    # Load auditor
    model_path = MODELS_DIR / f"deep_auditor_{source}_{label_col}.pt"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return

    auditor = DeepAuditor.load(str(model_path)).to(DEVICE).eval()

    # Text embedding
    text_emb = (
        torch.tensor(np.load(S2ORC_PATH, mmap_mode="r"), dtype=torch.float32)
        .mean(0, keepdim=True)
        .to(DEVICE)
    )

    # Wrapped model
    wrapper = CLIPVisionWrapper(clip_model, auditor, text_emb).to(DEVICE)
    wrapper.eval()

    target_layer = [clip_model.visual.transformer.resblocks[-1]]

    # Load images
    dataset_name = LABEL_DATASET[label_col]
    samples      = load_images_and_labels(dataset_name, label_col)

    if len(samples) < 2:
        logger.warning(f"Not enough images for {source}/{label_col}, skipping.")
        return

    correct_examples   = []
    incorrect_examples = []

    with EigenCAM(
        model=wrapper,
        target_layers=target_layer,
        reshape_transform=reshape_transform_vit,
    ) as cam:
        for img_pil, true_label in samples:
            if (len(correct_examples) >= N_CORRECT and
                    len(incorrect_examples) >= N_INCORRECT):
                break

            img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = wrapper(img_tensor)
            pred_label = logits.argmax(dim=1).item()
            is_correct = pred_label == true_label

            if is_correct and len(correct_examples) >= N_CORRECT:
                continue
            if not is_correct and len(incorrect_examples) >= N_INCORRECT:
                continue

            grayscale_cam = cam(input_tensor=img_tensor)[0]

            img_rgb = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
            overlay = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

            entry = {
                "original":   img_rgb,
                "overlay":    overlay,
                "true_label": true_label,
                "pred_label": pred_label,
                "correct":    is_correct,
            }

            if is_correct:
                correct_examples.append(entry)
            else:
                incorrect_examples.append(entry)

    all_examples = correct_examples + incorrect_examples
    if not all_examples:
        logger.warning(f"No examples collected for {source}/{label_col}")
        return

    logger.info(f"  Correct: {len(correct_examples)}  Incorrect: {len(incorrect_examples)}")

    label_names = LABEL_NAMES.get(label_col, {})
    n_cols      = len(all_examples)
    fig, axes   = plt.subplots(2, n_cols, figsize=(n_cols * 2.8, 6))

    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    fig.suptitle(
        f"Grad-CAM — CLIP ViT-L/14 Deep Auditor / {label_col}\n"
        f"Top: original face    Bottom: Grad-CAM overlay    "
        f"Green title = correct    Red title = incorrect",
        fontsize=10,
    )

    for col, ex in enumerate(all_examples):
        true_name = label_names.get(ex["true_label"], str(ex["true_label"]))
        pred_name = label_names.get(ex["pred_label"], str(ex["pred_label"]))
        color     = "green" if ex["correct"] else "red"

        axes[0, col].imshow(ex["original"])
        axes[0, col].axis("off")
        axes[0, col].set_title(f"True: {true_name}", fontsize=8)

        axes[1, col].imshow(ex["overlay"])
        axes[1, col].axis("off")
        axes[1, col].set_title(f"Pred: {pred_name}", fontsize=8, color=color)

    if correct_examples and incorrect_examples:
        split_x = len(correct_examples) / n_cols
        fig.text(
            split_x, 0.01,
            f"<-- Correct ({len(correct_examples)})   |   "
            f"Incorrect ({len(incorrect_examples)}) -->",
            ha="center", fontsize=9, color="gray",
        )

    plt.tight_layout()
    out_path = OUT_DIR / f"gradcam_{source}_{label_col}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    for label_col in ["gender", "race", "age"]:
        generate_gradcam_figure(label_col)
    logger.info("Done.")


if __name__ == "__main__":
    main()