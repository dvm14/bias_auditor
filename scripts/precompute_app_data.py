# This work was done with the help of Claude (https://claude.ai/share/8c1843e5-e882-409c-aae5-b0a144a43d20)
"""
scripts/precompute_app_data.py

Pre-computes all data needed for the bias auditor app.
Runs inference using cached embeddings (no GPU needed for predictions).
Generates per-image EigenCAM heatmaps for CLIP only.
Saves everything to app/static/app_data.json.

Run once before deploying:
    python scripts/precompute_app_data.py

Requirements:
    pip install grad-cam open_clip_torch torch numpy pandas pillow
"""

import sys
import json
import logging
import base64
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import DeepAuditor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
EMB_DIR    = BASE_DIR / "data" / "processed" / "embeddings"
LABEL_DIR  = BASE_DIR / "data" / "processed" / "labels"
MODELS_DIR = BASE_DIR / "models"
AUDIT_CSV  = BASE_DIR / "data" / "outputs" / "audit_results" / "audit_results.csv"
S2ORC_PATH = EMB_DIR / "s2orc_safety_sentence.npy"
OUT_DIR    = BASE_DIR / "app" / "static"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_IMAGES = 8  # images per combination

# ── label maps ────────────────────────────────────────────────────────────────
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

# Which dataset has each label_col
LABEL_DATASET = {
    "gender": "fairface",
    "race": "fairface",
    "age": "fairface",
}

# Subgroup axes available per task
TASK_AXES = {
    "gender": ["race", "age"],
    "race":   ["gender", "age"],
    "age":    ["gender", "race"],
}

# Human-readable task/axis descriptions
TASK_DESCRIPTIONS = {
    "gender": "Gender Prediction",
    "race":   "Race Prediction",
    "age":    "Age Prediction",
}

AXIS_DESCRIPTIONS = {
    "race":   "by Racial Group",
    "age":    "by Age Group",
    "gender": "by Gender",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def image_to_base64(img_pil: Image.Image, size=(224, 224)) -> str:
    img_pil = img_pil.resize(size)
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def heatmap_to_base64(overlay_np: np.ndarray) -> str:
    img = Image.fromarray(overlay_np)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_label_df(label_col: str, split: str = "test") -> pd.DataFrame:
    dataset = LABEL_DATASET[label_col]
    path = LABEL_DIR / f"{dataset}_{split}_labels.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    return df


def load_embeddings(label_col: str, source: str, split: str = "test") -> np.ndarray:
    dataset = LABEL_DATASET[label_col]
    path = EMB_DIR / f"{dataset}_{split}_{source}.npy"
    return np.load(path, mmap_mode="r")


def load_svm(source: str, label_col: str):
    path = MODELS_DIR / f"svm_{source}_{label_col}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def get_predictions_and_confidence(
    embeddings: np.ndarray,
    label_col: str,
    source: str,
    auditor_type: str = "deep",
    deep_model=None,
    svm_model=None,
    text_emb=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predictions, confidences) arrays."""
    if auditor_type == "deep" and deep_model is not None:
        X = torch.tensor(embeddings, dtype=torch.float32)
        t = text_emb.expand(len(X), -1)
        with torch.no_grad():
            logits = deep_model(X, t)
            probs  = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        return preds, confs
    elif auditor_type == "svm" and svm_model is not None:
        preds = svm_model.predict(embeddings)
        probs = svm_model.predict_proba(embeddings)
        confs = probs.max(axis=1)
        return preds, confs
    return np.array([]), np.array([])


def select_interesting_images(
    df: pd.DataFrame,
    clip_preds: np.ndarray,
    deepface_preds: np.ndarray,
    clip_confs: np.ndarray,
    deepface_confs: np.ndarray,
    label_col: str,
    subgroup_col: str,
    subgroup_val: int,
    n: int = N_IMAGES,
) -> pd.DataFrame:
    """Select a mix of correct/incorrect and clip/deepface disagreements."""
    sub = df[df[subgroup_col] == subgroup_val].copy()
    if len(sub) == 0:
        return sub

    idx = sub.index.tolist()
    true_labels = sub[label_col].values

    c_preds = clip_preds[idx]
    d_preds = deepface_preds[idx]
    c_confs = clip_confs[idx]
    d_confs = deepface_confs[idx]

    sub = sub.copy()
    sub["clip_pred"]      = c_preds
    sub["deepface_pred"]  = d_preds
    sub["clip_conf"]      = c_confs
    sub["deepface_conf"]  = d_confs
    sub["clip_correct"]   = c_preds == true_labels
    sub["deepface_correct"] = d_preds == true_labels
    sub["both_correct"]   = sub["clip_correct"] & sub["deepface_correct"]
    sub["both_wrong"]     = ~sub["clip_correct"] & ~sub["deepface_correct"]
    sub["disagree"]       = sub["clip_pred"] != sub["deepface_pred"]

    # Selection strategy: prioritise disagreements and failures
    selected = []

    # 2 cases where both wrong (most interesting failures)
    both_wrong = sub[sub["both_wrong"]].nlargest(2, "clip_conf")
    selected.append(both_wrong)

    # 2 cases where clip right, deepface wrong
    clip_wins = sub[sub["clip_correct"] & ~sub["deepface_correct"]].nlargest(2, "clip_conf")
    selected.append(clip_wins)

    # 2 cases where deepface right, clip wrong
    deepface_wins = sub[~sub["clip_correct"] & sub["deepface_correct"]].nlargest(2, "deepface_conf")
    selected.append(deepface_wins)

    # fill remaining with both correct (high confidence)
    already = pd.concat(selected).index if selected else pd.Index([])
    remaining = sub[sub["both_correct"] & ~sub.index.isin(already)]
    remaining = remaining.nlargest(n - len(already), "clip_conf")
    selected.append(remaining)

    result = pd.concat(selected).drop_duplicates().head(n)
    return result


def generate_eigencam(
    img_pil: Image.Image,
    clip_model,
    preprocess,
    wrapper,
    target_layer,
) -> np.ndarray:
    """Generate EigenCAM overlay for a single image. Returns RGB numpy array."""
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    def reshape_transform_vit(tensor, height=16, width=16):
        result = tensor[:, 1:, :]
        result = result.reshape(result.size(0), height, width, result.size(2))
        result = result.permute(0, 3, 1, 2)
        return result

    img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

    with EigenCAM(
        model=wrapper,
        target_layers=target_layer,
        reshape_transform=reshape_transform_vit,
    ) as cam:
        grayscale_cam = cam(input_tensor=img_tensor)[0]

    img_rgb = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
    return overlay


def plain_english_explanation(
    true_label: str,
    clip_pred: str,
    deepface_pred: str,
    clip_correct: bool,
    deepface_correct: bool,
    label_col: str,
) -> str:
    """Generate a one-sentence plain-English explanation per image."""
    task = label_col

    if clip_correct and deepface_correct:
        return (
            f"Both models correctly identified this person as {true_label}. "
            f"The gender signal in this face is strong and consistent."
        ) if task == "gender" else (
            f"Both models correctly predicted {true_label}. "
            f"This is a clear example the models handle well."
        )

    if clip_correct and not deepface_correct:
        return (
            f"CLIP correctly predicted {true_label} but DeepFace predicted "
            f"{deepface_pred} instead. DeepFace, trained for identity "
            f"matching rather than demographic prediction, missed the signal "
            f"that CLIP picked up."
        )

    if not clip_correct and deepface_correct:
        return (
            f"DeepFace correctly predicted {true_label} but CLIP predicted "
            f"{clip_pred} instead. Despite CLIP's stronger overall accuracy, "
            f"this face falls in a region of its embedding space where "
            f"the {task} signal is ambiguous."
        )

    if clip_pred == deepface_pred:
        return (
            f"Both models predicted {clip_pred} when the true label is "
            f"{true_label}. This is a systematic failure — both encoders "
            f"place this face in the wrong region of their embedding spaces, "
            f"suggesting the visual features for {true_label} are atypical "
            f"or underrepresented in training."
        )

    return (
        f"Both models got this wrong: CLIP predicted {clip_pred} and "
        f"DeepFace predicted {deepface_pred}, but the true label is "
        f"{true_label}. The models not only failed but disagreed with "
        f"each other, indicating high uncertainty about this face."
    )


def plain_english_metrics(
    accuracy: float,
    demographic_parity: float,
    equalized_odds: float,
    equal_opportunity: float,
    label_col: str,
    subgroup_name: str,
    subgroup_col: str,
) -> dict:
    """Return plain-English descriptions of each metric."""
    acc_pct   = round(accuracy * 100, 1)
    dp_pct    = round(demographic_parity * 100, 1)
    eo_pct    = round(equalized_odds * 100, 1)
    eop_pct   = round(equal_opportunity * 100, 1)

    task_word = {"gender": "gender", "race": "race", "age": "age group"}[label_col]
    axis_word = {"race": "racial group", "age": "age group", "gender": "gender"}[subgroup_col]

    return {
        "accuracy": {
            "value": f"{acc_pct}%",
            "label": "Accuracy",
            "explanation": (
                f"The model correctly predicted {task_word} for "
                f"{acc_pct}% of {subgroup_name} faces in the test set."
            ),
        },
        "demographic_parity": {
            "value": f"{dp_pct}%",
            "label": "Demographic Parity Gap",
            "explanation": (
                f"The model's prediction rate for {subgroup_name} differs "
                f"from other {axis_word}s by {dp_pct} percentage points. "
                f"A gap above 10% is considered a fairness warning."
            ),
        },
        "equalized_odds": {
            "value": f"{eo_pct}%",
            "label": "Equalized Odds Gap",
            "explanation": (
                f"When the true label is {subgroup_name}, the model's "
                f"error rate varies by {eo_pct} percentage points across "
                f"different {axis_word}s. Lower is fairer."
            ),
        },
        "equal_opportunity": {
            "value": f"{eop_pct}%",
            "label": "Equal Opportunity Gap",
            "explanation": (
                f"The model is {eop_pct} percentage points more likely "
                f"to miss a true {subgroup_name} face compared to the "
                f"best-served group. This measures who the model "
                f"underserves most."
            ),
        },
    }


# ── CLIP wrapper for EigenCAM ─────────────────────────────────────────────────
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


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    import open_clip

    logger.info("Loading audit metrics CSV...")
    audit_df = pd.read_csv(AUDIT_CSV)
    deep_metrics = audit_df[audit_df["auditor"] == "deep"]

    logger.info("Loading S2ORC text embedding...")
    text_emb_np = np.load(S2ORC_PATH, mmap_mode="r")
    text_emb = (
        torch.tensor(text_emb_np, dtype=torch.float32)
        .mean(0, keepdim=True)
        .to(DEVICE)
    )

    logger.info("Loading CLIP model...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.to(DEVICE).eval()

    app_data = {
        "tasks": {},
        "label_names": LABEL_NAMES,
        "task_descriptions": TASK_DESCRIPTIONS,
        "axis_descriptions": AXIS_DESCRIPTIONS,
    }

    for label_col in ["gender", "race", "age"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing task: {label_col}")

        # Load deep auditors for both sources
        clip_deep = DeepAuditor.load(
            str(MODELS_DIR / f"deep_auditor_clip_{label_col}.pt")
        ).to(DEVICE).eval()

        deepface_deep = DeepAuditor.load(
            str(MODELS_DIR / f"deep_auditor_deepface_{label_col}.pt")
        ).to(DEVICE).eval()

        # CLIP wrapper for EigenCAM
        wrapper = CLIPVisionWrapper(clip_model, clip_deep, text_emb).to(DEVICE)
        wrapper.eval()
        target_layer = [clip_model.visual.transformer.resblocks[-1]]

        # Load label df and embeddings
        label_df = load_label_df(label_col)
        clip_emb     = load_embeddings(label_col, "clip")
        deepface_emb = load_embeddings(label_col, "deepface")

        # Align lengths
        n = min(len(label_df), len(clip_emb), len(deepface_emb))
        label_df     = label_df.iloc[:n].reset_index(drop=True)
        clip_emb     = clip_emb[:n]
        deepface_emb = deepface_emb[:n]

        # Get predictions for all images
        logger.info("  Running CLIP predictions...")
        clip_preds, clip_confs = get_predictions_and_confidence(
            clip_emb, label_col, "clip",
            auditor_type="deep", deep_model=clip_deep,
            text_emb=text_emb.expand(n, -1),
        )

        logger.info("  Running DeepFace predictions...")
        deepface_preds, deepface_confs = get_predictions_and_confidence(
            deepface_emb, label_col, "deepface",
            auditor_type="deep", deep_model=deepface_deep,
            text_emb=text_emb.expand(n, -1),
        )

        app_data["tasks"][label_col] = {"axes": {}}

        for subgroup_col in TASK_AXES[label_col]:
            logger.info(f"  Subgroup axis: {subgroup_col}")
            app_data["tasks"][label_col]["axes"][subgroup_col] = {"subgroups": {}}

            subgroup_vals = sorted(
                label_df[subgroup_col].dropna().unique().astype(int)
            ) if subgroup_col in label_df.columns else []

            for subgroup_val in subgroup_vals:
                subgroup_name = LABEL_NAMES[subgroup_col].get(
                    subgroup_val, str(subgroup_val)
                )
                logger.info(f"    Subgroup: {subgroup_name} ({subgroup_val})")

                # Get metrics from audit CSV
                metrics_row = deep_metrics[
                    (deep_metrics["source"] == "clip") &
                    (deep_metrics["label_col"] == label_col) &
                    (deep_metrics["subgroup_col"] == subgroup_col) &
                    (deep_metrics["subgroup_val"] == subgroup_val)
                ]

                if metrics_row.empty:
                    logger.warning(f"    No metrics found, skipping.")
                    continue

                m = metrics_row.iloc[0]
                metrics = plain_english_metrics(
                    accuracy=m["accuracy"],
                    demographic_parity=float(m["demographic_parity"] or 0),
                    equalized_odds=float(m["equalized_odds"] or 0),
                    equal_opportunity=float(m["equal_opportunity"] or 0),
                    label_col=label_col,
                    subgroup_name=subgroup_name,
                    subgroup_col=subgroup_col,
                )

                # Select interesting images
                selected = select_interesting_images(
                    label_df, clip_preds, deepface_preds,
                    clip_confs, deepface_confs,
                    label_col, subgroup_col, subgroup_val,
                )

                if len(selected) == 0:
                    logger.warning(f"    No images found for this subgroup.")
                    continue

                images_data = []
                for _, row in selected.iterrows():
                    img_path = BASE_DIR / row["image_path"]
                    if not img_path.exists():
                        continue

                    try:
                        img_pil = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        logger.warning(f"    Could not open {img_path}: {e}")
                        continue

                    true_label     = int(row[label_col])
                    clip_pred      = int(row["clip_pred"])
                    deepface_pred  = int(row["deepface_pred"])
                    clip_correct   = bool(row["clip_correct"])
                    deepface_correct = bool(row["deepface_correct"])

                    true_name     = LABEL_NAMES[label_col].get(true_label, str(true_label))
                    clip_name     = LABEL_NAMES[label_col].get(clip_pred, str(clip_pred))
                    deepface_name = LABEL_NAMES[label_col].get(deepface_pred, str(deepface_pred))

                    # Generate EigenCAM
                    try:
                        overlay = generate_eigencam(
                            img_pil, clip_model, preprocess,
                            wrapper, target_layer
                        )
                        heatmap_b64 = heatmap_to_base64(overlay)
                    except Exception as e:
                        logger.warning(f"    EigenCAM failed: {e}")
                        heatmap_b64 = None

                    explanation = plain_english_explanation(
                        true_name, clip_name, deepface_name,
                        clip_correct, deepface_correct, label_col,
                    )

                    images_data.append({
                        "image_b64":      image_to_base64(img_pil),
                        "heatmap_b64":    heatmap_b64,
                        "true_label":     true_name,
                        "clip_pred":      clip_name,
                        "clip_conf":      round(float(row["clip_conf"]) * 100, 1),
                        "clip_correct":   clip_correct,
                        "deepface_pred":  deepface_name,
                        "deepface_conf":  round(float(row["deepface_conf"]) * 100, 1),
                        "deepface_correct": deepface_correct,
                        "explanation":    explanation,
                    })

                app_data["tasks"][label_col]["axes"][subgroup_col]["subgroups"][str(subgroup_val)] = {
                    "name":    subgroup_name,
                    "metrics": metrics,
                    "images":  images_data,
                }
                logger.info(f"    Saved {len(images_data)} images.")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = OUT_DIR / "app_data.json"
    with open(out_path, "w") as f:
        json.dump(app_data, f, cls=NumpyEncoder)

    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"\nSaved app_data.json ({size_mb:.1f} MB) → {out_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()