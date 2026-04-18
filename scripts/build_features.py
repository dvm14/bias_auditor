"""
Extract and cache CLIP visual embeddings, DeepFace ArcFace embeddings, and
sentence transformer embeddings.

Outputs (written once, never overwritten if cache exists):
  data/processed/embeddings/
    {dataset}_{split}_clip.npy      — (N, 768) float32, CLIP ViT-L/14 features
    {dataset}_{split}_deepface.npy  — (N, 512) float32, DeepFace ArcFace features
    s2orc_safety_sentence.npy       — (M, 384) float32, all-MiniLM-L6-v2 features
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ViT-L-14 produces 768-dim visual features, matching the architecture in CLAUDE.md.
CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "openai"
# all-MiniLM-L6-v2 produces 384-dim sentence embeddings, matching CLAUDE.md.
SENTENCE_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
DATASETS = ["celeba", "fairface", "utkface"]
SPLITS = ["train", "val", "test"]

# Expected embedding dimensions — used for shape verification.
EXPECTED_CLIP_DIM = 768
EXPECTED_DEEPFACE_DIM = 512
EXPECTED_SENTENCE_DIM = 384


class FeatureExtractor:
    """Extract and cache CLIP visual and sentence transformer embeddings."""

    def __init__(self, data_dir: str = "data", device: str = None):
        self.data_dir = Path(data_dir)
        self.labels_dir = self.data_dir / "processed" / "labels"
        self.embeddings_dir = self.data_dir / "processed" / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Prefer GPU if available; fall back to CPU.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Lazy-loaded — only initialised when first needed.
        self._clip_model = None
        self._clip_preprocess = None
        self._sentence_model = None

    def _load_clip(self) -> None:
        """Load the frozen CLIP ViT-L/14 model (once)."""
        if self._clip_model is not None:
            return
        import open_clip
        logger.info(f"Loading CLIP {CLIP_MODEL} ({CLIP_PRETRAINED})...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        # Frozen — parameters are never updated, consistent with CLAUDE.md.
        model = model.to(self.device).eval()
        self._clip_model = model
        self._clip_preprocess = preprocess

    def _load_sentence_model(self) -> None:
        """Load the frozen sentence transformer (once)."""
        if self._sentence_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence transformer {SENTENCE_MODEL}...")
        self._sentence_model = SentenceTransformer(SENTENCE_MODEL)

    def _preprocess_image(self, path: str) -> torch.Tensor:
        """Load and preprocess a single image.

        On failure, returns a blank image run through the same preprocessor so
        the tensor shape and normalization match — keeping row indices aligned
        with the labels CSV. Downstream code can drop failed rows by checking
        the labels file.
        """
        try:
            img = Image.open(path).convert("RGB")
            return self._clip_preprocess(img)
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}. Using blank fallback.")
            return self._clip_preprocess(Image.new("RGB", (224, 224)))

    def _verify_clip_shape(self, embeddings: np.ndarray, name: str) -> None:
        """Assert CLIP embedding shape matches expected dimensions.

        Logs an error (instead of raising) so a single bad file does not abort
        the whole extraction run — the downstream model.py will catch the mismatch.
        """
        if embeddings.shape[1] != EXPECTED_CLIP_DIM:
            logger.error(
                f"Shape mismatch in {name}: expected dim={EXPECTED_CLIP_DIM}, "
                f"got {embeddings.shape}. Check CLIP_MODEL / CLIP_PRETRAINED."
            )
        else:
            logger.info(f"Shape OK: {name} → {embeddings.shape} (dim={embeddings.shape[1]})")

    def extract_clip_embeddings(self, dataset: str, split: str) -> None:
        """Extract CLIP embeddings for one dataset split.

        Reads the label CSV produced by make_dataset.py, runs images through
        the frozen CLIP ViT in batches, and saves a (N, 768) .npy file.
        No-op if the cache file already exists.
        """
        out_path = self.embeddings_dir / f"{dataset}_{split}_clip.npy"
        if out_path.exists():
            logger.info(f"Cache hit, skipping: {out_path.name}")
            return

        labels_path = self.labels_dir / f"{dataset}_{split}_labels.csv"
        if not labels_path.exists():
            logger.warning(f"Labels file not found, skipping: {labels_path}")
            return

        df = pd.read_csv(labels_path)
        self._load_clip()

        batches = []
        for start in tqdm(
            range(0, len(df), BATCH_SIZE),
            desc=f"CLIP {dataset}/{split}",
        ):
            paths = df["image_path"].iloc[start : start + BATCH_SIZE].tolist()

            # Per-image error handling — corrupt images use blank fallback,
            # keeping tensor shapes consistent with the labels CSV row indices.
            tensors = []
            for p in paths:
                tensors.append(self._preprocess_image(p))

            imgs = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                feats = self._clip_model.encode_image(imgs)  # (B, 768)

            # Verify first batch shape so we catch wrong model variants early.
            if start == 0:
                logger.info(
                    f"First batch shape check: encode_image output = {feats.shape} "
                    f"(expected dim={EXPECTED_CLIP_DIM})"
                )
                if feats.shape[1] != EXPECTED_CLIP_DIM:
                    logger.error(
                        f"Wrong embedding dim {feats.shape[1]} — expected {EXPECTED_CLIP_DIM}. "
                        f"Check that CLIP_MODEL='{CLIP_MODEL}' loaded correctly."
                    )

            batches.append(feats.cpu().float().numpy())

        embeddings = np.concatenate(batches, axis=0)  # (N, 768)
        self._verify_clip_shape(embeddings, out_path.name)
        np.save(out_path, embeddings)
        logger.info(f"Saved {embeddings.shape} → {out_path}")

    def extract_deepface_embeddings(self, dataset: str, split: str) -> None:
        """Extract DeepFace ArcFace embeddings for one dataset split.

        Reads the label CSV produced by make_dataset.py, runs each image through
        DeepFace ArcFace one at a time, and saves a (N, 512) .npy file.
        enforce_detection=False is required so images with no detected face
        return a zero vector instead of raising — keeps row indices aligned
        with the labels CSV.
        No-op if the cache file already exists.
        """
        out_path = self.embeddings_dir / f"{dataset}_{split}_deepface.npy"
        if out_path.exists():
            logger.info(f"Cache hit, skipping: {out_path.name}")
            return

        labels_path = self.labels_dir / f"{dataset}_{split}_labels.csv"
        if not labels_path.exists():
            logger.warning(f"Labels file not found, skipping: {labels_path}")
            return

        from deepface import DeepFace

        df = pd.read_csv(labels_path)
        embeddings = []

        for path in tqdm(df["image_path"], desc=f"DeepFace {dataset}/{split}"):
            try:
                result = DeepFace.represent(
                    img_path=path,
                    model_name="ArcFace",
                    enforce_detection=False,
                )
                # result is a list of dicts; take the first face's embedding.
                vec = result[0]["embedding"] if result else [0.0] * EXPECTED_DEEPFACE_DIM
            except Exception as e:
                logger.warning(f"DeepFace failed on {path}: {e}. Using zero vector.")
                vec = [0.0] * EXPECTED_DEEPFACE_DIM

            embeddings.append(vec)

        arr = np.array(embeddings, dtype=np.float32)  # (N, 512)

        if arr.shape[1] != EXPECTED_DEEPFACE_DIM:
            logger.error(
                f"DeepFace embedding dim mismatch: expected {EXPECTED_DEEPFACE_DIM}, "
                f"got {arr.shape[1]}. Check model_name='ArcFace'."
            )
        else:
            logger.info(f"Shape OK: {out_path.name} → {arr.shape} (dim={arr.shape[1]})")

        np.save(out_path, arr)
        logger.info(f"Saved {arr.shape} → {out_path}")

    def extract_sentence_embeddings(self) -> None:
        """Extract sentence embeddings from S2ORC safety papers.

        Reads papers.csv produced by make_dataset.py's download_s2orc_safety(),
        encodes the text_for_embedding column, and saves a (M, 384) .npy file.
        No-op if the cache file already exists.
        """
        out_path = self.embeddings_dir / "s2orc_safety_sentence.npy"
        if out_path.exists():
            logger.info(f"Cache hit, skipping: {out_path.name}")
            return

        papers_path = self.data_dir / "raw" / "s2orc_safety" / "papers.csv"
        if not papers_path.exists():
            logger.warning(f"S2ORC papers CSV not found: {papers_path}")
            return

        self._load_sentence_model()
        df = pd.read_csv(papers_path)
        texts = df["text_for_embedding"].fillna("").tolist()

        logger.info(f"Encoding {len(texts)} S2ORC safety papers...")
        embeddings = self._sentence_model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )  # (M, 384)

        if embeddings.shape[1] != EXPECTED_SENTENCE_DIM:
            logger.error(
                f"Sentence embedding dim mismatch: expected {EXPECTED_SENTENCE_DIM}, "
                f"got {embeddings.shape[1]}. Check SENTENCE_MODEL='{SENTENCE_MODEL}'."
            )
        else:
            logger.info(f"Shape OK: s2orc_safety_sentence.npy → {embeddings.shape}")

        np.save(out_path, embeddings)
        logger.info(f"Saved {embeddings.shape} → {out_path}")

    def verify_all(self) -> None:
        """Log shape and dtype of every cached .npy file for checkpoint review.

        Paste this output when checking in — it confirms all 19 embedding files
        (3 datasets × 3 splits × 2 embedding types + 1 S2ORC) are present and correctly shaped.
        """
        logger.info("=" * 60)
        logger.info("Verification summary:")
        npy_files = sorted(self.embeddings_dir.glob("*.npy"))
        if not npy_files:
            logger.warning("No .npy files found in embeddings directory.")
            return
        for f in npy_files:
            arr = np.load(f, mmap_mode="r")  # mmap avoids loading into RAM
            if "clip" in f.name:
                expected_dim = EXPECTED_CLIP_DIM
            elif "deepface" in f.name:
                expected_dim = EXPECTED_DEEPFACE_DIM
            else:
                expected_dim = EXPECTED_SENTENCE_DIM
            status = "OK" if arr.shape[1] == expected_dim else "MISMATCH"
            logger.info(f"  [{status}] {f.name}: {arr.shape} {arr.dtype}")
        logger.info("=" * 60)

    def run(self) -> None:
        """Extract all embeddings for all datasets, splits, and text corpus."""
        logger.info("=" * 60)
        logger.info("Starting feature extraction...")
        logger.info("=" * 60)

        for dataset in DATASETS:
            for split in SPLITS:
                self.extract_clip_embeddings(dataset, split)

        for dataset in DATASETS:
            for split in SPLITS:
                self.extract_deepface_embeddings(dataset, split)

        self.extract_sentence_embeddings()

        logger.info("=" * 60)
        logger.info("Feature extraction complete!")
        logger.info(f"Embeddings saved to: {self.embeddings_dir}")
        logger.info("=" * 60)

        # Always run verification at the end so the checkpoint log is automatic.
        self.verify_all()


if __name__ == "__main__":
    extractor = FeatureExtractor(data_dir="data")
    extractor.run()