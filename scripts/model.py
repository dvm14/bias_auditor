"""
Three-tier bias auditor models.

NaiveAuditor  — majority-class baseline; no learning.
ClassicalAuditor — SVM on CLIP visual embeddings + SHAP explainability.
DeepAuditor   — late-fusion deep model:
                  vision tower:  (N, 768) → projection MLP → (N, 256)
                  text tower:    (N, 384) → projection MLP → (N, 256)
                  fusion:        concat [v ; t] → (N, 512) → classifier head
                Only the projection MLPs and classifier head are trained;
                encoders are frozen (embeddings come pre-extracted from build_features.py).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NaiveAuditor
# ---------------------------------------------------------------------------

class NaiveAuditor:
    """Majority-class baseline auditor.

    Always predicts the most frequent class in the training set.
    Used to establish a floor for accuracy and fairness metrics — any
    auditor that can't beat this is not learning anything useful.
    """

    def __init__(self):
        self.majority_class_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveAuditor":
        classes, counts = np.unique(y, return_counts=True)
        self.majority_class_ = int(classes[np.argmax(counts)])
        logger.info(f"NaiveAuditor: majority class = {self.majority_class_} "
                    f"({counts.max()}/{len(y)} = {counts.max()/len(y):.1%})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.majority_class_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return uniform probability over classes (2-class assumed)."""
        if self.majority_class_ is None:
            raise RuntimeError("Call fit() before predict().")
        n = len(X)
        proba = np.zeros((n, 2), dtype=float)
        proba[:, self.majority_class_] = 1.0
        return proba


# ---------------------------------------------------------------------------
# ClassicalAuditor
# ---------------------------------------------------------------------------

class ClassicalAuditor:
    """SVM-based fairness auditor with SHAP feature importance.

    Input: pre-extracted CLIP visual embeddings (N, 768).
    Pipeline: StandardScaler → SVC(probability=True, kernel='rbf').

    SHAP is computed lazily in explain() and never called during training,
    so the llvmlite/numba dependency is never triggered.
    """

    def __init__(self, C: float = 1.0, kernel: str = "rbf"):
        # kernel ignored when using LinearSVC — kept for API compatibility
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", CalibratedClassifierCV(LinearSVC(C=C, max_iter=2000))),
        ])
        self._shap_explainer = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalAuditor":
        logger.info(f"ClassicalAuditor: fitting SVM on {X.shape} embeddings...")
        self.pipeline.fit(X, y)
        logger.info("ClassicalAuditor: training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def explain(self, X_background: np.ndarray, X_explain: np.ndarray) -> np.ndarray:
        """Compute SHAP values for X_explain using LinearExplainer.

        Args:
            X_background: representative background sample (e.g. 100 train rows).
            X_explain:    samples to explain.

        Returns:
            shap_values: (N, 768) array of feature importances.
        """
        import shap

        if self._shap_explainer is None:
            X_bg_scaled = self.pipeline.named_steps["scaler"].transform(X_background)
            # Use the full calibrated pipeline for predict_proba
            self._shap_explainer = shap.KernelExplainer(
                self.pipeline.named_steps["svm"].predict_proba, X_bg_scaled
            )

        X_scaled = self.pipeline.named_steps["scaler"].transform(X_explain)
        shap_values = self._shap_explainer.shap_values(X_scaled)
        return shap_values


# ---------------------------------------------------------------------------
# DeepAuditor — projection MLPs + classifier head
# ---------------------------------------------------------------------------

class _ProjectionMLP(nn.Module):
    """Two-layer projection MLP with LayerNorm and GELU activation.

    Projects frozen encoder outputs to a shared 256-dim space before fusion.
    """

    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepAuditor(nn.Module):
    """Late-fusion bias auditor.

    Vision tower:  (B, vision_dim) features → projection MLP → (B, 256)
    Text tower:    (B, 384) sentence features → projection MLP → (B, 256)
    Fusion:        concat → (B, 512) → classifier head → (B, num_classes)

    Only the projection MLPs and classifier head have trainable parameters.
    The frozen encoders are NOT part of this module — embeddings are passed
    in directly (pre-extracted by build_features.py).

    vision_dim is a constructor argument so the same class works for both
    CLIP (768-dim) and DeepFace (512-dim) inputs.
    """

    TEXT_DIM = 384   # all-MiniLM-L6-v2 output
    PROJ_DIM = 256   # projection output (both towers)

    def __init__(self, vision_dim: int = 768, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.vision_dim = vision_dim
        self.num_classes = num_classes

        self.vision_proj = _ProjectionMLP(
            in_dim=vision_dim, out_dim=self.PROJ_DIM
        )
        self.text_proj = _ProjectionMLP(
            in_dim=self.TEXT_DIM, out_dim=self.PROJ_DIM
        )

        # Classifier head: 256 + 256 = 512 fused dim → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.PROJ_DIM * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_features: (B, vision_dim) pre-extracted visual embeddings.
            text_features:   (B, 384) pre-extracted sentence embeddings.
                             For a batch of images, pass the mean S2ORC
                             embedding (1, 384) broadcast across the batch,
                             or per-sample text embeddings if available.

        Returns:
            logits: (B, num_classes)
        """
        v = self.vision_proj(vision_features)   # (B, 256)
        t = self.text_proj(text_features)        # (B, 256)
        fused = torch.cat([v, t], dim=-1)        # (B, 512)
        return self.classifier(fused)            # (B, num_classes)

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "vision_dim": self.vision_dim,
            "num_classes": self.num_classes,
        }, path)

    @classmethod
    def load(cls, path: str, num_classes: int = 2, **kwargs) -> "DeepAuditor":
        checkpoint = torch.load(path, map_location="cpu")
        vision_dim = checkpoint["vision_dim"]
        num_classes = checkpoint.get("num_classes", 2)
        model = cls(vision_dim=vision_dim, num_classes=num_classes, **kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
