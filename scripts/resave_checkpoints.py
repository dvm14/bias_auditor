# This work was done with the help of Claude (https://claude.ai/share/8c1843e5-e882-409c-aae5-b0a144a43d20)
import torch
from model import DeepAuditor

SOURCES = {"clip": 768, "deepface": 512}
LABELS  = {"gender": 2, "race": 7, "age": 9}

for src, vdim in SOURCES.items():
    for label, nc in LABELS.items():
        path = f"models/deep_auditor_{src}_{label}.pt"
        try:
            ckpt = torch.load(path, map_location="cpu")
            ckpt["num_classes"] = nc
            torch.save(ckpt, path)
            print(f"Re-saved {path}")
        except FileNotFoundError:
            print(f"Skipping {path} (not found)")