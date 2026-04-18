"""
Download and prepare datasets from HuggingFace for bias auditing.

Handles:
- CelebA: gender attribute
- FairFace: gender, race, age
- UTKFace: gender, race, age
"""

import os
import logging
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Download and split datasets into train/val/test."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "labels").mkdir(exist_ok=True)

        self.splits = {"train": 0.7, "val": 0.15, "test": 0.15}

    @staticmethod
    def _bin_age(age: int) -> int:
        """Bin a raw age integer into FairFace-compatible groups (0–8)."""
        if age <= 2:   return 0
        if age <= 9:   return 1
        if age <= 19:  return 2
        if age <= 29:  return 3
        if age <= 39:  return 4
        if age <= 49:  return 5
        if age <= 59:  return 6
        if age <= 69:  return 7
        return 8

    def download_celeba(self) -> pd.DataFrame:
        """Download CelebA dataset from HuggingFace."""
        logger.info("Downloading CelebA...")
        try:
            dataset = load_dataset(
                "flwrlabs/celeba",
                split="train",
                streaming=False,
                token=os.environ.get("HF_TOKEN"),
            )
            df_list = []

            for idx, sample in enumerate(tqdm(dataset, desc="Processing CelebA")):
                if "image" in sample:
                    # Male is boolean; int(True)=1=Male, int(False)=0=Female → 0=Female, 1=Male
                    gender = int(sample.get("Male", False))
                    img = sample["image"]
                    img_path = self.raw_dir / "celeba" / f"celeba_{idx:06d}.jpg"
                    img_path.parent.mkdir(parents=True, exist_ok=True)

                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(img_path)
                    elif isinstance(img, bytes):
                        Image.open(io.BytesIO(img)).convert("RGB").save(img_path)

                    df_list.append({
                        "dataset": "celeba",
                        "image_id": f"celeba_{idx:06d}",
                        "image_path": str(img_path),
                        "gender": gender,  # 0=Female, 1=Male
                    })

                if idx >= 9999:  # Limit for demo (10 000 samples: indices 0–9999)
                    break

            df = pd.DataFrame(df_list)
            logger.info(f"Downloaded {len(df)} CelebA samples")
            return df

        except Exception as e:
            logger.error(f"Error downloading CelebA: {e}")
            return pd.DataFrame()

    def download_fairface(self) -> pd.DataFrame:
        """Download FairFace dataset from HuggingFace."""
        logger.info("Downloading FairFace...")
        try:
            dataset = load_dataset(
                "HuggingFaceM4/FairFace",
                "0.25",  # padding=0.25 version; required config name
                split="train",
                streaming=False,
                token=os.environ.get("HF_TOKEN"),
            )
            df_list = []

            for idx, sample in enumerate(tqdm(dataset, desc="Processing FairFace")):
                if "image" in sample:
                    # gender: 0=Male, 1=Female in source — flip to 0=Female, 1=Male
                    raw_gender = sample.get("gender", -1)
                    gender = (1 - raw_gender) if raw_gender >= 0 else -1
                    age = sample.get("age", -1)    # already 0–8 bins
                    race = sample.get("race", -1)  # 0–6 integer

                    if gender >= 0 and age >= 0 and race >= 0:
                        img = sample["image"]
                        img_path = self.raw_dir / "fairface" / f"fairface_{idx:06d}.jpg"
                        img_path.parent.mkdir(parents=True, exist_ok=True)

                        if isinstance(img, Image.Image):
                            img.convert("RGB").save(img_path)
                        elif isinstance(img, bytes):
                            Image.open(io.BytesIO(img)).convert("RGB").save(img_path)

                        df_list.append({
                            "dataset": "fairface",
                            "image_id": f"fairface_{idx:06d}",
                            "image_path": str(img_path),
                            "gender": gender,  # 0=Female, 1=Male
                            "race": race,      # 0=East Asian,1=Indian,2=Black,3=White,4=Middle Eastern,5=Latino,6=SE Asian
                            "age": age,        # 0=0-2, 1=3-9, ..., 8=70+
                        })

                if idx >= 9999:  # Limit for demo (10 000 samples: indices 0–9999)
                    break

            df = pd.DataFrame(df_list)
            logger.info(f"Downloaded {len(df)} FairFace samples")
            return df

        except Exception as e:
            logger.error(f"Error downloading FairFace: {e}")
            return pd.DataFrame()

    def download_utkface(self) -> pd.DataFrame:
        """Download UTKFace dataset from HuggingFace."""
        logger.info("Downloading UTKFace...")
        try:
            dataset = load_dataset(
                "Subh775/UTKFace_demographics_V1",
                split="train",
                streaming=False,
                token=os.environ.get("HF_TOKEN"),
            )
            df_list = []

            for idx, sample in enumerate(tqdm(dataset, desc="Processing UTKFace")):
                if "image" in sample:
                    # Raw encoding: 0=Male, 1=Female — flip to match CelebA/FairFace (0=Female, 1=Male)
                    raw_age = sample.get("age", -1)
                    raw_gender = sample.get("gender", -1)
                    gender = (1 - raw_gender) if raw_gender >= 0 else -1  # 0=Female, 1=Male
                    race = sample.get("race", -1)  # 0=White,1=Black,2=Asian,3=Indian,4=Others
                    age = self._bin_age(raw_age) if raw_age >= 0 else -1  # bin to 0–8 groups

                    if age >= 0 and gender >= 0 and race >= 0:
                        img = sample["image"]
                        img_path = self.raw_dir / "utkface" / f"utkface_{idx:06d}.jpg"
                        img_path.parent.mkdir(parents=True, exist_ok=True)

                        if isinstance(img, Image.Image):
                            img.convert("RGB").save(img_path)
                        elif isinstance(img, bytes):
                            Image.open(io.BytesIO(img)).convert("RGB").save(img_path)

                        df_list.append({
                            "dataset": "utkface",
                            "image_id": f"utkface_{idx:06d}",
                            "image_path": str(img_path),
                            "age": age,       # 0–8 bins (same as FairFace)
                            "gender": gender,  # 0=Female, 1=Male
                            "race": race,      # 0=White,1=Black,2=Asian,3=Indian,4=Others
                        })

                if idx >= 9999:  # Limit for demo (10 000 samples: indices 0–9999)
                    break

            df = pd.DataFrame(df_list)
            logger.info(f"Downloaded {len(df)} UTKFace samples")
            return df

        except Exception as e:
            logger.error(f"Error downloading UTKFace: {e}")
            return pd.DataFrame()

    @staticmethod
    def _pick_stratify(df: pd.DataFrame, dataset_name: str, label: str) -> pd.Series:
        """Return the finest stratification key where every stratum has >= 2 members.

        Tries gender+race+age, then gender+race, then gender, then None.
        Evaluated independently per split so that rare strata in a smaller
        subset don't cause failures.
        """
        all_cols = [c for c in ["gender", "race", "age"] if c in df.columns]
        candidates = [
            all_cols,
            [c for c in ["gender", "race"] if c in df.columns],
            [c for c in ["gender"] if c in df.columns],
        ]
        for cols in candidates:
            if not cols:
                continue
            key = df[cols].astype(str).agg("_".join, axis=1)
            if key.value_counts().min() >= 2:
                logger.info(f"Stratifying {dataset_name} {label} on: {cols}")
                return key
            logger.warning(
                f"Rare strata with cols={cols} in {dataset_name} {label} — trying coarser key."
            )
        logger.warning(f"Stratification disabled for {dataset_name} {label}.")
        return None

    def split_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, pd.DataFrame]:
        """Split dataset into train/val/test stratified by all available demographics."""
        logger.info(f"Splitting {dataset_name} into train/val/test...")

        # Train/test split (70/30) — pick key based on full df
        train, temp = train_test_split(
            df,
            test_size=0.3,
            random_state=42,
            stratify=self._pick_stratify(df, dataset_name, "full"),
        )

        # Val/test split (50/50 of remaining 30%) — re-pick key based on temp
        val, test = train_test_split(
            temp,
            test_size=0.5,
            random_state=42,
            stratify=self._pick_stratify(temp, dataset_name, "temp"),
        )

        return {"train": train, "val": val, "test": test}

    def save_splits(self, splits: Dict[str, pd.DataFrame], dataset_name: str) -> None:
        """Save train/val/test splits to CSV."""
        for split_name, df in splits.items():
            output_path = (
                self.processed_dir / "labels" / f"{dataset_name}_{split_name}_labels.csv"
            )
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} {split_name} samples: {output_path}")

    def download_s2orc_safety(self) -> None:
        """Download S2ORC safety corpus for ethics/safety paper embeddings."""
        logger.info("Downloading S2ORC safety corpus...")
        safety_dir = self.raw_dir / "s2orc_safety"
        safety_dir.mkdir(parents=True, exist_ok=True)

        output_path = safety_dir / "papers.csv"
        if output_path.exists():
            logger.info(f"S2ORC safety corpus already exists, skipping: {output_path}")
            return

        try:
            dataset = load_dataset(
                "AlgorithmicResearchGroup/s2orc-safety",
                split="train",
                streaming=False,
                token=os.environ.get("HF_TOKEN"),
            )

            records = []
            for sample in tqdm(dataset, desc="Processing S2ORC safety"):
                abstract = (sample.get("abstract") or "").strip()
                title = (sample.get("parsed_title") or "").strip()
                summary = (sample.get("summary") or "").strip()

                # Skip papers with no usable text
                if not abstract and not summary:
                    continue

                records.append({
                    "corpus_id": sample.get("corpus_id"),
                    "title": title,
                    "abstract": abstract,
                    "summary": summary,
                    # Combined field used by build_features.py for sentence embeddings
                    "text_for_embedding": f"{title}. {abstract}" if abstract else f"{title}. {summary}",
                })

            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} S2ORC safety papers: {output_path}")

        except Exception as e:
            logger.error(f"Error downloading S2ORC safety corpus: {e}")

    def run(self) -> None:
        """Download all datasets and create train/val/test splits."""
        logger.info("=" * 60)
        logger.info("Starting dataset download and preparation...")
        logger.info("=" * 60)

        # Download datasets
        dfs = {}
        dfs["celeba"] = self.download_celeba()
        dfs["fairface"] = self.download_fairface()
        dfs["utkface"] = self.download_utkface()
        self.download_s2orc_safety()

        # Split and save
        for dataset_name, df in dfs.items():
            if len(df) > 0:
                splits = self.split_dataset(df, dataset_name)
                self.save_splits(splits, dataset_name)

        logger.info("=" * 60)
        logger.info("Dataset preparation complete!")
        logger.info(f"Raw data: {self.raw_dir}")
        logger.info(f"Labels: {self.processed_dir / 'labels'}")
        logger.info("=" * 60)


if __name__ == "__main__":
    builder = DatasetBuilder(data_dir="data")
    builder.run()
