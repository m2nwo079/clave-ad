"""Label-free image listing and loading. Paths are relative to each dataset root."""

import csv
from pathlib import Path

from PIL import Image


def mvtec_categories(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if (p / "train" / "good").is_dir())


def mvtec_train_normals(root: Path, category: str) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in (root / category / "train" / "good").glob("*.png"))


def visa_rows(split_csv: Path) -> list[dict]:
    with open(split_csv, newline="") as f:
        return list(csv.DictReader(f))


def visa_train_normals(split_csv: Path, category: str) -> list[str]:
    # the 1cls train split holds normal images only
    return sorted(r["image"] for r in visa_rows(split_csv) if r["object"] == category and r["split"] == "train")


def load_image(root: Path, rel_path: str) -> Image.Image:
    return Image.open(root / rel_path).convert("RGB")
