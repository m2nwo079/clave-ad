"""Format check sample and B1 reference selection (design doc 5, 13.3).

This module may read test labels and masks to exclude combined and Other images. It outputs image path lists
only; the operating path reads these lists, never the labels.
"""

import random
from pathlib import Path

from PIL import Image

from clave_ad.data.datasets import mvtec_categories, mvtec_train_normals, visa_rows, visa_train_normals


def mvtec_test_defects(root: Path, category: str, excluded: list[str]) -> list[str]:
    test = root / category / "test"
    return sorted(str(p.relative_to(root)) for d in sorted(test.iterdir())
                  if d.is_dir() and d.name != "good" and d.name not in excluded for p in d.glob("*.png"))


def mvtec_test_labels(root: Path, category: str) -> set[str]:
    return {d.name for d in (root / category / "test").iterdir() if d.is_dir() and d.name != "good"}


class MaskValueError(ValueError):
    """Mask pixel values are not class ids of id2class.py; carries only the kinds of unexpected values."""


def visa_test_defects(root: Path, split_csv: Path, category: str, num_classes: int, excluded_id: int | None) -> list[str]:
    """Anomaly test images of the 1cls split, without images whose original mask contains the excluded class id.

    Every mask must hold single-channel class ids 0..num_classes-1 (0 is background); otherwise MaskValueError is
    raised before any sample is drawn. Value counts are never returned or recorded.
    """
    keep, unexpected, seen = [], set(), set()
    for r in visa_rows(split_csv):
        if r["object"] != category or r["split"] != "test" or r["label"] != "anomaly":
            continue
        mask = Image.open(root / r["mask"])
        colors = mask.getcolors(maxcolors=256) if mask.mode in ("L", "P") else None
        if colors is None:
            unexpected.add(f"mode {mask.mode}")
            continue
        values = {value for _, value in colors}
        seen |= values
        unexpected |= {str(v) for v in values if not 0 <= v < num_classes}
        if excluded_id is None or excluded_id not in values:
            keep.append(r["image"])
    if unexpected:
        raise MaskValueError(f"{category}: mask values outside class ids 0..{num_classes - 1}: {sorted(unexpected)}")
    # with two or more candidate types, masks holding only 0 and 1 are probably binary, not class ids
    candidates = num_classes - 1 - (excluded_id is not None)
    if seen <= {0, 1} and candidates >= 2:
        raise MaskValueError(f"{category}: masks look binary (values 0 and 1 only)")
    return sorted(keep)


def draw(seed: int, pools: dict[str, list[str]], exclude: set[str] = frozenset()) -> dict[str, str]:
    """One path per category, categories in sorted order, from a single seeded generator."""
    rng = random.Random(seed)
    return {c: rng.choice([p for p in pools[c] if p not in exclude]) for c in sorted(pools)}


def format_check_sample(cfg: dict, types: dict) -> list[dict]:
    """Per category: one train normal, one test defect (format check seed), one B1 reference (reference seed)."""
    data, seeds = cfg["data"], cfg["sampling"]
    mvtec_root, visa_root, split_csv = Path(data["mvtec_root"]), Path(data["visa_root"]), Path(data["visa_split_csv"])
    mvtec_cats = mvtec_categories(mvtec_root)
    visa_cats = sorted(types["visa"]["categories"])
    normals = {("mvtec_ad", c): mvtec_train_normals(mvtec_root, c) for c in mvtec_cats}
    normals |= {("visa", c): visa_train_normals(split_csv, c) for c in visa_cats}
    defects = {("mvtec_ad", c): mvtec_test_defects(mvtec_root, c, types["mvtec_ad"]["excluded"]) for c in mvtec_cats}
    ids = types["visa"]["excluded_mask_ids"]
    # id2class.py: background 0, the candidate types, then Other where the category has it
    classes = {c: 1 + len(types["visa"]["categories"][c]) + (c in ids) for c in visa_cats}
    defects |= {("visa", c): visa_test_defects(visa_root, split_csv, c, classes[c], ids.get(c)) for c in visa_cats}

    rows = []
    for dataset in ("mvtec_ad", "visa"):
        cats = {c for d, c in normals if d == dataset}
        normal = draw(seeds["sample_seed"], {c: normals[(dataset, c)] for c in cats})
        defect = draw(seeds["sample_seed"] + 1, {c: defects[(dataset, c)] for c in cats})
        reference = draw(seeds["reference_seed"], {c: normals[(dataset, c)] for c in cats}, set(normal.values()))
        for c in sorted(cats):
            # split tells the operating side which images are train normals (the B2 colour scale uses them)
            rows.append({"dataset": dataset, "category": c, "role": "sample", "split": "train", "path": normal[c]})
            rows.append({"dataset": dataset, "category": c, "role": "sample", "split": "test", "path": defect[c]})
            rows.append({"dataset": dataset, "category": c, "role": "reference", "split": "train", "path": reference[c]})
    return rows


def check_type_names(cfg: dict, types: dict) -> list[str]:
    """Problems between configs/format_check/type_names.yaml and the MVTec AD test folders (empty when consistent)."""
    root = Path(cfg["data"]["mvtec_root"])
    problems = []
    if set(mvtec_categories(root)) != set(types["mvtec_ad"]["categories"]):
        problems.append("MVTec AD categories differ from type_names.yaml")
    for c, names in types["mvtec_ad"]["categories"].items():
        expected = set(names.values()) | set(types["mvtec_ad"]["excluded"])
        found = mvtec_test_labels(root, c) if (root / c).is_dir() else set()
        if found - expected or set(names.values()) - found:
            problems.append(f"{c}: test folders {sorted(found)} do not match type_names.yaml")
    return problems
