"""Format check sample, B1 reference selection and the example library split (design doc 5, 13, 13.3).

This module may read test labels and masks to exclude combined and Other images and to find single-type images.
It outputs image path lists (the library also its type names) only; the operating path reads these lists, never
the labels.
"""

import random
from collections import defaultdict
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


def check_visa_class_ids(types: dict, class_ids: dict) -> list[str]:
    """Problems between configs/main_run/visa_class_ids.yaml and type_names.yaml (empty when consistent)."""
    visa, table = types["visa"], class_ids["categories"]
    problems = [] if set(table) == set(visa["categories"]) else ["VisA categories differ"]
    for c, entry in table.items():
        if set(entry["ids"].values()) != set(visa["categories"].get(c, {})):
            problems.append(f"{c}: type names differ")
        if entry["other"] != visa["excluded_mask_ids"].get(c):
            problems.append(f"{c}: Other id differs from excluded_mask_ids")
        if 0 in entry["ids"] or entry["other"] in entry["ids"]:
            problems.append(f"{c}: background or Other id used for a type")
    return problems


def test_images(data: dict) -> list[dict]:
    """Every test image (normal and defective) of both datasets, without labels."""
    mvtec_root = Path(data["mvtec_root"])
    rows = [{"dataset": "mvtec_ad", "category": c, "path": str(p.relative_to(mvtec_root))}
            for c in mvtec_categories(mvtec_root) for p in sorted((mvtec_root / c / "test").glob("*/*.png"))]
    rows += [{"dataset": "visa", "category": r["object"], "path": r["image"]}
             for r in visa_rows(Path(data["visa_split_csv"])) if r["split"] == "test"]
    return rows


def single_type_pools(data: dict, types: dict, class_ids: dict, exclude: set[tuple[str, str]]) -> dict:
    """(dataset, category, type) -> sorted test image paths showing that type only, without excluded images.

    MVTec AD: images in a candidate type folder (combined is not a candidate). VisA: images whose original mask holds
    exactly one non-background class id and that id is a candidate type (not Other). Mask ids missing from
    visa_class_ids.yaml raise MaskValueError with the kinds of values only.
    """
    pools = defaultdict(list)
    mvtec_root = Path(data["mvtec_root"])
    for c, names in types["mvtec_ad"]["categories"].items():
        for name, label in names.items():
            paths = (str(p.relative_to(mvtec_root)) for p in (mvtec_root / c / "test" / label).glob("*.png"))
            pools[("mvtec_ad", c, name)] = sorted(p for p in paths if ("mvtec_ad", p) not in exclude)
    visa_root, table = Path(data["visa_root"]), class_ids["categories"]
    for c, entry in table.items():
        for name in entry["ids"].values():
            pools[("visa", c, name)] = []
    unexpected = set()
    for r in visa_rows(Path(data["visa_split_csv"])):
        if r["split"] != "test" or r["label"] != "anomaly" or ("visa", r["image"]) in exclude:
            continue
        entry = table[r["object"]]
        mask = Image.open(visa_root / r["mask"])
        colors = mask.getcolors(maxcolors=256) if mask.mode in ("L", "P") else None
        if colors is None:
            unexpected.add(f"mode {mask.mode}")
            continue
        values = {value for _, value in colors} - {0}
        unexpected |= {f"{r['object']}: {v}" for v in values if v not in entry["ids"] and v != entry["other"]}
        if len(values) == 1 and next(iter(values)) in entry["ids"]:
            pools[("visa", r["object"], entry["ids"][next(iter(values))])].append(r["image"])
    if unexpected:
        raise MaskValueError(f"mask values not in visa_class_ids.yaml: {sorted(unexpected)}")
    return {key: sorted(paths) for key, paths in pools.items()}


def library_split(pools: dict, per_type: int, seeds: list[int]) -> dict[int, list[dict]]:
    """Per seed, up to per_type images of every type: one generator per seed, types in sorted order."""
    out = {}
    for seed in seeds:
        rng = random.Random(seed)
        out[seed] = [{"dataset": d, "category": c, "type": t, "path": p} for (d, c, t) in sorted(pools)
                     for p in rng.sample(pools[(d, c, t)], min(per_type, len(pools[(d, c, t)])))]
    return out
