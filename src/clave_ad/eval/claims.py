"""Ground-truth correctness of existence, location and type claims (design doc 8, 12, 13.1 A5 and A9).

This is the only place that reads test labels and masks for evaluation.
"""

import csv
from math import floor
from pathlib import Path

import numpy as np
from PIL import Image


def cell_boxes(size: tuple[int, int], grid: int) -> dict[str, tuple[int, int, int, int]]:
    width, height = size
    return {f"r{i + 1}c{j + 1}": (floor(j * width / grid), floor(i * height / grid),
                                  floor((j + 1) * width / grid), floor((i + 1) * height / grid))
            for i in range(grid) for j in range(grid)}


class GroundTruth:
    """Labels and masks of test images. VisA type names come from visa_class_ids.yaml, not from a list order."""

    def __init__(self, data: dict, types: dict, class_ids: dict, grid: int, min_pixels: int) -> None:
        self.mvtec_root, self.visa_root = Path(data["mvtec_root"]), Path(data["visa_root"])
        self.types, self.class_ids = types, class_ids["categories"]
        self.grid, self.min_pixels = grid, min_pixels
        with open(data["visa_split_csv"], newline="") as f:
            self.visa = {r["image"]: r for r in csv.DictReader(f)}

    def mask(self, dataset: str, path: str) -> np.ndarray | None:
        """Mask array of a defective image (0 is background), None for a normal image."""
        if dataset == "mvtec_ad":
            p = Path(path)
            if p.parent.name == "good" or p.parts[1] == "train":
                return None
            return np.asarray(Image.open(self.mvtec_root / p.parts[0] / "ground_truth" / p.parent.name /
                                         f"{p.stem}_mask.png"))
        row = self.visa[path]
        if row["label"] != "anomaly":
            return None
        return np.asarray(Image.open(self.visa_root / row["mask"]))

    def truth(self, dataset: str, path: str, category: str) -> dict:
        """Defective or not, ground-truth cells, candidate type names, and the design doc 8 separate flag.

        separate is "combined" for an MVTec AD combined image, "other" for a VisA mask holding the Other id.
        """
        mask = self.mask(dataset, path)
        if mask is None:
            return {"defective": False, "cells": set(), "types": set(), "separate": None}
        if dataset == "mvtec_ad":
            label = Path(path).parent.name
            names = self.types["mvtec_ad"]["categories"][category]
            found = {name for name, original in names.items() if original == label}
            separate = "combined" if label in self.types["mvtec_ad"]["excluded"] else None
        else:
            entry, values = self.class_ids[category], set(np.unique(mask).tolist())
            found = {entry["ids"][v] for v in values if v in entry["ids"]}
            separate = "other" if entry["other"] is not None and entry["other"] in values else None
        height, width = mask.shape[:2]
        cells = {c for c, (left, top, right, bottom) in cell_boxes((width, height), self.grid).items()
                 if np.count_nonzero(mask[top:bottom, left:right]) >= self.min_pixels}
        return {"defective": True, "cells": cells, "types": found, "separate": separate}

    def pixels_in_box(self, dataset: str, path: str, box: list[int]) -> int:
        """Mask pixels inside a (left, top, right, bottom) box in original pixels; 0 for a normal image (A4)."""
        mask = self.mask(dataset, path)
        if mask is None:
            return 0
        left, top, right, bottom = box
        return int(np.count_nonzero(mask[top:bottom, left:right]))


def judge(answer: dict, gt: dict, correct_share: float) -> dict:
    """Claim correctness; a defective answer on a normal image makes every claim wrong (A5)."""
    if not answer["is_defective"]:
        return {"existence": not gt["defective"], "location": None, "type": None, "full": not gt["defective"]}
    if not gt["defective"]:
        return {"existence": False, "location": False, "type": False, "full": False}
    claimed = answer["location_cells"]
    location = bool(claimed) and sum(c in gt["cells"] for c in claimed) / len(claimed) >= correct_share
    kind = answer["defect_type"] in gt["types"]
    return {"existence": True, "location": location, "type": kind, "full": location and kind}


def _self_check() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # fake MVTec AD: one defect in the top-left cell of a 9 x 9 mask, one combined image
        for label in ("crack", "combined"):
            (root / "m/toy/ground_truth" / label).mkdir(parents=True)
            mask = np.zeros((9, 9), np.uint8)
            mask[0, 0] = 255
            Image.fromarray(mask).save(root / "m/toy/ground_truth" / label / "000_mask.png")
        # fake VisA: class ids 1 (a) and 3 (Other) in cells r3c3 and r1c3
        (root / "v/toy/Masks").mkdir(parents=True)
        mask = np.zeros((9, 9), np.uint8)
        mask[8, 8], mask[0, 8] = 1, 3
        Image.fromarray(mask).save(root / "v/toy/Masks/1.png")
        rows = [{"object": "toy", "split": "test", "label": "anomaly", "image": "toy/1.JPG", "mask": "toy/Masks/1.png"},
                {"object": "toy", "split": "test", "label": "normal", "image": "toy/2.JPG", "mask": ""}]
        with open(root / "split.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        types = {"mvtec_ad": {"excluded": ["combined"], "categories": {"toy": {"crack": "crack"}}},
                 "visa": {"categories": {"toy": {"a": "A", "b": "B"}}}}
        ids = {"categories": {"toy": {"ids": {1: "a", 2: "b"}, "other": 3}}}
        data = {"mvtec_root": root / "m", "visa_root": root / "v", "visa_split_csv": root / "split.csv"}
        gt = GroundTruth(data, types, ids, 3, 1)
        crack = gt.truth("mvtec_ad", "toy/test/crack/000.png", "toy")
        assert crack == {"defective": True, "cells": {"r1c1"}, "types": {"crack"}, "separate": None}
        assert gt.truth("mvtec_ad", "toy/test/combined/000.png", "toy")["separate"] == "combined"
        assert gt.truth("mvtec_ad", "toy/test/good/000.png", "toy")["defective"] is False
        visa = gt.truth("visa", "toy/1.JPG", "toy")
        assert visa == {"defective": True, "cells": {"r1c3", "r3c3"}, "types": {"a"}, "separate": "other"}
        assert gt.truth("visa", "toy/2.JPG", "toy")["defective"] is False
        # a cell needs at least min_pixels mask pixels
        assert GroundTruth(data, types, ids, 3, 2).truth("visa", "toy/1.JPG", "toy")["cells"] == set()
        assert gt.pixels_in_box("visa", "toy/1.JPG", [6, 6, 9, 9]) == 1
        assert gt.pixels_in_box("visa", "toy/1.JPG", [0, 0, 3, 3]) == 0

    defective = {"defective": True, "cells": {"r1c1", "r1c2"}, "types": {"crack"}}
    normal = {"defective": False, "cells": set(), "types": set()}
    yes = {"is_defective": True, "defect_type": "crack", "location_cells": ["r1c1", "r3c3"]}
    # A9: exactly half of the claimed cells is enough
    assert judge(yes, defective, 0.5) == {"existence": True, "location": True, "type": True, "full": True}
    assert judge({**yes, "location_cells": ["r1c1", "r3c3", "r3c2"]}, defective, 0.5)["location"] is False
    assert judge({**yes, "location_cells": []}, defective, 0.5)["location"] is False
    assert judge({**yes, "defect_type": "dent"}, defective, 0.5)["full"] is False
    # A5: a defective answer on a normal image makes every claim wrong
    assert judge(yes, normal, 0.5) == {"existence": False, "location": False, "type": False, "full": False}
    assert judge({"is_defective": False}, normal, 0.5)["full"] is True
    assert judge({"is_defective": False}, defective, 0.5)["existence"] is False
    print("claims self-check ok")


if __name__ == "__main__":
    _self_check()
