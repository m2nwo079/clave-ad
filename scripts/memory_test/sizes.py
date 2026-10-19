"""Image size candidates of the memory test: distinct sizes of the dataset images (headers only, no ground truth)
and the one whose image takes the most tokens in the model's processor (configs/extension/plan.yaml).

Run with --config to read the image headers locally and write results/extension/image_sizes_<UTC>.json (sizes and
counts per dataset, no paths); the kernel memory test reads that file instead of opening the images. Without
arguments it runs its self-check.
"""

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.runinfo import environment, header

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def image_sizes(root: Path) -> Counter:
    """Distinct (width, height) of the images under root with their counts; PIL reads only the header."""
    sizes = Counter()
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in SUFFIXES:
            with Image.open(p) as image:
                sizes[image.size] += 1
    return sizes


def size_rows(sizes: Counter) -> list[dict]:
    return [{"width": w, "height": h, "images": sizes[(w, h)]} for w, h in sorted(sizes)]


def load_sizes(result: dict) -> Counter:
    """Distinct sizes of every dataset in a size result file, counts summed over the datasets."""
    sizes = Counter()
    for d in result["metrics"]["datasets"].values():
        for r in d["sizes"]:
            sizes[(r["width"], r["height"])] += r["images"]
    return sizes


def heaviest(sizes: Counter, image_tokens: Callable[[int, int], int]) -> tuple[tuple[int, int], dict]:
    """Size with the most image tokens and the table of every candidate.

    Ties (for example a fixed token budget) go to the larger pixel count, then to the larger width.
    """
    counts = {s: image_tokens(*s) for s in sizes}
    best = max(counts, key=lambda s: (counts[s], s[0] * s[1], s))
    table = {f"{w}x{h}": {"images": sizes[(w, h)], "image_tokens": counts[(w, h)]} for w, h in sorted(sizes)}
    return best, table


def _self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a").mkdir()
        for name, size in (("a/1.png", (30, 20)), ("a/2.png", (30, 20)), ("b.JPG", (10, 40)), ("c.txt", None)):
            if size is None:
                (root / name).write_text("not an image")
            else:
                Image.new("RGB", size).save(root / name, format="PNG" if name.endswith("png") else "JPEG")
        sizes = image_sizes(root)
        assert sizes == Counter({(30, 20): 2, (10, 40): 1})
        # round trip through the result layout; a size found in two datasets sums its counts
        result = {"metrics": {"datasets": {"a": {"sizes": size_rows(sizes)},
                                           "b": {"sizes": size_rows(Counter({(30, 20): 5}))}}}}
        assert load_sizes(result) == Counter({(30, 20): 7, (10, 40): 1})
    # tokens grow with the area: the larger image wins
    best, table = heaviest(sizes, lambda w, h: w * h // 100)
    assert best == (30, 20) and table == {"10x40": {"images": 1, "image_tokens": 4},
                                          "30x20": {"images": 2, "image_tokens": 6}}
    # tokens grow with the height only: the taller image wins although it is smaller
    assert heaviest(sizes, lambda w, h: h)[0] == (10, 40)
    # fixed budget: a tie, broken by the pixel count
    assert heaviest(sizes, lambda w, h: 280)[0] == (30, 20)
    print("sizes self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    root = Path(cfg["test_input"]["size_root"])
    datasets = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        sizes = image_sizes(folder)
        datasets[folder.name] = {"images": sum(sizes.values()), "sizes": size_rows(sizes)}
        print(f"{folder.name}: {sum(sizes.values())} images, {len(sizes)} sizes")
    result = header("image_sizes", config_path, "+".join(datasets), cfg["seed"])
    result["input_folder"] = str(root)
    result["metrics"] = {"datasets": datasets}
    result["environment"] = environment()
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        _self_check()
