"""Merge a downloaded session cache bundle into the local cache, and stage it for the private cache dataset.

The bundle of a session holds every cached request so far (restored and new), so it replaces the staged one. An
entry already in the local cache must be identical; a difference stops the merge.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main_run/run.yaml")
    parser.add_argument("--download", required=True, help="folder the session output was downloaded to")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    bundle = Path(args.download) / cfg["bundle_dir"]
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    added = same = 0
    with open(bundle / "cache.jsonl") as f:
        for line in f:
            item = json.loads(line)
            path = cache_dir / f"{item['key']}.json"
            if path.exists():
                if json.loads(path.read_text()) != item["entry"]:
                    sys.exit(f"cached request differs from the local cache: {item['key']}")
                same += 1
            else:
                path.write_text(json.dumps(item["entry"]))
                added += 1
    stage = Path(cfg["cache_dataset_dir"])
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("cache.jsonl", "oom.jsonl"):
        shutil.copy(bundle / name, stage / name)
    print(f"{added} new and {same} known requests -> {cache_dir}; bundle staged in {stage}")


if __name__ == "__main__":
    main()
