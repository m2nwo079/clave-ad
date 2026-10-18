"""Parsed VLM answers of the main run, looked up in the request cache from the session records (no ground truth).

Run as a script to check that every request of the sessions and smoke runs resolves to a cache entry and that the
cache holds nothing else.
"""

import argparse
import json
import sys
from pathlib import Path
from string import Template

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.vlm.cache_key import request_key


class Requests:
    """Rebuilds the cache key of a session record the same way scripts/main_run/run.py made it."""

    def __init__(self, main_run: dict) -> None:
        vlm = yaml.safe_load(Path(main_run["vlm"]).read_text())
        rules = yaml.safe_load(Path(main_run["rules"]).read_text())
        self.types = yaml.safe_load(Path(vlm["data"]["type_names"]).read_text())
        self.revision = vlm["model"]["revision"]
        self.gen = {**vlm["generation"], "max_retries": rules["format"]["max_retries"]}
        prompt_dir = Path(vlm["prompts"]["dir"])
        self.templates = {k: (prompt_dir / v).read_text() for k, v in vlm["prompts"].items()
                          if k not in ("dir", "output_rules")}
        self.output_rules = (prompt_dir / vlm["prompts"]["output_rules"]).read_text().strip()
        split = json.loads(Path(main_run["split_result"]).read_text())
        self.images = {(r["dataset"], r["path"]): r for r in split["vlm_images"]}
        self.references = {(r["dataset"], r["category"]): r
                           for r in json.loads(Path(main_run["references"]).read_text())["images"]
                           if r["role"] == "reference"}
        self.cache_dir = Path(main_run["cache_dir"])

    def candidates(self, dataset: str, category: str) -> list[str]:
        return list(self.types[dataset]["categories"][category])

    def key(self, record: dict, b0_answer: dict | None = None) -> str:
        d, c = record["dataset"], record["category"]
        row = self.images[(d, record["path"])]
        names = ", ".join(self.candidates(d, c))
        values = {"category": c.replace("_", " "), "candidates": names, "output_rules": self.output_rules}
        # $error is filled per retry inside the request, so the key holds the template with it unfilled
        retry = Template(self.templates["retry"]).safe_substitute(candidates=names, output_rules=self.output_rules)
        kind = record["kind"]
        # same substitution as clave_ad.vlm.chat.render, which needs torch
        if kind == "b3":
            prompt = Template(self.templates["b3"]).substitute(category=values["category"],
                                                               answer=json.dumps(b0_answer))
        else:
            prompt = Template(self.templates[kind]).substitute(values)
        if kind in ("b0", "b3"):
            images = [row["sha256"]]
        elif kind == "b1":
            images = [self.references[(d, c)]["sha256"], row["sha256"]]
        else:
            images = record["inputs_sha256"]
        return request_key(self.revision, self.gen, kind, images, prompt, retry)

    def entry(self, key: str) -> dict:
        return json.loads((self.cache_dir / f"{key}.json").read_text())


def load(result_paths: list[str], requests: Requests) -> tuple[dict, dict, set[str]]:
    """Request records with their parsed answer per (kind, dataset, path), P1 records, and the cache keys used.

    A B3 key needs the parsed B0 answer, so B0 records are resolved first.
    """
    records = [r for p in result_paths for r in json.loads(Path(p).read_text())["records"]]
    answers, p1, keys = {}, {}, set()
    order = {"b0": 0, "b1": 1, "b2": 1, "requery": 1, "b3": 2}
    for r in sorted((r for r in records if r["kind"] != "p1"), key=lambda r: order[r["kind"]]):
        b0 = answers.get(("b0", r["dataset"], r["path"]))
        key = requests.key(r, b0["parsed"] if b0 else None)
        keys.add(key)
        answers[(r["kind"], r["dataset"], r["path"])] = {"record": r, "parsed": requests.entry(key)["parsed"]}
    for r in records:
        if r["kind"] == "p1":
            p1[(r["dataset"], r["path"])] = r
    return answers, p1, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/run.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    requests = Requests(yaml.safe_load(Path(cfg["main_run"]).read_text()))
    keys = set()
    # a smoke run is resolved on its own, since its B0 answers are also in the sessions
    for paths in (cfg["sessions"], *[[p] for p in cfg["smoke"]]):
        answers, _, used = load(paths, requests)
        keys |= used
        for (kind, _, _), a in answers.items():
            if (a["parsed"] is not None) != a["record"]["ok"]:
                sys.exit(f"cache entry and record disagree on parsing: {kind}")
    cached = {p.stem for p in requests.cache_dir.glob("*.json")}
    print(f"{len(keys)} request keys, {len(cached)} cache entries, "
          f"missing {len(keys - cached)}, unused {len(cached - keys)}")
    sys.exit(0 if keys == cached else 1)


if __name__ == "__main__":
    main()
