"""Kaggle kernel entry point: one main run VLM session, one worker per T4 (CUDA_VISIBLE_DEVICES), then merge.

Inputs: the code dataset, the image dataset (images.json), the detector kernel output (maps.json beside
features.json) and, from the second session on, the cache dataset (cache.jsonl). Output: the session result, the
shard files and the cache bundle only, so the output stays a few files.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

inputs = Path("/kaggle/input")
# mount paths differ between Kaggle versions, so locate the inputs by their stamp files
source = next(inputs.rglob("GIT_COMMIT")).parent
data_root = next(inputs.rglob("images.json")).parent
detector_root = next(p.parent for p in inputs.rglob("maps.json") if (p.parent / "features.json").exists())
cache_bundle = next(inputs.rglob("cache.jsonl"), None)
repo = Path("/tmp/clave-ad")
shutil.copytree(source, repo, dirs_exist_ok=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers==5.19.0", "bitsandbytes==0.50.2"],
    check=True,
)

import torch
import yaml

names = [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]
if not names or not all("T4" in name for name in names):
    sys.exit(f"Expected T4 GPUs, got {names}")

# map, crop and verification checks run before any GPU time is spent on generation
for check in ("src/clave_ad/data/maps.py", "src/clave_ad/verifiers/verify.py"):
    subprocess.run([sys.executable, check], cwd=repo, check=True)

config = "configs/main_run/run.yaml"
cfg = yaml.safe_load((repo / config).read_text())
vlm = yaml.safe_load((repo / cfg["vlm"]).read_text())

# restore the request cache and the out-of-memory history of earlier sessions
cache_dir = repo / cfg["cache_dir"]
cache_dir.mkdir(parents=True, exist_ok=True)
if cache_bundle is not None:
    with open(cache_bundle) as f:
        for line in f:
            item = json.loads(line)
            (cache_dir / f"{item['key']}.json").write_text(json.dumps(item["entry"]))
    oom = cache_bundle.parent / "oom.jsonl"
    if oom.exists():
        shutil.copy(oom, cache_dir / "_oom.jsonl")

# workers are new processes, so the allocator setting reaches them before their first torch import
alloc_conf = vlm["runtime"]["pytorch_cuda_alloc_conf"]
workers = [
    subprocess.Popen(
        [sys.executable, "scripts/main_run/run.py", "--config", config, "--data-root", str(data_root),
         "--detector-root", str(detector_root), "--shard", str(d), "--num-shards", str(len(names))],
        cwd=repo, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(d), "PYTORCH_CUDA_ALLOC_CONF": alloc_conf},
    )
    for d in range(len(names))
]
codes = [w.wait() for w in workers]

# merge whatever shards exist and bundle the cache, then keep only results and the bundle
report = subprocess.run([sys.executable, "scripts/main_run/report.py", "--config", config], cwd=repo)
working = Path("/kaggle/working")
for pattern in ("vlm_*.json", "vlm_shards/*.json"):
    for p in (repo / cfg["output_dir"]).glob(pattern):
        target = working / cfg["output_dir"] / p.relative_to(repo / cfg["output_dir"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, target)
if (repo / cfg["bundle_dir"]).exists():
    shutil.copytree(repo / cfg["bundle_dir"], working / cfg["bundle_dir"], dirs_exist_ok=True)
sys.exit(max(codes + [report.returncode]))
