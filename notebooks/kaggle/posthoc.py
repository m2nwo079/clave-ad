"""Kaggle kernel entry point: the post-hoc c4 experiment, one worker per T4 (CUDA_VISIBLE_DEVICES).

Inputs: the code dataset and the main run image dataset (images.json). Output: the shard files and the cache bundle
only; the report with ground truth runs locally (scripts/posthoc/report.py).
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

# parser and grid checks run before any GPU time is spent on generation
for check in ("src/clave_ad/vlm/parse.py", "scripts/posthoc/grid.py"):
    subprocess.run([sys.executable, check], cwd=repo, check=True)

config = "configs/posthoc/c4_cause.yaml"
cfg = yaml.safe_load((repo / config).read_text())
main_run = yaml.safe_load((repo / cfg["main_run"]).read_text())
vlm = yaml.safe_load((repo / main_run["vlm"]).read_text())
# the committed targets file of this experiment (exactly one)
targets = sorted((repo / cfg["output_dir"]).glob("targets_*.json"))
if len(targets) != 1:
    sys.exit(f"expected one targets file, found {len(targets)}")

# workers are new processes, so the allocator setting reaches them before their first torch import
alloc_conf = vlm["runtime"]["pytorch_cuda_alloc_conf"]
workers = [
    subprocess.Popen(
        [sys.executable, "scripts/posthoc/run.py", "--config", config, "--targets", str(targets[0]),
         "--data-root", str(data_root), "--shard", str(d), "--num-shards", str(len(names))],
        cwd=repo, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(d), "PYTORCH_CUDA_ALLOC_CONF": alloc_conf},
    )
    for d in range(len(names))
]
codes = [w.wait() for w in workers]

# keep only the shard files and one cache bundle (raw outputs stay out of the result files)
working = Path("/kaggle/working")
shards = repo / cfg["output_dir"] / "posthoc_shards"
if shards.exists():
    shutil.copytree(shards, working / cfg["output_dir"] / "posthoc_shards", dirs_exist_ok=True)
bundle = working / "cache" / "posthoc_bundle"
bundle.mkdir(parents=True, exist_ok=True)
with open(bundle / "cache.jsonl", "w") as f:
    for p in sorted((repo / cfg["cache_dir"]).glob("*.json")):
        f.write(json.dumps({"key": p.stem, "entry": json.loads(p.read_text())}) + "\n")
sys.exit(max(codes))
