"""Kaggle kernel entry point: one format check worker per T4 (CUDA_VISIBLE_DEVICES), then merge the shards."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

inputs = Path("/kaggle/input")
# dataset mount paths differ between Kaggle versions, so locate them by their stamp files
source = next(inputs.rglob("GIT_COMMIT")).parent
data_root = next(inputs.rglob("samples.json")).parent
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

# map restore, JET and crop checks run before any GPU time is spent on generation
subprocess.run([sys.executable, "src/clave_ad/data/maps.py"], cwd=repo, check=True)

config = "configs/format_check/qwen3_5_9b.yaml"
# workers are new processes, so the allocator setting reaches them before their first torch import
alloc_conf = yaml.safe_load((repo / config).read_text())["runtime"]["pytorch_cuda_alloc_conf"]
workers = [
    subprocess.Popen(
        [sys.executable, "scripts/format_check/run.py", "--config", config, "--data-root", str(data_root),
         "--shard", str(d), "--num-shards", str(len(names))],
        cwd=repo, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(d), "PYTORCH_CUDA_ALLOC_CONF": alloc_conf},
    )
    for d in range(len(names))
]
codes = [w.wait() for w in workers]

# merge whatever shards exist, then keep results and raw outputs even when a worker failed
report = subprocess.run([sys.executable, "scripts/format_check/report.py", "--config", config], cwd=repo)
for folder in ("results/format_check", "cache/format_check"):
    if (repo / folder).exists():
        shutil.copytree(repo / folder, Path("/kaggle/working") / folder, dirs_exist_ok=True)
sys.exit(max(codes + [report.returncode]))
