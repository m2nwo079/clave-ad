"""Kaggle kernel entry point: copy the code dataset, install pinned packages, run the memory test."""

import shutil
import subprocess
import sys
from pathlib import Path

# The dataset mount path differs between Kaggle versions, so locate it by the stamp file
source = next(Path("/kaggle/input").rglob("GIT_COMMIT")).parent
repo = Path("/tmp/clave-ad")
shutil.copytree(source, repo, dirs_exist_ok=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers==5.19.0", "bitsandbytes==0.50.2"],
    check=True,
)

import torch

names = [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]
if not names or not all("T4" in name for name in names):
    sys.exit(f"Expected T4 GPUs, got {names}")

# run.py writes a result file even when it fails, so copy before propagating its exit code
run = subprocess.run([sys.executable, "scripts/memory_test/run.py", "--config", "configs/memory_test/qwen3_5_9b_1gpu.yaml"], cwd=repo)
shutil.copytree(repo / "results" / "memory_test", "/kaggle/working/results/memory_test", dirs_exist_ok=True)
sys.exit(run.returncode)
