"""Kaggle kernel entry point: AnomalyCLIP on the format check images (one T4), then keep results and the map cache."""

import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

inputs = Path("/kaggle/input")
# dataset mount paths differ between Kaggle versions, so locate them by their stamp files
source = next(inputs.rglob("GIT_COMMIT")).parent
data_root = next(inputs.rglob("samples.json")).parent
repo = Path("/tmp/clave-ad")
shutil.copytree(source, repo, dirs_exist_ok=True)

# use the Kaggle packages; install only the approved ones that are missing (versions go into the result file)
modules = {"scipy": "scipy", "numpy": "numpy", "tqdm": "tqdm", "ftfy": "ftfy", "regex": "regex",
           "setuptools": "setuptools", "yaml": "pyyaml"}
missing = [pip for module, pip in modules.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

import torch
import yaml

names = [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]
if not names or "T4" not in names[0]:
    sys.exit(f"Expected a T4 GPU, got {names}")

config = "configs/format_check/anomalyclip.yaml"
det = yaml.safe_load((repo / config).read_text())["detector"]
# official code at the pinned commit; the checkpoints inside are checked against their sha256 by detect.py
official = Path("/tmp/anomalyclip")
with urllib.request.urlopen(f"{det['repo']}/archive/{det['commit']}.tar.gz") as response:
    tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz").extractall(official, filter="data")
checkout = next(official.iterdir())

run = subprocess.run(
    [sys.executable, "scripts/format_check/detect.py", "--config", config, "--data-root", str(data_root),
     "--repo", str(checkout), "--clip-dir", "/tmp/clip"],
    cwd=repo, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
)
for folder in ("results/format_check", "cache/format_check_detector"):
    if (repo / folder).exists():
        shutil.copytree(repo / folder, Path("/kaggle/working") / folder, dirs_exist_ok=True)
sys.exit(run.returncode)
