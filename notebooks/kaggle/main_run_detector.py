"""Kaggle kernel entry point: main run detector, one worker per T4 (CUDA_VISIBLE_DEVICES), then merge the shards."""

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
data_root = next(inputs.rglob("images.json")).parent
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
if not names or not all("T4" in name for name in names):
    sys.exit(f"Expected T4 GPUs, got {names}")

# map, crop and cell checks run before any GPU time is spent
subprocess.run([sys.executable, "src/clave_ad/data/maps.py"], cwd=repo, check=True)

config = "configs/main_run/detector.yaml"
cfg = yaml.safe_load((repo / config).read_text())
det = yaml.safe_load((repo / cfg["detector_config"]).read_text())["detector"]
# official code at the pinned commit; the checkpoints inside are checked against their sha256 by detect.py
official = Path("/tmp/anomalyclip")
with urllib.request.urlopen(f"{det['repo']}/archive/{det['commit']}.tar.gz") as response:
    tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz").extractall(official, filter="data")
checkout = next(official.iterdir())

# the CLIP backbone is downloaded once before the workers start, so they never write the same file
download = ("import sys; from pathlib import Path; sys.path.insert(0, 'src'); "
            "from clave_ad.detectors.anomalyclip import import_official; "
            f"import_official(Path({str(checkout)!r})); "
            "from AnomalyCLIP_lib.model_load import _MODELS, _download; "
            f"_download(_MODELS[{det['settings']['backbone']!r}], '/tmp/clip')")
subprocess.run([sys.executable, "-c", download], cwd=repo, check=True)

workers = [
    subprocess.Popen(
        [sys.executable, "scripts/main_run/detect.py", "--config", config, "--data-root", str(data_root),
         "--repo", str(checkout), "--clip-dir", "/tmp/clip", "--shard", str(d), "--num-shards", str(len(names))],
        cwd=repo, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(d)},
    )
    for d in range(len(names))
]
codes = [w.wait() for w in workers]

# merge whatever shards exist, then keep results and the cache even when a worker failed or ran out of time
report = subprocess.run([sys.executable, "scripts/main_run/detect_report.py", "--config", config], cwd=repo)
for folder in ("results/main_run", cfg["cache_dir"]):
    if (repo / folder).exists():
        shutil.copytree(repo / folder, Path("/kaggle/working") / folder, dirs_exist_ok=True)
sys.exit(max(codes + [report.returncode]))
