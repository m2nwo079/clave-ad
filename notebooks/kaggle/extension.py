"""Kaggle kernel entry point of the extension experiments (configs/extension/plan.yaml), one session per run.

X0 runs the memory test on one T4 and then the format check (memory_test.yaml, format_check.yaml); X1 to X4 run
scripts/main_run/run.py with configs/extension/run.yaml, one worker per T4, then merge. Inputs: the code dataset,
the format-check sample (samples.json, X0), the main-run images (images.json, X1 to X4), the main-run detector
kernel output and, from the second VLM session on, the extension cache dataset (cache.jsonl). Output: result files
and one cache bundle.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

inputs = Path("/kaggle/input")
# mount paths differ between Kaggle versions, so locate the inputs by their stamp files
source = next(inputs.rglob("GIT_COMMIT")).parent
repo = Path("/tmp/clave-ad")
shutil.copytree(source, repo, dirs_exist_ok=True)

import yaml

config = "configs/extension/run.yaml"
cfg = yaml.safe_load((repo / config).read_text())
plan = yaml.safe_load((repo / "configs/extension/plan.yaml").read_text())
session = cfg["sessions"][cfg["session"]]
x0 = session.get("mode") == "memory_and_format_check"
memory_cfg = yaml.safe_load((repo / "configs/extension/memory_test.yaml").read_text())
vlm_path = memory_cfg["vlm"] if x0 else session.get("vlm", cfg["vlm"])
vlm = yaml.safe_load((repo / vlm_path).read_text())

# pinned packages: the common ones, plus those of the candidate model (none for Qwen)
packages = dict(plan["models"]["kernel_packages"]["common"])
for c in plan["models"]["candidates"]:
    if c["name"] == vlm["model"]["name"]:
        packages.update(plan["models"]["kernel_packages"][c["id"]])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *[f"{k}=={v}" for k, v in packages.items()]],
               check=True)

import torch

names = [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]
if not names or not all("T4" in name for name in names):
    sys.exit(f"Expected T4 GPUs, got {names}")

# parser, template, grid, map and verification checks run before any GPU time is spent on generation
for check in ("src/clave_ad/vlm/parse.py", "src/clave_ad/vlm/template.py", "scripts/posthoc/grid.py",
              "scripts/memory_test/sizes.py",
              "src/clave_ad/data/maps.py", "src/clave_ad/verifiers/verify.py"):
    subprocess.run([sys.executable, check], cwd=repo, check=True)
# attention wrapper against transformers sdpa (imports the package, so it runs as a module)
subprocess.run([sys.executable, "-m", "clave_ad.vlm.model"], cwd=repo, env={**os.environ, "PYTHONPATH": "src"},
               check=True)

alloc_conf = vlm["runtime"]["pytorch_cuda_alloc_conf"]
# unbuffered output, so the worker progress lines reach the kernel log as they are printed
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": alloc_conf, "PYTHONUNBUFFERED": "1"}
# a hung worker is killed this long after its own time limit, which it checks only between Python steps
grace_seconds = 600
working = Path("/kaggle/working")


def keep(pattern: str, folder: str) -> None:
    for p in (repo / folder).glob(pattern):
        # files committed with the code (earlier records) are not outputs of this run
        if (source / p.relative_to(repo)).exists():
            continue
        target = working / p.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, target)


def wait(worker: subprocess.Popen, deadline: float) -> int:
    try:
        return worker.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        print("worker killed after its time budget", flush=True)
        worker.kill()
        return worker.wait()


def bundle(cache_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cache.jsonl", "w") as f:
        for p in sorted(cache_dir.glob("*.json")):
            f.write(json.dumps({"key": p.stem, "entry": json.loads(p.read_text())}) + "\n")


if x0:
    # the candidate image sizes come from the committed sizes file, so the main-run images are not attached
    try:
        memory = subprocess.run([sys.executable, "scripts/memory_test/run.py", "--config",
                                 "configs/extension/memory_test.yaml"],
                                cwd=repo, env={**env, "CUDA_VISIBLE_DEVICES": "0"},
                                timeout=memory_cfg["time_limit_minutes"] * 60 + grace_seconds).returncode
    except subprocess.TimeoutExpired:
        print("memory test killed after its time limit", flush=True)
        memory = 1
    codes = [memory]
    fc_config = "configs/extension/format_check.yaml"
    fc = yaml.safe_load((repo / fc_config).read_text())
    # the format check runs only on a model that passed the memory test
    if memory == 0:
        data_root = next(inputs.rglob("samples.json")).parent
        workers = [
            subprocess.Popen(
                [sys.executable, "scripts/format_check/run.py", "--config", fc_config, "--data-root", str(data_root),
                 "--shard", str(d), "--num-shards", str(len(names))],
                cwd=repo, env={**env, "CUDA_VISIBLE_DEVICES": str(d)},
            )
            for d in range(len(names))
        ]
        # one deadline for both workers, which started together
        deadline = time.monotonic() + fc["time_budget_hours"] * 3600 + grace_seconds
        codes += [wait(w, deadline) for w in workers]
        codes.append(subprocess.run([sys.executable, "scripts/format_check/report.py", "--config", fc_config],
                                    cwd=repo).returncode)
        bundle(repo / fc["cache_dir"], working / "cache" / "extension_x0_bundle")
    keep("memory_test_*.json", fc["output_dir"])
    keep("format_check_*.json", fc["output_dir"])
    keep("shards/*.json", fc["output_dir"])
    sys.exit(max(codes))

data_root = next(inputs.rglob("images.json")).parent
detector_root = next(p.parent for p in inputs.rglob("maps.json") if (p.parent / "features.json").exists())
cache_bundle = next(inputs.rglob("cache.jsonl"), None)
cache_dir = repo / cfg["cache_dir"]
cache_dir.mkdir(parents=True, exist_ok=True)
# restore the extension request cache and the out-of-memory history of earlier sessions
if cache_bundle is not None:
    with open(cache_bundle) as f:
        for line in f:
            item = json.loads(line)
            (cache_dir / f"{item['key']}.json").write_text(json.dumps(item["entry"]))
    oom = cache_bundle.parent / "oom.jsonl"
    if oom.exists():
        shutil.copy(oom, cache_dir / "_oom.jsonl")

# workers are new processes, so the allocator setting reaches them before their first torch import
workers = [
    subprocess.Popen(
        [sys.executable, "scripts/main_run/run.py", "--config", config, "--data-root", str(data_root),
         "--detector-root", str(detector_root), "--shard", str(d), "--num-shards", str(len(names))],
        cwd=repo, env={**env, "CUDA_VISIBLE_DEVICES": str(d)},
    )
    for d in range(len(names))
]
codes = [w.wait() for w in workers]
report = subprocess.run([sys.executable, "scripts/main_run/report.py", "--config", config], cwd=repo)
keep("vlm_*.json", cfg["output_dir"])
keep("vlm_shards/*.json", cfg["output_dir"])
if (repo / cfg["bundle_dir"]).exists():
    shutil.copytree(repo / cfg["bundle_dir"], working / cfg["bundle_dir"], dirs_exist_ok=True)
sys.exit(max(codes + [report.returncode]))
