"""Common fields of results/*.json files (CLAUDE.md 6.1)."""

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ("torch", "torchvision", "transformers", "bitsandbytes", "accelerate", "pillow", "pyyaml", "scipy", "numpy",
            "tqdm", "setuptools", "packaging", "ftfy", "wcwidth", "regex")


def git_commit() -> str:
    # Kaggle runs a git archive with no .git, so the push script writes GIT_COMMIT
    stamp = Path("GIT_COMMIT")
    if stamp.exists():
        return stamp.read_text().strip()
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for name in PACKAGES:
        try:
            env[name] = version(name)
        except PackageNotFoundError:
            env[name] = None
    try:
        import torch
        env["cuda"] = torch.version.cuda
    except ImportError:
        env["cuda"] = None
    return env


def header(prefix: str, config_path: Path, dataset: str, seed: int) -> dict:
    created_at = datetime.now(timezone.utc)
    return {
        "run_id": f"{prefix}_{created_at:%Y%m%dT%H%M%SZ}",
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": file_sha256(config_path),
        "dataset": dataset,
        "seed": seed,
        "created_at": created_at.isoformat(),
    }
