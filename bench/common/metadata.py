"""Run metadata attached to every saved result (CLAUDE.md benchmarking rule 2).

Everything degrades gracefully: on a machine without git, nvidia-smi, CUDA or a given package,
the corresponding field is the string ``"unavailable"`` rather than an exception.
GPU packages are inspected via installed-distribution metadata, never imported here.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as md
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

UNAVAILABLE = "unavailable"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Distribution names to try for each package (first hit wins).
PACKAGE_DISTS: dict[str, tuple[str, ...]] = {
    "torch": ("torch",),
    "triton": ("triton", "pytorch-triton"),
    "vllm": ("vllm",),
    "flashinfer": ("flashinfer-python", "flashinfer"),
}


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 15.0) -> str | None:
    """stdout of a command, or None if it is missing, fails, or times out."""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(repo: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Commit SHA, dirty flag (uncommitted tracked or untracked changes) and branch."""
    repo = Path(repo)
    sha = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    if sha is None:
        return {"sha": UNAVAILABLE, "dirty": UNAVAILABLE, "branch": UNAVAILABLE}
    status = _run(["git", "status", "--porcelain"], cwd=repo)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    return {
        "sha": sha,
        "dirty": UNAVAILABLE if status is None else bool(status),
        "branch": branch or UNAVAILABLE,
    }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name, dists in PACKAGE_DISTS.items():
        versions[name] = UNAVAILABLE
        for dist in dists:
            try:
                versions[name] = md.version(dist)
                break
            except md.PackageNotFoundError:
                continue
    return versions


def gpu_info(device: int = 0) -> dict[str, Any]:
    """GPU name, driver and total memory from nvidia-smi; CUDA runtime from torch if loaded."""
    info: dict[str, Any] = {
        "name": UNAVAILABLE,
        "driver": UNAVAILABLE,
        "memory_total_mib": UNAVAILABLE,
        "cuda_runtime": UNAVAILABLE,
    }
    if shutil.which("nvidia-smi"):
        out = _run([
            "nvidia-smi", "-i", str(device),
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ])
        if out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            if len(parts) == 3:
                info["name"], info["driver"] = parts[0], parts[1]
                try:
                    info["memory_total_mib"] = float(parts[2])
                except ValueError:
                    pass
    # Only consult torch if the caller already imported it: importing torch here would make
    # metadata collection slow and could initialise CUDA as a side effect.
    torch = sys.modules.get("torch")
    if torch is not None:
        if getattr(torch.version, "cuda", None):
            info["cuda_runtime"] = torch.version.cuda
        try:
            if info["name"] == UNAVAILABLE and torch.cuda.is_available():
                info["name"] = torch.cuda.get_device_name(device)
        except Exception:
            pass
    return info


def collect_metadata(device: int = 0) -> dict[str, Any]:
    now = _dt.datetime.now().astimezone()
    return {
        "timestamp": now.isoformat(timespec="seconds"),
        "timestamp_utc": now.astimezone(_dt.timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git": git_info(),
        "gpu": gpu_info(device),
        "packages": package_versions(),
    }
