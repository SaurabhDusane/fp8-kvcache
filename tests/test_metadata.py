"""Run metadata: git info, graceful 'unavailable' fields on CPU."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bench.common import metadata as meta
from bench.common.metadata import UNAVAILABLE, collect_metadata, git_info, gpu_info


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo,
                   check=True, capture_output=True)


def test_git_info_clean_and_dirty(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a.txt").write_text("a")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    info = git_info(tmp_path)
    assert len(info["sha"]) == 40 and info["dirty"] is False and info["branch"] == "main"
    (tmp_path / "a.txt").write_text("changed")
    assert git_info(tmp_path)["dirty"] is True


def test_git_info_outside_repo(tmp_path: Path) -> None:
    assert git_info(tmp_path) == {"sha": UNAVAILABLE, "dirty": UNAVAILABLE, "branch": UNAVAILABLE}


def test_gpu_info_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(meta.shutil, "which", lambda name: None)
    monkeypatch.delitem(meta.sys.modules, "torch", raising=False)
    assert gpu_info() == {"name": UNAVAILABLE, "driver": UNAVAILABLE,
                          "memory_total_mib": UNAVAILABLE, "cuda_runtime": UNAVAILABLE}


def test_gpu_info_parses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(meta.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(meta, "_run", lambda cmd, **kw: "NVIDIA GeForce RTX 4080 Laptop GPU, 581.42, 12282")
    info = gpu_info()
    assert info["name"] == "NVIDIA GeForce RTX 4080 Laptop GPU"
    assert info["driver"] == "581.42" and info["memory_total_mib"] == 12282.0


def test_package_versions_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(dist: str) -> str:
        raise meta.md.PackageNotFoundError(dist)

    monkeypatch.setattr(meta.md, "version", not_found)
    assert meta.package_versions() == {k: UNAVAILABLE for k in ("torch", "triton", "vllm", "flashinfer")}


def test_collect_metadata_is_json_serialisable() -> None:
    m = collect_metadata()
    json.dumps(m)
    assert {"timestamp", "git", "gpu", "packages"} <= set(m)
    assert set(m["packages"]) == {"torch", "triton", "vllm", "flashinfer"}
    assert m["git"]["sha"] != ""
