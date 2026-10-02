"""Bootstrap checks: packaging rules, env_check behaviour, and the gpu auto-skip."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROTECTED = {"torch", "triton", "vllm", "flashinfer", "flashinfer-python", "pytorch-triton"}


def _dist_name(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;@]", requirement.strip(), maxsplit=1)[0].lower()


def test_package_imports() -> None:
    import kvcache

    assert kvcache.__version__


def test_pyproject_does_not_depend_on_gpu_stack() -> None:
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    project = cfg["project"]
    core = {_dist_name(r) for r in project.get("dependencies", [])}
    assert not core & PROTECTED, f"protected packages in core deps: {core & PROTECTED}"
    extras = project["optional-dependencies"]
    # GPU-stack packages may only appear in the sandbox-only cpu-dev extra.
    for name, reqs in extras.items():
        found = {_dist_name(r) for r in reqs} & PROTECTED
        if name == "cpu-dev":
            assert {"torch", "triton"} <= found
        else:
            assert not found, f"extra {name!r} lists protected packages {found}"
    markers = " ".join(cfg["tool"]["pytest"]["ini_options"]["markers"])
    assert "slow:" in markers and "gpu:" in markers


def test_setup_local_never_installs_gpu_stack() -> None:
    script = (REPO / "scripts" / "setup_local.sh").read_text()
    install_lines = [l for l in script.splitlines() if "pip install" in l and not l.lstrip().startswith("#")]
    assert install_lines
    for line in install_lines:
        assert "--upgrade" not in line and " -U " not in line
        for pkg in ("torch", "triton", "vllm", "flashinfer"):
            assert not re.search(rf"\b{pkg}\b", line), line
    light = re.search(r"LIGHT_TOOLS=\((.*)\)", script)
    assert light and set(light.group(1).split()) == {
        "pytest", "numpy", "pandas", "matplotlib", "httpx", "openai", "datasets"
    }


def test_env_check_runs_and_exits_cleanly() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "env_check.py")],
        capture_output=True, text=True, timeout=300,
    )
    out = proc.stdout
    for key in ("torch", "triton", "vllm", "flashinfer", "nvcc path", "driver", "RESULT:"):
        assert key in out, f"missing {key!r} in output:\n{out}\n{proc.stderr}"
    assert proc.returncode == 0, f"env_check failed:\n{out}\n{proc.stderr}"


@pytest.mark.gpu
def test_fp8_e4m3_roundtrip_on_gpu() -> None:
    import torch

    x = torch.tensor([0.0, 1.0, -2.0, 0.5, 448.0, -448.0], dtype=torch.float16, device="cuda")
    x8 = x.to(torch.float8_e4m3fn)
    assert x8.dtype == torch.float8_e4m3fn and x8.element_size() == 1
    assert torch.equal(x8.to(torch.float16), x)


def test_env_check_without_torch_exits_cleanly(tmp_path: Path) -> None:
    # Shadow torch with a module that fails to import -> exercises the "torch missing" path.
    (tmp_path / "torch.py").write_text("raise ImportError('hidden for test')\n")
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "env_check.py")],
        capture_output=True, text=True, timeout=300,
        env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NO CUDA (torch missing)" in proc.stdout


def test_fp8_sanity_check_logic_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import env_check
    finally:
        sys.path.pop(0)
    ok, lines = env_check.fp8_sanity_check(torch, device="cpu")
    assert ok, "\n".join(lines)
