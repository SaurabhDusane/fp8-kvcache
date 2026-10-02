"""Result writer/loader/summary, using a temporary results root."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from bench.common.gpu_monitor import MonitorResult, parse_line
from bench.common.results import load_results, save_result, write_summary

META = {"timestamp": "2026-10-02T10:00:00+00:00", "git": {"sha": "abc123", "dirty": False},
        "gpu": {"name": "FakeGPU", "driver": "1.0"}, "packages": {"torch": "2.x"}}
SAMPLES = [{"t": 0.2 * i, **parse_line(l)} for i, l in enumerate([
    "2610, 9001, 141.5, 150, 71, 100, 4321",
    "2000, 9001, 150.0, 150, 85, 100, 4321",
    "1900, 9001, 150.0, 150, 86, 100, 4321",
])]


def test_save_result_layout_and_content(tmp_path: Path) -> None:
    path = save_result("peak bw/test", {"size_mb": 256}, {"gbps": [400.5, 401.0]},
                       MonitorResult(samples=SAMPLES), root=tmp_path, metadata=META)
    assert path.parent.parent == tmp_path / "raw"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.parent.name)
    assert re.fullmatch(r"peak_bw_test_\d{6}\.json", path.name)
    rec = json.loads(path.read_text())
    assert rec["metadata"] == META and rec["config"] == {"size_mb": 256}
    assert rec["metrics"]["gbps"] == [400.5, 401.0]
    assert len(rec["gpu"]["samples"]) == 3
    assert rec["gpu"]["summary"]["sm_clock_mhz"]["median"] == 2000.0
    assert rec["gpu"]["throttle"]["throttled"] is True  # 2000 < 0.85 * 2610
    assert not list(path.parent.glob("*.tmp"))


def test_save_result_same_second_does_not_overwrite(tmp_path: Path) -> None:
    paths = {save_result("x", {}, {"i": i}, None, root=tmp_path, metadata=META) for i in range(3)}
    assert len(paths) == 3


def test_save_result_accepts_sample_list_and_numpy(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    path = save_result("np", {"shape": (2, 3)}, {"t": np.float32(1.5), "arr": np.arange(3)},
                       SAMPLES, root=tmp_path, metadata=META)
    rec = json.loads(path.read_text())
    assert rec["metrics"] == {"t": 1.5, "arr": [0, 1, 2]} and rec["config"]["shape"] == [2, 3]
    assert rec["gpu"]["summary"]["sm_clock_mhz"]["n"] == 3


def test_save_result_without_samples(tmp_path: Path) -> None:
    rec = json.loads(save_result("n", {}, {}, None, root=tmp_path, metadata=META).read_text())
    assert rec["gpu"]["samples"] == [] and rec["gpu"]["throttle"] is None


def test_load_results(tmp_path: Path) -> None:
    save_result("peak_bw", {"method": "copy", "size": {"mb": 256}}, {"gbps": 400.0},
                SAMPLES, root=tmp_path, metadata=META)
    save_result("peak_bw", {"method": "sum", "size": {"mb": 256}}, {"gbps": 450.0},
                SAMPLES, root=tmp_path, metadata=META)
    save_result("other", {}, {"x": 1}, None, root=tmp_path, metadata=META)

    df = load_results("peak_bw_*", root=tmp_path)
    assert len(df) == 2
    assert list(df["config.method"]) == ["copy", "sum"]
    assert list(df["metrics.gbps"]) == [400.0, 450.0]
    assert df["config.size.mb"].tolist() == [256, 256]
    assert (df["gpu_name"] == "FakeGPU").all() and (df["git_sha"] == "abc123").all()
    assert df["throttled"].tolist() == [True, True]
    assert df["_raw"].iloc[0]["metrics"]["gbps"] == 400.0

    assert len(load_results(root=tmp_path)) == 3
    day = next((tmp_path / "raw").iterdir()).name
    assert len(load_results(f"{day}/other_*", root=tmp_path)) == 1
    assert load_results("missing_*", root=tmp_path).empty
    assert load_results(root=tmp_path / "nowhere").empty


def test_load_results_env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KVCACHE_RESULTS_DIR", str(tmp_path))
    save_result("envtest", {}, {"x": 1}, None, metadata=META)
    assert len(load_results("envtest_*")) == 1


def test_write_summary_markdown_string(tmp_path: Path) -> None:
    [p] = write_summary("baseline", "# Baseline\n\n| a |\n|---|\n| 1 |", root=tmp_path)
    assert p == tmp_path / "summary" / "baseline.md"
    assert p.read_text().endswith("| 1 |\n")


def test_write_summary_dataframe(tmp_path: Path) -> None:
    df = pd.DataFrame({"method": ["copy", "su|m"], "gbps": [401.23456, None], "_raw": [{}, {}]})
    md, csv = write_summary("peak_bw", df, root=tmp_path)
    text = md.read_text().splitlines()
    assert text[0] == "| method | gbps |"
    assert text[2] == "| copy | 401.2 |"
    assert text[3] == "| su\\|m |  |"
    back = pd.read_csv(csv)
    assert list(back.columns) == ["method", "gbps"] and back["gbps"].iloc[0] == 401.23456


def test_roundtrip_with_load_results_into_summary(tmp_path: Path) -> None:
    save_result("r", {"b": 1}, {"ms": 0.5}, SAMPLES, root=tmp_path, metadata=META)
    df = load_results("r_*", root=tmp_path)[["config.b", "metrics.ms", "throttled"]]
    md, _ = write_summary("r", df, root=tmp_path)
    assert "| 1 | 0.5 | True |" in md.read_text()
