"""Result writer/loader (CLAUDE.md benchmarking rules 2 and 7).

- ``save_result`` writes one JSON per run to
  ``bench/results/raw/<YYYY-MM-DD>/<name>_<HHMMSS>.json`` with metadata, config, metrics and
  GPU samples.
- ``load_results`` loads saved runs into a flat pandas DataFrame (one row per run).
- ``write_summary`` writes small markdown (and CSV for DataFrames) to ``bench/results/summary/``.

The results root defaults to ``<repo>/bench/results``; override it with the ``root`` argument
or the ``KVCACHE_RESULTS_DIR`` environment variable (used by tests).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from bench.common.gpu_monitor import MonitorResult, summarize
from bench.common.metadata import REPO_ROOT, collect_metadata

if TYPE_CHECKING:
    import pandas as pd

SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def results_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("KVCACHE_RESULTS_DIR")
    return Path(env) if env else REPO_ROOT / "bench" / "results"


def _safe_name(name: str) -> str:
    safe = _NAME_RE.sub("_", name).strip("_")
    if not safe:
        raise ValueError(f"invalid result name: {name!r}")
    return safe


def _json_default(obj: Any) -> Any:
    """Make numpy scalars/arrays, torch scalars, Paths and dataclass-like objects serialisable."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "tolist"):  # numpy arrays/scalars, torch tensors
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (Path, _dt.date)):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _gpu_block(gpu_samples: MonitorResult | Sequence[dict[str, Any]] | None) -> dict[str, Any]:
    if gpu_samples is None:
        return {"samples": [], "summary": summarize([]), "throttle": None,
                "warnings": ["no GPU samples recorded"]}
    if isinstance(gpu_samples, MonitorResult):
        return gpu_samples.to_dict()
    samples = list(gpu_samples)
    return MonitorResult(samples=samples).to_dict()


def save_result(name: str, config: dict[str, Any], metrics: dict[str, Any],
                gpu_samples: MonitorResult | Sequence[dict[str, Any]] | None = None, *,
                root: Path | str | None = None,
                metadata: dict[str, Any] | None = None) -> Path:
    """Write one run to raw/<date>/<name>_<HHMMSS>.json and return the path.

    ``gpu_samples`` is a ``MonitorResult`` (preferred: keeps warnings and timing) or a list of
    sample dicts. ``metadata`` defaults to ``collect_metadata()``.
    """
    safe = _safe_name(name)
    meta = metadata if metadata is not None else collect_metadata()
    now = _dt.datetime.now()
    out_dir = results_root(root) / "raw" / now.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe}_{now.strftime('%H%M%S')}"
    path = out_dir / f"{stem}.json"
    i = 1
    while path.exists():  # several saves within one second
        path = out_dir / f"{stem}_{i}.json"
        i += 1
    record = {
        "schema_version": SCHEMA_VERSION,
        "name": safe,
        "saved_at": now.astimezone().isoformat(timespec="seconds"),
        "metadata": meta,
        "config": config,
        "metrics": metrics,
        "gpu": _gpu_block(gpu_samples),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=_json_default, allow_nan=True))
    tmp.replace(path)  # atomic: a crashed run never leaves a half-written result
    return path


def _flatten(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


def _row(record: dict[str, Any], path: Path) -> dict[str, Any]:
    meta = record.get("metadata", {})
    gpu = record.get("gpu", {}) or {}
    throttle = gpu.get("throttle") or {}
    sm = (gpu.get("summary") or {}).get("sm_clock_mhz") or {}
    row: dict[str, Any] = {
        "name": record.get("name"),
        "path": str(path),
        "saved_at": record.get("saved_at"),
        "git_sha": (meta.get("git") or {}).get("sha"),
        "git_dirty": (meta.get("git") or {}).get("dirty"),
        "gpu_name": (meta.get("gpu") or {}).get("name"),
        "gpu_driver": (meta.get("gpu") or {}).get("driver"),
        "throttled": throttle.get("throttled"),
        "sm_clock_median_mhz": sm.get("median"),
        "sm_clock_max_mhz": sm.get("max"),
    }
    for pkg, ver in (meta.get("packages") or {}).items():
        row[f"pkg.{pkg}"] = ver
    row.update(_flatten(record.get("config") or {}, "config"))
    row.update(_flatten(record.get("metrics") or {}, "metrics"))
    row["_raw"] = record
    return row


def load_results(pattern: str = "*", root: Path | str | None = None) -> "pd.DataFrame":
    """Load saved runs whose file name matches ``pattern`` into a DataFrame, oldest first.

    ``pattern`` is a glob on the file name (``"peak_bw_*"``) searched in every date directory,
    or a path relative to ``raw/`` if it contains "/" (``"2026-10-02/peak_bw_*"``).
    Nested config/metrics keys become ``config.<k>`` / ``metrics.<k>`` columns; the full
    record is kept in ``_raw``.
    """
    import pandas as pd

    raw = results_root(root) / "raw"
    if not pattern.endswith(".json"):
        pattern += ".json"
    glob = pattern if "/" in pattern else f"*/{pattern}"
    paths = sorted(raw.glob(glob)) if raw.exists() else []
    rows = []
    for path in paths:
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt result file {path}: {exc}") from exc
        rows.append(_row(record, path))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["saved_at", "path"], kind="stable").reset_index(drop=True)
    return df


def _fmt_cell(v: Any, floatfmt: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return format(v, floatfmt)
    return str(v).replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown(df: "pd.DataFrame", floatfmt: str = ".4g") -> str:
    """GitHub markdown table without the optional ``tabulate`` dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt_cell(v, floatfmt) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_summary(name: str, df_or_markdown: "pd.DataFrame | str", *,
                  root: Path | str | None = None, floatfmt: str = ".4g") -> list[Path]:
    """Write ``summary/<name>.md`` (and ``<name>.csv`` with full precision for a DataFrame).

    Returns the written paths. Content must come from saved results, never typed by hand.
    """
    safe = _safe_name(name)
    out_dir = results_root(root) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{safe}.md"
    if isinstance(df_or_markdown, str):
        text = df_or_markdown if df_or_markdown.endswith("\n") else df_or_markdown + "\n"
        md_path.write_text(text)
        return [md_path]
    df = df_or_markdown.drop(columns=["_raw"], errors="ignore")
    md_path.write_text(dataframe_to_markdown(df, floatfmt))
    csv_path = out_dir / f"{safe}.csv"
    df.to_csv(csv_path, index=False)
    return [md_path, csv_path]
