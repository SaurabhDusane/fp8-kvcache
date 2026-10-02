#!/usr/bin/env bash
# Local GPU machine setup (WSL2, ~/kvcache, uv-managed .venv). Safe to rerun.
#
# - Installs this repo editable WITHOUT dependencies.
# - Installs the light tools (pytest, numpy, pandas, matplotlib, httpx, openai, datasets).
# - NEVER installs or upgrades torch, triton, vllm, or flashinfer: their currently installed
#   versions are pinned via a constraints file for the install, and verified unchanged after.
#   If any of them would change, the script aborts with an error.
#
# Usage (from the repo root, venv active or ./.venv present):
#   bash scripts/setup_local.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found on PATH." >&2
  exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PY="$VIRTUAL_ENV/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  echo "ERROR: no active venv and no $REPO_ROOT/.venv. Activate your venv first." >&2
  exit 1
fi
echo "Using python: $PY"

LIGHT_TOOLS=(pytest numpy pandas matplotlib httpx openai datasets)

# Prints "dist==version" for each installed protected package (nothing if not installed).
snapshot_protected() {
  "$PY" - <<'PYEOF'
import importlib.metadata as md
for dist in ("torch", "triton", "pytorch-triton", "vllm", "flashinfer-python", "flashinfer"):
    try:
        print(f"{dist}=={md.version(dist)}")
    except md.PackageNotFoundError:
        pass
PYEOF
}

TMPDIR_SETUP="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_SETUP"' EXIT
CONSTRAINTS="$TMPDIR_SETUP/protected-constraints.txt"

BEFORE="$(snapshot_protected)"
printf '%s\n' "$BEFORE" > "$CONSTRAINTS"
echo "Protected GPU packages (pinned for this install):"
if [[ -n "$BEFORE" ]]; then sed 's/^/  /' "$CONSTRAINTS"; else echo "  (none installed)"; fi

echo "==> Installing kvcache (editable, --no-deps)"
uv pip install --python "$PY" -e . --no-deps

echo "==> Installing light tools: ${LIGHT_TOOLS[*]}"
# No --upgrade: already-satisfied packages are left alone, so reruns are no-ops.
uv pip install --python "$PY" --constraint "$CONSTRAINTS" "${LIGHT_TOOLS[@]}"

AFTER="$(snapshot_protected)"
if [[ "$BEFORE" != "$AFTER" ]]; then
  echo "ERROR: protected GPU packages changed during setup!" >&2
  echo "--- before"; echo "$BEFORE"; echo "--- after"; echo "$AFTER"
  exit 1
fi

echo "==> Done. Protected GPU packages unchanged."
"$PY" -c "import kvcache; print('kvcache', kvcache.__version__, 'from', kvcache.__file__)"
