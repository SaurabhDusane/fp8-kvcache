"""Start/stop a ``vllm serve`` subprocess for benchmark sweeps.

- The server runs in its own process group (vLLM spawns engine-core workers) with stdout+stderr
  written to a log file.
- ``start()`` polls ``/health`` until it answers 200. If the process exits or the timeout
  passes first, it raises ``ServerStartError`` carrying the relevant part of the log. Callers
  must report that error; this module never retries with different settings.
- ``stop()`` sends SIGINT to the group (vLLM's graceful shutdown), then SIGTERM, then SIGKILL.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Sequence

import httpx

from bench.serving.sweep import extract_startup_error


class ServerStartError(RuntimeError):
    def __init__(self, message: str, log_excerpt: str, returncode: int | None) -> None:
        super().__init__(message)
        self.log_excerpt = log_excerpt
        self.returncode = returncode


def build_vllm_cmd(model: str, kv_cache_dtype: str, *, port: int, max_model_len: int,
                   gpu_memory_utilization: float, extra_args: Sequence[str] = (),
                   vllm_bin: str = "vllm") -> list[str]:
    """The exact command line; only these settings plus the caller's ``extra_args``."""
    return [vllm_bin, "serve", model,
            "--kv-cache-dtype", kv_cache_dtype,
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--port", str(port),
            *extra_args]


class VllmServer:
    def __init__(self, cmd: Sequence[str], port: int, log_path: Path | str, *,
                 host: str = "127.0.0.1", startup_timeout_s: float = 900.0,
                 env: dict[str, str] | None = None) -> None:
        self.cmd = list(cmd)
        self.port, self.host = port, host
        self.log_path = Path(log_path)
        self.startup_timeout_s = startup_timeout_s
        self.env = env
        self.proc: subprocess.Popen[bytes] | None = None
        self.startup_s: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except FileNotFoundError:
            return ""

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _healthy(self) -> bool:
        try:
            return httpx.get(self.base_url + "/health", timeout=5).status_code == 200
        except httpx.HTTPError:
            return False

    def start(self) -> float:
        """Launch and wait until healthy. Returns startup seconds."""
        if self._healthy():
            raise ServerStartError(
                f"something is already serving on port {self.port}; stop it first "
                "(the sweep must control the server it measures)", "", None)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "wb")
        log.write(("$ " + " ".join(self.cmd) + "\n").encode())
        log.flush()
        t0 = time.monotonic()
        try:
            self.proc = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True,
                                         env={**os.environ, **(self.env or {})})
        except OSError as exc:
            log.close()
            raise ServerStartError(f"could not launch {self.cmd[0]!r}: {exc}", str(exc), None) from exc
        log.close()  # the child holds its own handle
        while True:
            if self.proc.poll() is not None:
                rc = self.proc.returncode
                self.proc = None
                raise ServerStartError(f"server exited with code {rc} during startup",
                                       extract_startup_error(self.log_text()), rc)
            if self._healthy():
                self.startup_s = time.monotonic() - t0
                return self.startup_s
            if time.monotonic() - t0 > self.startup_timeout_s:
                excerpt = extract_startup_error(self.log_text())
                self.stop()
                raise ServerStartError(
                    f"server not healthy after {self.startup_timeout_s:.0f} s", excerpt, None)
            time.sleep(2.0)

    def stop(self, grace_s: float = 60.0) -> int | None:
        """Graceful shutdown of the whole process group. Returns the exit code."""
        if self.proc is None:
            return None
        proc, self.proc = self.proc, None
        if proc.poll() is None:
            for sig, wait_s in ((signal.SIGINT, grace_s), (signal.SIGTERM, 30.0),
                                (signal.SIGKILL, 10.0)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=wait_s)
                    break
                except subprocess.TimeoutExpired:
                    continue
        # Wait for the port to close so the next config can bind it.
        deadline = time.monotonic() + 30
        while self._healthy() and time.monotonic() < deadline:
            time.sleep(1.0)
        return proc.returncode

    def reset_prefix_cache(self) -> bool:
        """Best effort POST /reset_prefix_cache (only exposed in vLLM dev mode). True on 200."""
        try:
            return httpx.post(self.base_url + "/reset_prefix_cache", timeout=30).status_code == 200
        except httpx.HTTPError:
            return False

    def __enter__(self) -> "VllmServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
