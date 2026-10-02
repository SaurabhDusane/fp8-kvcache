"""Tiny OpenAI-compatible streaming server for CPU tests of the load generator.

Stdlib asyncio only. Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions
(``stream: true`` only). Behaviour per request:
  - an immediate role-only chunk (empty content, like vLLM),
  - first content chunk after ``ttft_s + prefill_s_per_token * prompt_tokens``,
  - then one content chunk (one token) every ``itl_s``,
  - ``max_tokens`` tokens if ``ignore_eos`` else ``min(max_tokens, eos_after)``,
  - a usage chunk if ``stream_options.include_usage``, then ``data: [DONE]``.
Prompt tokens are whitespace-separated words of all message contents (matches StubTokenizer).
``fail_every=N`` answers every Nth chat request with HTTP 500.

Run standalone:  python -m bench.serving.fake_server --port 8001 --ttft-ms 50 --itl-ms 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeServerConfig:
    model: str = "fake-model"
    ttft_s: float = 0.05
    itl_s: float = 0.01
    prefill_s_per_token: float = 0.0
    eos_after: int | None = None
    fail_every: int = 0


@dataclass
class FakeServerState:
    requests: list[dict[str, Any]] = field(default_factory=list)  # parsed chat bodies, in order
    arrival_times: list[float] = field(default_factory=list)      # time.perf_counter()
    active: int = 0
    peak_active: int = 0


def _prompt_tokens(body: dict[str, Any]) -> int:
    return sum(len(str(m.get("content", "")).split()) for m in body.get("messages", []))


class FakeServer:
    def __init__(self, config: FakeServerConfig | None = None, host: str = "127.0.0.1",
                 port: int = 0) -> None:
        self.config = config or FakeServerConfig()
        self.state = FakeServerState()
        self.host, self.port = host, port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.base_events.Server | None = None
        self._thread: threading.Thread | None = None
        self._chat_count = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ---------------------------------------------------------------- HTTP plumbing
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:  # keep-alive: serve requests until the client closes
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode("latin-1").split("\r\n")
                method, path, _ = lines[0].split(" ", 2)
                headers = {k.strip().lower(): v.strip()
                           for k, v in (l.split(":", 1) for l in lines[1:] if ":" in l)}
                body = await reader.readexactly(int(headers.get("content-length", "0")))
                await self._route(method, path, body, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    @staticmethod
    async def _send_json(writer: asyncio.StreamWriter, status: int, obj: Any) -> None:
        data = json.dumps(obj).encode()
        reason = {200: "OK", 404: "Not Found", 400: "Bad Request", 500: "Internal Server Error"}
        writer.write(f"HTTP/1.1 {status} {reason.get(status, 'X')}\r\nContent-Type: application/json"
                     f"\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data)
        await writer.drain()

    @staticmethod
    async def _send_chunk(writer: asyncio.StreamWriter, payload: str) -> None:
        data = payload.encode()
        writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        await writer.drain()

    async def _route(self, method: str, path: str, body: bytes,
                     writer: asyncio.StreamWriter) -> None:
        if method == "GET" and path == "/health":
            await self._send_json(writer, 200, {})
        elif method == "GET" and path == "/v1/models":
            await self._send_json(writer, 200, {"object": "list",
                                                "data": [{"id": self.config.model, "object": "model"}]})
        elif method == "POST" and path == "/v1/chat/completions":
            await self._chat(json.loads(body or b"{}"), writer)
        else:
            await self._send_json(writer, 404, {"error": f"no route {method} {path}"})

    # ---------------------------------------------------------------- chat completions
    async def _chat(self, req: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        cfg, st = self.config, self.state
        st.requests.append(req)
        st.arrival_times.append(time.perf_counter())
        self._chat_count += 1
        if cfg.fail_every and self._chat_count % cfg.fail_every == 0:
            await self._send_json(writer, 500, {"error": "injected failure"})
            return
        if not req.get("stream"):
            await self._send_json(writer, 400, {"error": "fake server only supports stream=true"})
            return
        st.active += 1
        st.peak_active = max(st.peak_active, st.active)
        try:
            n_prompt = _prompt_tokens(req)
            n_out = int(req.get("max_tokens") or 16)
            if not req.get("ignore_eos") and cfg.eos_after is not None:
                n_out = min(n_out, cfg.eos_after)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\n\r\n")
            base = {"id": "fake", "object": "chat.completion.chunk", "model": cfg.model}

            def event(obj: dict[str, Any]) -> str:
                return f"data: {json.dumps({**base, **obj})}\n\n"

            await self._send_chunk(writer, event(
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}))
            await asyncio.sleep(cfg.ttft_s + cfg.prefill_s_per_token * n_prompt)
            for i in range(n_out):
                if i:
                    await asyncio.sleep(cfg.itl_s)
                finish = "length" if i == n_out - 1 else None
                await self._send_chunk(writer, event({"choices": [{
                    "index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": finish}]}))
            if (req.get("stream_options") or {}).get("include_usage"):
                await self._send_chunk(writer, event({"choices": [], "usage": {
                    "prompt_tokens": n_prompt, "completion_tokens": n_out,
                    "total_tokens": n_prompt + n_out}}))
            await self._send_chunk(writer, "data: [DONE]\n\n")
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            st.active -= 1

    # ---------------------------------------------------------------- lifecycle
    async def _start_async(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        self.port = self._server.sockets[0].getsockname()[1]

    def start(self) -> "FakeServer":
        """Serve from a background thread with its own event loop; returns once listening."""
        ready = threading.Event()

        def target() -> None:
            self._loop = asyncio.new_event_loop()
            self._loop.run_until_complete(self._start_async())
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=target, name="fake-openai-server", daemon=True)
        self._thread.start()
        if not ready.wait(10):
            raise RuntimeError("fake server did not start")
        return self

    def stop(self) -> None:
        if self._loop is None:
            return

        async def shutdown() -> None:
            assert self._server is not None
            self._server.close()
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()

        asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(10)
        self._loop = None

    def __enter__(self) -> "FakeServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description="fake OpenAI-compatible streaming server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--ttft-ms", type=float, default=50.0)
    ap.add_argument("--itl-ms", type=float, default=10.0)
    ap.add_argument("--model", default="fake-model")
    a = ap.parse_args()
    srv = FakeServer(FakeServerConfig(model=a.model, ttft_s=a.ttft_ms / 1e3, itl_s=a.itl_ms / 1e3),
                     host=a.host, port=a.port).start()
    print(f"fake server on {srv.base_url} (Ctrl-C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
