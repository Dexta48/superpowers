"""The office: a local web page where you watch the four agents work.

`Office` is the shared state the pipeline writes to as it runs; `serve` puts
it behind a localhost-only HTTP server that the page polls."""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("office.html")
DESKS = ("news", "chart", "signal", "risk")


def _desk():
    return {"status": "idle", "doing": "", "thought": "", "screen": "", "log": [], "summary": ""}


class Office:
    def __init__(self, demo: bool = False):
        self._lock = threading.Lock()
        self.run_requested = threading.Event()
        self._s = {
            "demo": demo, "run": 0, "phase": "idle", "error": "", "started": "", "next_run_at": None,
            "tickers": [], "desks": {d: _desk() for d in DESKS},
            "board": [], "market_context": "", "portfolio_note": "", "briefing": "",
            "chart_read": [], "cross_market": "",
        }

    def set(self, **kw) -> None:
        with self._lock:
            self._s.update(kw)

    def begin_run(self, started: str) -> None:
        with self._lock:
            self._s.update(run=self._s["run"] + 1, phase="data", error="", started=started,
                           next_run_at=None, desks={d: _desk() for d in DESKS})

    def start(self, desk: str) -> None:
        with self._lock:
            self._s["phase"] = desk
            self._s["desks"][desk].update(status="working", doing="reading the brief")

    def done(self, desk: str, summary: str) -> None:
        with self._lock:
            self._s["desks"][desk].update(status="done", doing="", summary=summary)

    def fail(self, message: str) -> None:
        with self._lock:
            self._s["error"] = message
            for d in self._s["desks"].values():
                if d["status"] == "working":
                    d.update(status="error", doing="")
            self._s["phase"] = "idle"

    def emitter(self, desk: str):
        def emit(kind: str, text: str) -> None:
            with self._lock:
                d = self._s["desks"][desk]
                if kind == "doing":
                    d["doing"] = text
                elif kind == "thought":
                    d["thought"] = (d["thought"] + text)[-700:]
                elif kind == "screen":
                    d["screen"] = (d["screen"] + text)[-2500:]
                elif kind == "screen_reset":
                    d["screen"] = ""
                elif kind == "log":
                    d["log"] = (d["log"] + [text])[-8:]
        return emit

    def json(self) -> bytes:
        with self._lock:
            return json.dumps(self._s).encode()

    def wait_for_next(self, minutes: float | None) -> None:
        """Block until the Run button is pressed or the interval elapses."""
        self.set(phase="idle", next_run_at=time.time() + minutes * 60 if minutes else None)
        self.run_requested.clear()
        deadline = time.time() + minutes * 60 if minutes else None
        # Short waits in a loop: on Windows one long wait cannot be interrupted by Ctrl-C.
        while not self.run_requested.wait(0.5):
            if deadline and time.time() >= deadline:
                return

    def serve(self, port: int) -> ThreadingHTTPServer:
        """Serve on `port`, or on a free port if another program already has it."""
        office = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
                elif self.path == "/state":
                    self._send(200, "application/json", office.json())
                else:
                    self._send(404, "text/plain", b"not found")

            def do_POST(self):
                # Same-origin only: a page on another site must not be able to
                # trigger paid model runs by posting to localhost.
                origin = self.headers.get("Origin")
                if self.path == "/run" and origin in (None, f"http://{self.headers.get('Host')}"):
                    office.run_requested.set()
                    self._send(204, "text/plain", b"")
                else:
                    self._send(403, "text/plain", b"forbidden")

            def log_message(self, *a):
                pass

        class Server(ThreadingHTTPServer):
            # Off, because on Windows address reuse lets two programs share a port
            # silently and the browser reaches whichever was there first.
            allow_reuse_address = False

        def taken(p: int) -> bool:
            with socket.socket() as probe:
                probe.settimeout(0.5)
                return probe.connect_ex(("127.0.0.1", p)) == 0

        try:
            if port and taken(port):
                raise OSError("port in use")
            server = Server(("127.0.0.1", port), Handler)
        except OSError:
            server = Server(("127.0.0.1", 0), Handler)  # another program has that port: take any free one
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
