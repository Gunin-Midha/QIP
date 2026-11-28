"""The dashboard's local web server (owner 2026-11-28: Mac only, Grafana style).

Serves ``dashboard.html`` and read-only JSON from ``dashboard_data`` on 127.0.0.1 (this Mac
only; never another interface). This is the repo's only server: it never calls out, needs no
key, and changes nothing (memory is opened read-only). Requests whose Host header is not this
machine are refused, so a web page elsewhere cannot read the data through the browser
(DNS rebinding).

Endpoints: ``/`` the page; ``/api/live``, ``/api/alphas``, ``/api/alpha?id=``, ``/api/runs``,
``/api/pool``, ``/api/levers``, ``/api/models``, ``/api/overview``.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .dashboard_data import DashboardData

HOST = "127.0.0.1"
PAGE = Path(__file__).with_name("dashboard.html")
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def make_handler(data: DashboardData):
    class Handler(BaseHTTPRequestHandler):
        server_version = "qip-dashboard"

        def log_message(self, fmt, *args):   # quiet: the terminal stays readable
            return

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            if host not in LOCAL_HOSTS:
                self._send(403, b"local only", "text/plain")
                return
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/live":
                    self._json({"runs": data.live()})
                elif url.path == "/api/alphas":
                    self._json(data.alphas())
                elif url.path == "/api/alpha":
                    one = data.alpha((q.get("id") or [""])[0])
                    self._json(one if one is not None else {"error": "not found"},
                               200 if one is not None else 404)
                elif url.path == "/api/runs":
                    self._json(data.runs())
                elif url.path == "/api/pool":
                    self._json(data.pool())
                elif url.path == "/api/levers":
                    self._json(data.levers())
                elif url.path == "/api/models":
                    self._json(data.models())
                elif url.path == "/api/overview":
                    self._json(data.overview())
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as exc:   # one bad request must not stop the server
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def serve(data_dir: str | Path, port: int = 8050, open_browser: bool = True,
          ready: threading.Event | None = None) -> None:
    """Serve until Ctrl-C."""
    data = DashboardData(data_dir)
    data.refresh(0)
    httpd = ThreadingHTTPServer((HOST, port), make_handler(data))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"dashboard: {url}  (this Mac only; Ctrl-C to stop)", flush=True)
    if ready is not None:
        ready.set()
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\ndashboard stopped")
    finally:
        httpd.server_close()
