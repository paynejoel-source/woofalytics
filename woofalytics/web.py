from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .service import BarkMonitor


STATIC_INDEX = Path(__file__).resolve().parent / "web" / "index.html"


def build_server(host: str, port: int, monitor: BarkMonitor) -> ThreadingHTTPServer:
    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                body = STATIC_INDEX.read_bytes()
                self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
                return

            if self.path == "/api/status":
                body = json.dumps(monitor.snapshot()).encode("utf-8")
                self._send(HTTPStatus.OK, body, "application/json")
                return

            if self.path == "/api/events.csv":
                csv_path = monitor.snapshot()["events_csv_path"]
                body = Path(csv_path).read_bytes()
                self._send(HTTPStatus.OK, body, "text/csv; charset=utf-8")
                return

            if self.path.startswith("/clips/"):
                clip_name = Path(self.path.removeprefix("/clips/")).name
                clip_path = monitor._config.clips_dir / clip_name
                if clip_path.exists():
                    body = clip_path.read_bytes()
                    self._send(HTTPStatus.OK, body, "audio/wav")
                    return
                self._send(HTTPStatus.NOT_FOUND, b"Clip not found", "text/plain; charset=utf-8")
                return

            self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/record":
                body = json.dumps(monitor.trigger_manual_clip()).encode("utf-8")
                self._send(HTTPStatus.OK, body, "application/json")
                return

            self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), RequestHandler)
