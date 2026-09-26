"""Language-neutral JSON/HTTP API for the deterministic isiXhosa analyzer."""
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .grammar import analyze_text

API_VERSION = "v1"
MAX_REQUEST_BYTES = 1_048_576


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = "XNLP/1"

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._send_json(HTTPStatus.OK, {
            "status": "ok",
            "api_version": API_VERSION,
            "language": "isiXhosa",
            "processor": "deterministic-evidence-bound-analyzer",
        })

    def do_POST(self) -> None:
        if self.path != "/v1/analyze":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
            return

        if content_length < 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
            return
        if content_length > MAX_REQUEST_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            self._send_json(HTTPStatus.BAD_REQUEST, {
                "error": "invalid_request",
                "details": "Expected a JSON object with a string field named 'text'.",
            })
            return

        self._send_json(HTTPStatus.OK, {
            "api_version": API_VERSION,
            "language": "isiXhosa",
            "processor": "deterministic-evidence-bound-analyzer",
            "result": analyze_text(payload["text"]),
        })

    def _send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        if getattr(self.server, "log_requests", True):
            super().log_message(format, *args)


class XNLPHTTPServer(ThreadingHTTPServer):
    """Threaded local HTTP server for the versioned XNLP JSON API."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], *, log_requests: bool = True):
        self.log_requests = log_requests
        super().__init__(address, _RequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the XNLP isiXhosa analysis API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    with XNLPHTTPServer((args.host, args.port)) as server:
        print(f"XNLP API listening at http://{args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("XNLP API stopped")


if __name__ == "__main__":
    main()