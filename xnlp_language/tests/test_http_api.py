"""Contract tests for the language-neutral XNLP HTTP API."""
from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

from xnlp_language.server import XNLPHTTPServer


def _request(server: XNLPHTTPServer, method: str, path: str, body: bytes = b""):
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    connection = HTTPConnection(*server.server_address)
    connection.request(method, path, body=body, headers={
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "Connection": "close",
    })
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    thread.join(timeout=2)
    return response.status, json.loads(payload.decode("utf-8"))


def test_health_contract():
    with XNLPHTTPServer(("127.0.0.1", 0), log_requests=False) as server:
        status, payload = _request(server, "GET", "/health")

    assert status == 200
    assert payload["status"] == "ok"
    assert payload["api_version"] == "v1"
    assert payload["language"] == "isiXhosa"


def test_analysis_contract_preserves_text_and_marks_uncertainty():
    text = "Molo,  xyzqwerty!\n"
    body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    with XNLPHTTPServer(("127.0.0.1", 0), log_requests=False) as server:
        status, payload = _request(server, "POST", "/v1/analyze", body)

    result = payload["result"]
    assert status == 200
    assert payload["api_version"] == "v1"
    assert result["text"] == text
    assert "".join(token["surface"] for token in result["tokens"]) == text
    unknown = next(token for token in result["tokens"] if token["surface"] == "xyzqwerty")
    assert unknown["analysis"]["status"] == "UNKNOWN"


def test_analysis_rejects_invalid_payload():
    body = json.dumps({"text": 42}).encode("utf-8")
    with XNLPHTTPServer(("127.0.0.1", 0), log_requests=False) as server:
        status, payload = _request(server, "POST", "/v1/analyze", body)

    assert status == 400
    assert payload["error"] == "invalid_request"