"""A fake ComfyUI server (stdlib http.server on a random port) for the tests.

It speaks the endpoints the package uses, records every request (with its
headers) and behaves like the real server where it matters: ``/prompt``
refuses unknown node classes and model files with ComfyUI's 400 shapes, and
``/history/{id}`` is ``{}`` until a test finishes or fails the job.
"""

from __future__ import annotations

import json
import re
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from comfyui_shared.capabilities import REQUIRED_NODES

MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def png_photo(width: int, height: int) -> bytes:
    """A PNG header JarvisImage can read the size of."""
    ihdr = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00\x00\x00\x00" + b"\x00" * 32


ALL_NODES = sorted({n for nodes in REQUIRED_NODES.values() for n in nodes} | {"CLIPVisionLoader", "CLIPVisionEncode"})

FULL_MODELS: dict[str, list[str]] = {
    "diffusion_models": ["wan2.1_t2v_1.3B_fp16.safetensors", "wan2.1_i2v_480p_14B_fp16.safetensors"],
    "text_encoders": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
    "vae": ["wan_2.1_vae.safetensors"],
    "clip_vision": ["clip_vision_h.safetensors"],
    "checkpoints": ["sd_xl_base_1.0.safetensors"],
    "loras": [],
}

# loader class -> (input name, model folder)
_LOADERS = {
    "UNETLoader": ("unet_name", "diffusion_models"),
    "CLIPLoader": ("clip_name", "text_encoders"),
    "VAELoader": ("vae_name", "vae"),
    "CLIPVisionLoader": ("clip_name", "clip_vision"),
    "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
}


class FakeComfyUI:
    def __init__(self) -> None:
        self.nodes: list[str] = list(ALL_NODES)
        self.models: dict[str, list[str]] = {k: list(v) for k, v in FULL_MODELS.items()}
        self.has_models_endpoint = True
        self.required_key: str | None = None
        self.history: dict[str, dict[str, Any] | None] = {}
        self.files: dict[tuple[str, str], bytes] = {}
        self.requests: list[dict[str, Any]] = []
        self.prompts: list[dict[str, Any]] = []
        self.uploads: list[bytes] = []
        self._next = 0
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> "FakeComfyUI":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def paths(self, method: str | None = None) -> list[str]:
        return [r["path"] for r in self.requests if method is None or r["method"] == method]

    # ── Test controls ─────────────────────────────────────────────────

    def object_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {n: {"input": {"required": {}}} for n in self.nodes}
        for node, (field, folder) in _LOADERS.items():
            if node in info:
                info[node]["input"]["required"][field] = [list(self.models.get(folder, []))]
        return info

    def finish(self, prompt_id: str, filename: str, data: bytes, *, key: str = "images", subfolder: str = "jarvis") -> None:
        self.files[(subfolder, filename)] = data
        self.history[prompt_id] = {
            "prompt": [],
            "outputs": {"11": {key: [{"filename": filename, "subfolder": subfolder, "type": "output"}]}},
            "status": {"status_str": "success", "completed": True, "messages": []},
        }

    def fail(self, prompt_id: str, exception_type: str, exception_message: str, node_type: str = "KSampler") -> None:
        self.history[prompt_id] = {
            "prompt": [],
            "outputs": {},
            "status": {
                "status_str": "error", "completed": False,
                "messages": [
                    ["execution_start", {"prompt_id": prompt_id}],
                    ["execution_error", {
                        "prompt_id": prompt_id, "node_id": "8", "node_type": node_type,
                        "exception_type": exception_type, "exception_message": exception_message,
                    }],
                ],
            },
        }

    # ── Server ────────────────────────────────────────────────────────

    def _validate(self, workflow: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
        for node in workflow.values():
            if node.get("class_type") not in self.nodes:
                return 400, {"error": {
                    "type": "invalid_prompt",
                    "message": f"Cannot execute because node {node.get('class_type')} does not exist.",
                    "details": "", "extra_info": {},
                }, "node_errors": {}}
        node_errors: dict[str, Any] = {}
        for node_id, node in workflow.items():
            loader = _LOADERS.get(node["class_type"])
            if loader:
                field, folder = loader
                value = node["inputs"].get(field)
                if value not in self.models.get(folder, []):
                    node_errors[node_id] = {"class_type": node["class_type"], "dependent_outputs": [], "errors": [{
                        "type": "value_not_in_list", "message": "Value not in list",
                        "details": f"{field}: '{value}' not in {self.models.get(folder, [])}",
                        "extra_info": {},
                    }]}
        if node_errors:
            return 400, {"error": {
                "type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                "details": "", "extra_info": {},
            }, "node_errors": node_errors}
        return None

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, body: Any, ctype: str = "application/json") -> None:
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _route(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                parsed = urlparse(self.path)
                path = parsed.path
                if path.startswith("/api/"):  # ComfyUI serves every route under /api too
                    path = path[len("/api"):]
                fake.requests.append({"method": method, "path": path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                if fake.required_key and self.headers.get("X-API-Key") != fake.required_key:
                    return self._send(401, {"error": "unauthorized"})
                if method == "GET" and path == "/object_info":
                    return self._send(200, fake.object_info())
                if method == "GET" and path == "/models":
                    if not fake.has_models_endpoint:
                        return self._send(404, b"404: Not Found", "text/plain")
                    return self._send(200, list(fake.models))
                if method == "GET" and path.startswith("/models/") and fake.has_models_endpoint:
                    folder = unquote(path[len("/models/"):])
                    if folder not in fake.models:
                        return self._send(404, b"404: Not Found", "text/plain")
                    return self._send(200, fake.models[folder])
                if method == "POST" and path == "/upload/image":
                    fake.uploads.append(body)
                    m = re.search(rb'name="image"; filename="([^"]+)"', body)
                    name = m.group(1).decode() if m else "upload.png"
                    return self._send(200, {"name": name, "subfolder": "", "type": "input"})
                if method == "POST" and path == "/prompt":
                    workflow = json.loads(body)["prompt"]
                    refused = fake._validate(workflow)
                    if refused:
                        return self._send(*refused)
                    fake._next += 1
                    prompt_id = f"p{fake._next}"
                    fake.prompts.append(workflow)
                    fake.history[prompt_id] = None
                    return self._send(200, {"prompt_id": prompt_id, "number": fake._next, "node_errors": {}})
                if method == "GET" and path.startswith("/history/"):
                    prompt_id = unquote(path[len("/history/"):])
                    entry = fake.history.get(prompt_id)
                    return self._send(200, {prompt_id: entry} if entry else {})
                if method == "GET" and path == "/view":
                    q = {k: v[0] for k, v in parse_qs(parsed.query, keep_blank_values=True).items()}
                    data = fake.files.get((q.get("subfolder", ""), q.get("filename", "")))
                    if data is None:
                        return self._send(404, b"404: Not Found", "text/plain")
                    return self._send(200, data, "application/octet-stream")
                return self._send(404, b"404: Not Found", "text/plain")

            def do_GET(self) -> None:
                self._route("GET")

            def do_POST(self) -> None:
                self._route("POST")

        return Handler
