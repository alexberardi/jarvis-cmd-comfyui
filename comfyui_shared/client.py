"""A small HTTP client for the user's ComfyUI server (stdlib only).

Endpoints used (ComfyUI's own server API, also served under ``/api``):

- ``GET  /object_info``          node classes the server has
- ``GET  /models`` + ``/models/{folder}``  model files per folder
- ``POST /upload/image``         put a chat photo in ComfyUI's input folder
- ``POST /prompt``               queue an API-format workflow → ``prompt_id``
- ``GET  /history/{prompt_id}``  ``{}`` while queued/running, then status + outputs
- ``GET  /view``                 download an output file

Auth: with an API key set, every request carries both ``X-API-Key: <key>``
(Comfy Cloud) and ``Authorization: Bearer <key>`` (most other hosted ComfyUI
services and auth proxies). With no key nothing is sent (a LAN server).

Every failure is a :class:`ComfyError` with a ``kind`` the callers turn into a
readable sentence: ``unreachable``, ``auth``, ``not_found``, ``rejected``
(``/prompt`` refused the workflow), ``http`` (anything else).
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

DEFAULT_TIMEOUT = 15.0
DOWNLOAD_TIMEOUT = 120.0


class ComfyError(Exception):
    """A ComfyUI request failed. ``kind`` says how; ``body`` is the parsed reply, if any."""

    def __init__(self, kind: str, message: str, *, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.body = body


def auth_headers(api_key: str | None) -> dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"X-API-Key": key, "Authorization": f"Bearer {key}"}


class ComfyClient:
    def __init__(self, base_url: str, api_key: str | None = None, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self._headers = auth_headers(api_key)
        self._timeout = timeout

    # ── Low level ─────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout: float | None = None,
    ) -> bytes:
        url = self.base_url + path
        if query:
            url += "?" + parse.urlencode(query)
        headers = dict(self._headers)
        if content_type:
            headers["Content-Type"] = content_type
        try:
            req = request.Request(url, data=body, method=method, headers=headers)
            with request.urlopen(req, timeout=timeout or self._timeout) as resp:
                return resp.read()
        except urlerror.HTTPError as e:
            raw = e.read() or b""
            try:
                parsed: Any = json.loads(raw)
            except ValueError:
                parsed = raw.decode("utf-8", "replace")[:500]
            kind = "auth" if e.code in (401, 403) else "not_found" if e.code == 404 else "http"
            raise ComfyError(kind, f"ComfyUI answered {e.code} for {path}", status=e.code, body=parsed) from e
        except (urlerror.URLError, OSError, ValueError) as e:
            # ValueError: a malformed URL in the settings
            raise ComfyError("unreachable", f"could not reach ComfyUI: {e}") from e

    def _json(self, method: str, path: str, **kw: Any) -> Any:
        raw = self._request(method, path, **kw)
        try:
            return json.loads(raw)
        except ValueError as e:
            raise ComfyError("http", f"ComfyUI sent non-JSON for {path}") from e

    # ── API ───────────────────────────────────────────────────────────

    def object_info(self) -> dict[str, Any]:
        info = self._json("GET", "/object_info")
        return info if isinstance(info, dict) else {}

    def model_folders(self) -> list[str]:
        folders = self._json("GET", "/models")
        return [str(f) for f in folders] if isinstance(folders, list) else []

    def models(self, folder: str) -> list[str]:
        files = self._json("GET", f"/models/{parse.quote(folder)}")
        return [str(f) for f in files] if isinstance(files, list) else []

    def upload_image(self, data: bytes, filename: str, mime: str) -> str:
        """Upload to ComfyUI's input folder; returns the name LoadImage takes."""
        boundary = uuid.uuid4().hex
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {mime}\r\n\r\n"
            ).encode() + data + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        reply = self._json(
            "POST", "/upload/image", body=b"".join(parts),
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        name = str((reply or {}).get("name") or filename)
        subfolder = str((reply or {}).get("subfolder") or "")
        return f"{subfolder}/{name}" if subfolder else name

    def queue_prompt(self, workflow: dict[str, Any], client_id: str) -> str:
        try:
            reply = self._json(
                "POST", "/prompt",
                body=json.dumps({"prompt": workflow, "client_id": client_id}).encode(),
                content_type="application/json",
            )
        except ComfyError as e:
            if e.status == 400:
                raise ComfyError("rejected", str(e), status=400, body=e.body) from e
            raise
        prompt_id = (reply or {}).get("prompt_id") if isinstance(reply, dict) else None
        if not prompt_id:
            raise ComfyError("rejected", "ComfyUI did not return a prompt_id", body=reply)
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        """The prompt's history entry, or None while it is still queued or running."""
        reply = self._json("GET", f"/history/{parse.quote(prompt_id)}")
        if isinstance(reply, dict):
            entry = reply.get(prompt_id)
            if isinstance(entry, dict):
                return entry
        return None

    def view(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        return self._request(
            "GET", "/view", query={"filename": filename, "subfolder": subfolder, "type": type_},
            timeout=DOWNLOAD_TIMEOUT,
        )
