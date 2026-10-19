"""ComfyUI failures as sentences a person can act on."""

from __future__ import annotations

from typing import Any

from comfyui_shared.client import ComfyError


def _short(text: Any, limit: int = 300) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def describe_request_error(e: ComfyError, url: str) -> str:
    if e.kind == "unreachable":
        return f"I couldn't reach your ComfyUI at {url}. Is it running?"
    if e.kind == "auth":
        return "Your ComfyUI refused the API key. Check the key in the ComfyUI package settings."
    if e.kind == "rejected":
        return describe_rejection(e.body)
    return f"Your ComfyUI answered with an error ({e.status or 'unknown'})."


def is_missing_model(body: Any) -> bool:
    for err in _node_errors(body):
        if err.get("type") == "value_not_in_list":
            return True
    return False


def _node_errors(body: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    node_errors = body.get("node_errors") if isinstance(body, dict) else None
    for node in (node_errors or {}).values():
        for err in (node or {}).get("errors") or []:
            if isinstance(err, dict):
                out.append({**err, "class_type": (node or {}).get("class_type")})
    return out


def describe_rejection(body: Any) -> str:
    """``POST /prompt`` answered 400: a missing model or node, or a bad input."""
    for err in _node_errors(body):
        if err.get("type") == "value_not_in_list":
            return f"Your ComfyUI doesn't have a model this needs ({_short(err.get('details'), 200)})."
    top = body.get("error") if isinstance(body, dict) else None
    message = (top or {}).get("message") if isinstance(top, dict) else None
    if isinstance(top, dict) and top.get("type") == "invalid_prompt" and "does not exist" in str(message):
        return f"Your ComfyUI is missing a node this needs: {_short(message, 200)} Updating ComfyUI may add it."
    errs = _node_errors(body)
    if errs:
        e = errs[0]
        return f"Your ComfyUI rejected the {e.get('class_type') or 'workflow'} step: {_short(e.get('details') or e.get('message'))}"
    return f"Your ComfyUI rejected the workflow: {_short(message or body)}"


def describe_execution_error(status: dict[str, Any] | None, noun: str) -> str:
    """A history entry whose status is ``error``: out of memory, a missing file, a crash."""
    data: dict[str, Any] = {}
    interrupted = False
    for msg in (status or {}).get("messages") or []:
        if isinstance(msg, list) and len(msg) == 2:
            if msg[0] == "execution_error" and isinstance(msg[1], dict):
                data = msg[1]
            elif msg[0] == "execution_interrupted":
                interrupted = True
    if interrupted and not data:
        return f"The {noun} was cancelled on your ComfyUI."
    exc_type = str(data.get("exception_type") or "")
    exc_msg = str(data.get("exception_message") or "")
    if "outofmemory" in exc_type.lower().replace(" ", "") or "out of memory" in exc_msg.lower():
        return (
            f"Your ComfyUI ran out of GPU memory making the {noun}. "
            "A shorter or smaller one may fit, or close other jobs on that GPU."
        )
    node = data.get("node_type") or "a"
    if exc_msg:
        return f"Your ComfyUI failed in the {node} step: {_short(exc_msg)}"
    return f"Your ComfyUI failed while making the {noun}."
