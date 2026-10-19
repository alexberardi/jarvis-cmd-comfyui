"""Queued ComfyUI jobs and delivering their results to the inbox.

The command queues a workflow (``POST /prompt`` answers in well under a second)
and saves a job record keyed by ``prompt_id``; the ``comfyui_jobs`` agent calls
:func:`poll_jobs` on a timer. A record lives in node storage, so a node restart
loses nothing. A job ends — and its record is deleted — once an inbox item
(the media, or a readable failure) has been posted.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any

from jarvis_command_sdk import ATTACHMENT_MAX_FILE_BYTES, Attachment, JarvisInbox, JarvisStorage

from comfyui_shared.capabilities import MODE_NOUNS
from comfyui_shared.client import ComfyClient, ComfyError
from comfyui_shared.errors import describe_execution_error, describe_request_error
from comfyui_shared.log import logger

COMMAND_NAME = "comfyui"
JOBS_STORAGE = "comfyui_jobs"
JOB_TIMEOUT = timedelta(minutes=30)
MAX_POST_ATTEMPTS = 5

URL_SECRET = "COMFYUI_URL"
KEY_SECRET = "COMFYUI_API_KEY"

_MIMES = {
    ".mp4": "video/mp4", ".webm": "video/webm", ".gif": "image/gif",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
}


def settings(secrets: dict[str, str] | None = None) -> tuple[str, str]:
    """(url, api_key) from the secrets the host passed, else from node storage."""
    secrets = secrets or {}
    store = JarvisStorage(COMMAND_NAME)
    url = secrets.get(URL_SECRET) or store.get_secret(URL_SECRET) or ""
    key = secrets.get(KEY_SECRET) or store.get_secret(KEY_SECRET) or ""
    return url.strip(), key.strip()


def save_job(job: dict[str, Any]) -> None:
    JarvisStorage(JOBS_STORAGE).save(job["prompt_id"], job)


def all_jobs() -> list[dict[str, Any]]:
    return JarvisStorage(JOBS_STORAGE).get_all()


def _delete(job: dict[str, Any]) -> None:
    JarvisStorage(JOBS_STORAGE).delete(job["prompt_id"])


def _parse(ts: Any) -> datetime:
    try:
        dt = datetime.fromisoformat(str(ts))
    except ValueError:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def pick_output(entry: dict[str, Any], mode: str) -> dict[str, str] | None:
    """The file to deliver: an ``output`` (not preview) file, video first for the video modes."""
    want_video = mode in ("video", "animate")
    found: list[tuple[int, dict[str, str]]] = []
    for node_out in (entry.get("outputs") or {}).values():
        for items in (node_out or {}).values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                mime = _MIMES.get(PurePosixPath(str(item["filename"])).suffix.lower())
                if mime is None:
                    continue
                rank = (0 if item.get("type", "output") == "output" else 2) + (
                    0 if mime.startswith("video/") == want_video else 1
                )
                found.append((rank, {
                    "filename": str(item["filename"]), "subfolder": str(item.get("subfolder") or ""),
                    "type": str(item.get("type") or "output"), "mime": mime,
                }))
    found.sort(key=lambda r: r[0])
    return found[0][1] if found else None


def _where(out: dict[str, str]) -> str:
    return f"{out['subfolder']}/{out['filename']}" if out["subfolder"] else out["filename"]


def _post(job: dict[str, Any], **kw: Any) -> str:
    user_id = job.get("user_id")
    return JarvisInbox(COMMAND_NAME).post(
        user_id=user_id,
        target_type="user" if user_id is not None else "household",
        create_push_notification=True,
        **kw,
    )


def _summary(job: dict[str, Any]) -> str:
    words = " ".join(str(job.get("request") or "").split())
    return words if len(words) <= 140 else words[:139] + "…"


def _details(job: dict[str, Any]) -> str:
    return f"Prompt: {job.get('prompt', '')}\nSeed: {job.get('seed', '')}"


def post_failure(job: dict[str, Any], reason: str) -> str:
    noun = MODE_NOUNS.get(job.get("mode", ""), "video")
    return _post(
        job, title=f"Couldn't make your {noun}", summary=reason,
        body=f"{reason}\n\nYou asked for: {job.get('request', '')}\n{_details(job)}",
    )


def _deliver(job: dict[str, Any], out: dict[str, str], data: bytes) -> str:
    noun = MODE_NOUNS.get(job.get("mode", ""), "video")
    title = f"Your {noun} is ready"
    if len(data) > ATTACHMENT_MAX_FILE_BYTES:
        mb = len(data) / (1 << 20)
        return post_failure(job, (
            f"The {noun} came out at {mb:.0f} MB, over the inbox's {ATTACHMENT_MAX_FILE_BYTES >> 20} MB limit. "
            f"It's saved on your ComfyUI as {_where(out)}."
        ))
    tag = _post(
        job, title=title, summary=_summary(job), body=_details(job),
        attachments=[Attachment(data=data, mime=out["mime"], filename=PurePosixPath(out["filename"]).name)],
    )
    if tag == "attachments_unsupported":
        logger.warning("comfyui: node inbox backend takes no attachments; posting text only", prompt_id=job["prompt_id"])
        return _post(job, title=title, summary=_summary(job), body=(
            "This Jarvis node can't attach files to inbox items yet (updating the node fixes that). "
            f"The {noun} is saved on your ComfyUI as {_where(out)}.\n\n{_details(job)}"
        ))
    if tag == "attachment_too_large":
        return post_failure(job, f"The {noun} is too large for the inbox. It's saved on your ComfyUI as {_where(out)}.")
    if tag == "attachment_type":
        return post_failure(job, (
            f"Your ComfyUI made a file the inbox can't show ({out['mime']}). "
            f"It's saved on your ComfyUI as {_where(out)}."
        ))
    return tag


def _finish(job: dict[str, Any], tag: str) -> None:
    """Delete the job once its item is posted; retry a failed post a few ticks, then give up."""
    if tag == "ok":
        _delete(job)
        return
    attempts = int(job.get("post_attempts") or 0) + 1
    logger.warning("comfyui: inbox post failed", prompt_id=job["prompt_id"], tag=tag, attempt=attempts)
    if attempts >= MAX_POST_ATTEMPTS:
        logger.error("comfyui: giving up on job after failed inbox posts", prompt_id=job["prompt_id"], tag=tag)
        _delete(job)
        return
    save_job({**job, "post_attempts": attempts})


def poll_job(client: ComfyClient, job: dict[str, Any], now: datetime) -> None:
    noun = MODE_NOUNS.get(job.get("mode", ""), "video")
    overdue = now - _parse(job.get("submitted_at")) > JOB_TIMEOUT
    try:
        entry = client.history(job["prompt_id"])
    except ComfyError as e:
        if e.kind == "auth":
            _finish(job, post_failure(job, describe_request_error(e, client.base_url)))
        elif overdue:
            _finish(job, post_failure(job, (
                f"I lost touch with your ComfyUI at {client.base_url} before the {noun} was done. "
                + describe_request_error(e, client.base_url)
            )))
        return
    if entry is None:
        if overdue:
            minutes = int(JOB_TIMEOUT.total_seconds() // 60)
            _finish(job, post_failure(job, (
                f"The {noun} didn't finish within {minutes} minutes; your ComfyUI may be busy or stuck."
            )))
        return
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        _finish(job, post_failure(job, describe_execution_error(status, noun)))
        return
    if not status.get("completed", True):
        return
    out = pick_output(entry, job.get("mode", "video"))
    if out is None:
        _finish(job, post_failure(job, f"Your ComfyUI finished but saved no {noun} I can send."))
        return
    try:
        data = client.view(out["filename"], out["subfolder"], out["type"])
    except ComfyError as e:
        if e.kind == "unreachable" and not overdue:
            return  # try the download again next tick
        _finish(job, post_failure(job, (
            f"The {noun} is done but I couldn't download it: {describe_request_error(e, client.base_url)} "
            f"It's saved on your ComfyUI as {_where(out)}."
        )))
        return
    _finish(job, _deliver(job, out, data))


def poll_jobs(now: datetime | None = None) -> None:
    jobs = all_jobs()
    if not jobs:
        return
    now = now or datetime.now(timezone.utc)
    url, key = settings()
    client = ComfyClient(url, key) if url else None
    for job in jobs:
        try:
            if client is None:
                if now - _parse(job.get("submitted_at")) > JOB_TIMEOUT:
                    _finish(job, post_failure(job, "ComfyUI isn't set up any more (no URL in the package settings)."))
                continue
            poll_job(client, job, now)
        except Exception as exc:  # one bad job must not stop the others
            logger.error("comfyui: job poll failed", prompt_id=job.get("prompt_id"), error=str(exc))
