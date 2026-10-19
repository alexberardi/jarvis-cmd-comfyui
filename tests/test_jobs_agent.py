"""The comfyui_jobs agent: failures become readable inbox items, timeouts,
oversized output, an old node without attachments, and inbox retries."""

from datetime import datetime, timedelta, timezone

import pytest

from agents.comfyui_jobs.agent import ComfyUIJobsAgent
from comfyui_shared import jobs
from comfyui_shared.jobs import MAX_POST_ATTEMPTS, all_jobs, save_job
from tests.fake_comfyui import MP4

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def _job(prompt_id="p1", mode="video", minutes_ago=1, user_id=7):
    job = {
        "prompt_id": prompt_id, "mode": mode, "request": "a fox in the snow",
        "prompt": "A red fox in deep snow.", "seed": 5, "user_id": user_id,
        "submitted_at": (NOW - timedelta(minutes=minutes_ago)).isoformat(),
    }
    save_job(job)
    return job


def _tick(now=NOW):
    ComfyUIJobsAgent()._tick(now)


@pytest.fixture
def server(configured, inbox):
    return configured


def test_schedule_and_identity():
    agent = ComfyUIJobsAgent()
    assert agent.name == "comfyui_jobs"
    assert agent.schedule.interval_seconds <= 30 and agent.schedule.run_on_startup


def test_no_jobs_no_requests(server):
    _tick()
    assert server.requests == []


def test_out_of_memory(server, inbox):
    _job()
    server.history["p1"] = None
    server.fail("p1", "torch.OutOfMemoryError", "Allocation on device 0 would exceed allowed memory.")
    _tick()
    [post] = inbox.posts
    assert post["title"] == "Couldn't make your video"
    assert "ran out of GPU memory" in post["summary"]
    assert post["push"] and post["user_id"] == 7 and post["attachments"] == []
    assert all_jobs() == []


def test_cuda_oom_message(server, inbox):
    _job()
    server.fail("p1", "RuntimeError", "CUDA out of memory. Tried to allocate 2.00 GiB")
    _tick()
    assert "ran out of GPU memory" in inbox.posts[0]["summary"]


def test_other_execution_error_is_readable(server, inbox):
    _job(mode="image")
    server.fail("p1", "FileNotFoundError", "No such file: models/vae/x.safetensors", node_type="VAELoader")
    _tick()
    [post] = inbox.posts
    assert post["title"] == "Couldn't make your picture"
    assert "VAELoader step" in post["summary"] and "No such file" in post["summary"]


def test_timeout(server, inbox):
    _job(minutes_ago=31)
    server.history["p1"] = None
    _tick()
    assert "didn't finish within 30 minutes" in inbox.posts[0]["summary"]
    assert all_jobs() == []


def test_server_down_kept_until_timeout(backend, inbox):
    backend.secrets[("COMFYUI_URL", "integration", None)] = "http://127.0.0.1:1"
    _job(minutes_ago=5)
    _tick()
    assert inbox.posts == [] and len(all_jobs()) == 1
    _tick(NOW + timedelta(minutes=30))
    [post] = inbox.posts
    assert "lost touch with your ComfyUI" in post["summary"] and "Is it running?" in post["summary"]
    assert all_jobs() == []


def test_wrong_key_fails_at_once(server, inbox):
    server.required_key = "right"
    _job()
    _tick()
    assert "refused the API key" in inbox.posts[0]["summary"]


def test_download_unreachable_retried_next_tick(server, inbox):
    _job()
    server.finish("p1", "v.mp4", MP4)
    server.drop_view = 1
    _tick()
    assert inbox.posts == [] and len(all_jobs()) == 1
    _tick(NOW + timedelta(seconds=15))
    [post] = inbox.posts
    assert post["title"] != "Couldn't make your video"
    assert post["attachments"][0].data == MP4
    assert all_jobs() == []


def test_household_when_no_user(server, inbox):
    _job(user_id=None)
    server.finish("p1", "v.mp4", MP4)
    _tick()
    assert inbox.posts[0]["target_type"] == "household"


def test_prefers_output_video_over_preview(server, inbox):
    _job()
    server.finish("p1", "v.mp4", MP4)
    server.files[("", "preview.png")] = b"\x89PNG\r\n\x1a\n"
    server.history["p1"]["outputs"]["9"] = {"images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}]}
    _tick()
    assert inbox.posts[0]["attachments"][0].filename == "v.mp4"


def test_vhs_gifs_key_works(server, inbox):
    _job()
    server.finish("p1", "v.mp4", MP4, key="gifs")
    _tick()
    assert inbox.posts[0]["attachments"][0].mime == "video/mp4"


def test_finished_without_media(server, inbox):
    _job()
    server.finish("p1", "notes.txt", b"hi")
    _tick()
    assert "saved no video" in inbox.posts[0]["summary"]


def test_oversized_output(server, inbox, monkeypatch):
    monkeypatch.setattr(jobs, "ATTACHMENT_MAX_FILE_BYTES", 32)
    _job()
    server.finish("p1", "big.mp4", MP4)
    _tick()
    [post] = inbox.posts
    assert post["title"] == "Couldn't make your video"
    assert "over the inbox's" in post["summary"] and "jarvis/big.mp4" in post["summary"]
    assert post["attachments"] == []
    assert all_jobs() == []


def test_old_node_without_attachments_gets_text(configured, old_inbox):
    _job()
    configured.finish("p1", "v.mp4", MP4)
    _tick()
    [post] = old_inbox.posts
    assert post["title"] == "Your video is ready"
    assert "can't attach files" in post["body"] and "jarvis/v.mp4" in post["body"]
    assert all_jobs() == []


def test_inbox_failure_retried_then_dropped(server, inbox):
    inbox.tag = "http_error"
    _job()
    server.finish("p1", "v.mp4", MP4)
    _tick()
    [job] = all_jobs()
    assert job["post_attempts"] == 1
    for _ in range(MAX_POST_ATTEMPTS - 1):
        _tick()
    assert all_jobs() == []
    assert len(inbox.posts) == MAX_POST_ATTEMPTS


def test_settings_removed(backend, inbox):
    _job(minutes_ago=31)
    _tick()
    assert "isn't set up" in inbox.posts[0]["summary"]


def test_one_bad_job_does_not_stop_others(server, inbox):
    save_job({"prompt_id": "bad", "mode": "video", "submitted_at": NOW.isoformat()})
    _job("p1")
    server.history["bad"] = {"status": {"status_str": "success", "completed": True}, "outputs": "not a dict"}
    server.finish("p1", "v.mp4", MP4)
    _tick()
    assert [p["title"] for p in inbox.posts] == ["Your video is ready"]
