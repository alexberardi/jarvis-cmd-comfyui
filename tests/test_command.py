"""The comfyui command end to end against the fake ComfyUI: it replies at once,
queues the right workflow, saves a job, and the agent's next tick delivers the
result to the inbox."""

from datetime import datetime, timezone

import pytest
from jarvis_command_sdk import JarvisImage, RequestInformation

from agents.comfyui_jobs.agent import ComfyUIJobsAgent
from commands.comfyui.command import ComfyUICommand
from comfyui_shared import capabilities
from comfyui_shared.jobs import all_jobs
from tests.fake_comfyui import MP4, PNG, png_photo

ALICE = 42


def _req(text="make a video of a fox running through snow", user_id=ALICE):
    return RequestInformation(voice_command=text, conversation_id="c1", user_id=user_id)


def _run(**kwargs):
    return ComfyUICommand().run(_req(), **kwargs)


def _message(resp):
    return (resp.context_data or {}).get("message") or resp.error_details or ""


def _tick():
    ComfyUIJobsAgent()._tick(datetime.now(timezone.utc))


def _node(workflow, class_type):
    return next(n for n in workflow.values() if n["class_type"] == class_type)["inputs"]


@pytest.fixture
def ready(configured, inbox, llm):
    return configured


class TestTextToVideo:
    def test_replies_at_once_then_delivers_to_inbox(self, ready, inbox, llm):
        resp = _run(mode="video", prompt="a fox running through snow")
        assert resp.success
        assert "inbox when it's ready, usually a few minutes" in _message(resp)
        assert inbox.posts == []

        wf = ready.prompts[0]
        assert _node(wf, "CLIPTextEncode")["text"] == llm.reply  # the LLM-expanded prompt
        assert "a fox running through snow" in llm.prompts[0]
        assert _node(wf, "UNETLoader")["unet_name"] == "wan2.1_t2v_1.3B_fp16.safetensors"
        latent = _node(wf, "EmptyHunyuanLatentVideo")
        assert (latent["width"], latent["height"], latent["length"]) == (832, 480, 49)
        assert isinstance(_node(wf, "KSampler")["seed"], int)
        [job] = all_jobs()
        assert job["prompt_id"] == "p1" and job["user_id"] == ALICE and job["mode"] == "video"

        _tick()  # still running: nothing posted, job kept
        assert inbox.posts == [] and len(all_jobs()) == 1

        ready.finish("p1", "video_00001_.mp4", MP4)
        _tick()
        [post] = inbox.posts
        assert post["title"] == "Your video is ready"
        assert post["push"] is True and post["target_type"] == "user" and post["user_id"] == ALICE
        [att] = post["attachments"]
        assert att.mime == "video/mp4" and att.data == MP4 and att.filename == "video_00001_.mp4"
        assert all_jobs() == []

    def test_llm_unavailable_uses_users_words(self, ready, llm):
        llm.reply = None
        _run(mode="video", prompt="a fox running through snow")
        assert _node(ready.prompts[0], "CLIPTextEncode")["text"] == "a fox running through snow"

    def test_seconds_and_orientation(self, ready):
        resp = _run(mode="video", prompt="rain", seconds=5, orientation="portrait")
        latent = _node(ready.prompts[0], "EmptyHunyuanLatentVideo")
        assert (latent["width"], latent["height"], latent["length"]) == (480, 832, 81)
        assert "5-second video" in _message(resp)

    def test_no_prompt_falls_back_to_voice_command(self, ready, llm):
        ComfyUICommand().run(_req("make a video of waves"), mode="video")
        assert "make a video of waves" in llm.prompts[0]


class TestAnimatePhoto:
    def test_uploads_photo_and_sizes_from_it(self, ready, inbox):
        photo_bytes = png_photo(1080, 1920)
        resp = _run(mode="animate", prompt="she waves", photos=[JarvisImage(photo_bytes)])
        assert resp.success
        [upload] = ready.uploads
        assert photo_bytes in upload
        wf = ready.prompts[0]
        assert _node(wf, "LoadImage")["image"].startswith("jarvis_") and _node(wf, "LoadImage")["image"].endswith(".png")
        i2v = _node(wf, "WanImageToVideo")
        assert i2v["height"] == 832 and i2v["width"] < i2v["height"]
        assert _node(wf, "UNETLoader")["unet_name"] == "wan2.1_i2v_480p_14B_fp16.safetensors"

        ready.finish("p1", "anim_00001_.mp4", MP4)
        _tick()
        assert inbox.posts[0]["attachments"][0].mime == "video/mp4"

    def test_photo_without_mode_means_animate(self, ready):
        cmd = ComfyUICommand()
        args = cmd.post_process_tool_call({"photos": [1]}, "make this move")
        assert args["mode"] == "animate"
        _run(photos=[JarvisImage(png_photo(640, 480))])
        assert "WanImageToVideo" in {n["class_type"] for n in ready.prompts[0].values()}

    def test_no_photo_asks_for_one(self, ready):
        resp = _run(mode="animate", prompt="make it move")
        assert "Send me the photo" in _message(resp)
        assert ready.prompts == []


class TestTextToImage:
    def test_picture_delivered_as_png(self, ready, inbox):
        resp = _run(mode="image", prompt="a lighthouse at dusk")
        assert "inbox" in _message(resp)
        wf = ready.prompts[0]
        assert _node(wf, "CheckpointLoaderSimple")["ckpt_name"] == "sd_xl_base_1.0.safetensors"
        assert _node(wf, "EmptyLatentImage")["width"] == 1024
        ready.finish("p1", "img_00001_.png", PNG)
        _tick()
        [post] = inbox.posts
        assert post["title"] == "Your picture is ready"
        assert post["attachments"][0].mime == "image/png"


class TestOnlyWhatTheServerCanDo:
    def test_unavailable_mode_says_why_and_offers_the_rest(self, ready):
        ready.models["diffusion_models"] = []
        resp = _run(mode="video", prompt="a fox")
        msg = _message(resp)
        assert "no Wan text-to-video model" in msg and "It can make pictures." in msg
        assert ready.prompts == []

    def test_model_installed_after_a_no_is_picked_up(self, ready):
        wan = ready.models["diffusion_models"]
        ready.models["diffusion_models"] = []
        assert "no Wan text-to-video model" in _message(_run(mode="video", prompt="a fox"))
        ready.models["diffusion_models"] = wan  # the user installs the model
        resp = _run(mode="video", prompt="a fox")
        assert resp.success and "inbox when it's ready" in _message(resp)
        assert len(ready.prompts) == 1 and len(all_jobs()) == 1

    def test_capabilities_read_once(self, ready):
        _run(mode="video", prompt="a")
        _run(mode="video", prompt="b")
        assert ready.paths().count("/object_info") == 1


class TestSubmitErrors:
    def test_not_configured(self, backend, inbox):
        resp = _run(mode="video", prompt="a fox")
        assert not resp.success and "Add your ComfyUI URL" in _message(resp)

    def test_server_down(self, backend, inbox, llm):
        backend.secrets[("COMFYUI_URL", "integration", None)] = "http://127.0.0.1:1"
        resp = _run(mode="video", prompt="a fox")
        assert not resp.success and "couldn't reach your ComfyUI at http://127.0.0.1:1" in _message(resp)
        assert all_jobs() == []

    def test_wrong_key(self, ready):
        ready.required_key = "right"
        resp = _run(mode="video", prompt="a fox")
        assert "refused the API key" in _message(resp)

    def test_key_from_settings_is_sent(self, ready, backend):
        ready.required_key = "right"
        backend.secrets[("COMFYUI_API_KEY", "integration", None)] = "right"
        assert _run(mode="video", prompt="a fox").success

    def test_secrets_kwarg_wins(self, ready, backend):
        del backend.secrets[("COMFYUI_URL", "integration", None)]
        resp = ComfyUICommand().run(_req(), mode="video", prompt="a fox", secrets={"COMFYUI_URL": ready.url})
        assert resp.success

    def test_model_removed_since_cached(self, ready):
        _run(mode="image", prompt="warm the cache")
        ready.models["checkpoints"] = ["other.safetensors"]
        resp = _run(mode="image", prompt="a lighthouse")
        assert not resp.success and "doesn't have a model" in _message(resp)
        assert capabilities.JarvisStorage(capabilities.STATE_STORAGE).get(capabilities.CAPS_KEY) is None
        _run(mode="image", prompt="a lighthouse")  # re-read, picks the new checkpoint
        assert _node(ready.prompts[-1], "CheckpointLoaderSimple")["ckpt_name"] == "other.safetensors"


def test_execute_with_wire_photo_and_host_secrets(ready):
    """The host path: execute() turns the wire photo into a JarvisImage and passes secrets."""
    import base64

    wire = [{"mime": "image/png", "data": base64.b64encode(png_photo(640, 480)).decode()}]
    resp = ComfyUICommand().execute(
        _req(), secrets={"COMFYUI_URL": ready.url}, mode="animate", prompt="waves", photos=wire,
    )
    assert resp.success and len(ready.uploads) == 1


def test_execute_without_url_is_missing_secret(ready):
    from jarvis_command_sdk import MissingSecretsError

    with pytest.raises(MissingSecretsError):
        ComfyUICommand().execute(_req(), secrets={}, mode="video", prompt="a fox")
