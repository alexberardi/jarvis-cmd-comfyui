"""Template filling: every placeholder filled with a typed value, the CLIP-vision drop, sizes."""

import json

import pytest

from comfyui_shared import workflows

VIDEO_MODELS = {"unet_name": "u.safetensors", "clip_name": "c.safetensors", "vae_name": "v.safetensors"}
MODELS = {
    "video": VIDEO_MODELS,
    "animate": {**VIDEO_MODELS, "clip_vision_name": "cv.safetensors"},
    "image": {"ckpt_name": "k.safetensors"},
}


@pytest.mark.parametrize("mode", ["video", "animate", "image"])
def test_every_placeholder_filled(mode):
    wf = workflows.build(
        mode, MODELS[mode], prompt="a fox", seed=123, width=832, height=480, length=49,
        image="in.png" if mode == "animate" else None,
    )
    assert "{{" not in json.dumps(wf)
    samplers = [n for n in wf.values() if n["class_type"] == "KSampler"]
    assert samplers[0]["inputs"]["seed"] == 123
    assert any(n["inputs"].get("text") == "a fox" for n in wf.values())


def test_video_values_typed():
    wf = workflows.build("video", VIDEO_MODELS, prompt="p", seed=1, width=480, height=832, length=33)
    latent = next(n for n in wf.values() if n["class_type"] == "EmptyHunyuanLatentVideo")["inputs"]
    assert (latent["width"], latent["height"], latent["length"]) == (480, 832, 33)
    save = next(n for n in wf.values() if n["class_type"] == "SaveVideo")["inputs"]
    assert save["format"] == "mp4"
    assert next(n for n in wf.values() if n["class_type"] == "CreateVideo")["inputs"]["fps"] == 16.0


def test_animate_without_clip_vision_drops_branch():
    wf = workflows.build(
        "animate", {**VIDEO_MODELS, "clip_vision_name": None}, prompt="p", seed=1,
        width=832, height=480, length=49, image="in.png",
    )
    assert not any(n["class_type"].startswith("CLIPVision") for n in wf.values())
    i2v = next(n for n in wf.values() if n["class_type"] == "WanImageToVideo")["inputs"]
    assert "clip_vision_output" not in i2v and i2v["start_image"] == ["12", 0]


@pytest.mark.parametrize("mode", ["video", "animate"])
def test_gguf_model_loaded_with_unet_loader_gguf(mode):
    wf = workflows.build(
        mode, {**MODELS[mode], "unet_name": "wan2.1-t2v-14b-Q5_K_S.gguf"}, prompt="p", seed=1,
        width=832, height=480, length=49, image="in.png" if mode == "animate" else None,
    )
    # UnetLoaderGGUF (ComfyUI-GGUF) takes only unet_name; UNETLoader can't read .gguf.
    assert wf["1"] == {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "wan2.1-t2v-14b-Q5_K_S.gguf"}}
    assert not any(n["class_type"] == "UNETLoader" for n in wf.values())
    assert "{{" not in json.dumps(wf)


def test_safetensors_model_keeps_unet_loader():
    wf = workflows.build("video", VIDEO_MODELS, prompt="p", seed=1, width=832, height=480, length=49)
    assert wf["1"]["class_type"] == "UNETLoader" and wf["1"]["inputs"]["weight_dtype"] == "default"


def test_missing_value_raises():
    with pytest.raises(KeyError):
        workflows.fill({"a": "{{nope}}"}, {})


def test_video_length_clamped_to_wan_frames():
    assert workflows.video_length(None) == (3, 49)
    assert workflows.video_length(10) == (5, 81)
    assert workflows.video_length(0) == (1, 17)
    assert workflows.video_length("2") == (2, 33)


def test_size_for_photo_keeps_aspect_in_multiples_of_16():
    assert workflows.size_for_photo(4032, 3024) == (832, 624)
    w, h = workflows.size_for_photo(1080, 1920)
    assert (w % 16, h % 16) == (0, 0) and h == 832 and w < h
    assert workflows.size_for_photo(None, None) == (832, 480)
