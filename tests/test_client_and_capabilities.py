"""ComfyClient against the fake server (auth header, errors) and capability detection."""

import socket

import pytest

from comfyui_shared import capabilities
from comfyui_shared.client import ComfyClient, ComfyError


def _closed_port_url() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


class TestClient:
    def test_no_key_sends_no_auth_header(self, comfy):
        ComfyClient(comfy.url).object_info()
        headers = comfy.requests[-1]["headers"]
        assert "x-api-key" not in headers and "authorization" not in headers

    def test_key_sent_as_x_api_key_and_bearer(self, comfy):
        comfy.required_key = "sekrit"
        ComfyClient(comfy.url + "/", "sekrit").object_info()
        headers = comfy.requests[-1]["headers"]
        assert headers["x-api-key"] == "sekrit"
        assert headers["authorization"] == "Bearer sekrit"

    def test_wrong_key_is_auth_error(self, comfy):
        comfy.required_key = "sekrit"
        with pytest.raises(ComfyError) as e:
            ComfyClient(comfy.url, "nope").object_info()
        assert e.value.kind == "auth"

    def test_server_down_is_unreachable(self):
        with pytest.raises(ComfyError) as e:
            ComfyClient(_closed_port_url(), timeout=2).object_info()
        assert e.value.kind == "unreachable"

    def test_bad_url_is_unreachable(self):
        with pytest.raises(ComfyError) as e:
            ComfyClient("not a url").object_info()
        assert e.value.kind == "unreachable"


class TestDetect:
    def test_full_server_offers_all_three(self, comfy):
        caps = capabilities.detect(ComfyClient(comfy.url))
        modes = caps["modes"]
        assert modes["video"] == {"ok": True, "models": {
            "unet_name": "wan2.1_t2v_1.3B_fp16.safetensors",
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "vae_name": "wan_2.1_vae.safetensors",
        }}
        assert modes["animate"]["models"]["unet_name"] == "wan2.1_i2v_480p_14B_fp16.safetensors"
        assert modes["animate"]["models"]["clip_vision_name"] == "clip_vision_h.safetensors"
        assert modes["image"] == {"ok": True, "models": {"ckpt_name": "sd_xl_base_1.0.safetensors"}}
        assert "/object_info" in comfy.paths() and "/models/diffusion_models" in comfy.paths()

    def test_image_only_server(self, comfy):
        comfy.models["diffusion_models"] = []
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert not modes["video"]["ok"] and "text-to-video" in modes["video"]["reason"]
        assert not modes["animate"]["ok"] and "image-to-video" in modes["animate"]["reason"]
        assert modes["image"]["ok"]

    def test_no_checkpoint_no_images(self, comfy):
        comfy.models["checkpoints"] = ["wan2.1_t2v_1.3B_fp16.safetensors"]
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert not modes["image"]["ok"] and "no image model" in modes["image"]["reason"]

    def test_only_wan22_moe_halves(self, comfy):
        comfy.models["diffusion_models"] = [
            "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        ]
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert not modes["video"]["ok"] and "two-part" in modes["video"]["reason"]

    def test_ti2v_is_not_image_to_video(self, comfy):
        comfy.models["diffusion_models"] = ["wan2.2_ti2v_5B_fp16.safetensors"]
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert not modes["video"]["ok"] and not modes["animate"]["ok"]

    def test_missing_node_named(self, comfy):
        comfy.nodes.remove("SaveVideo")
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert not modes["video"]["ok"] and "SaveVideo" in modes["video"]["reason"]
        assert modes["image"]["ok"]

    def test_no_clip_vision_still_animates(self, comfy):
        comfy.models["clip_vision"] = []
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert modes["animate"]["ok"] and modes["animate"]["models"]["clip_vision_name"] is None

    def test_old_server_without_models_endpoint_uses_object_info(self, comfy):
        comfy.has_models_endpoint = False
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert modes["video"]["models"]["unet_name"] == "wan2.1_t2v_1.3B_fp16.safetensors"
        assert modes["image"]["ok"]

    def test_combo_v3_format(self):
        assert capabilities._combo_options(["COMBO", {"options": ["a.safetensors"]}]) == ["a.safetensors"]
        assert capabilities._combo_options([["b.safetensors"], {}]) == ["b.safetensors"]


GGUF_WAN = [
    "wan2.1-t2v-14b-Q5_K_S.gguf",
    "wan22RemixT2VI2V_t2vHighV20-Q6_K.gguf",
    "wan22RemixT2VI2V_i2vHighV30-Q6_K.gguf",
]


class TestGGUF:
    """Wan models quantized to GGUF, loaded by ComfyUI-GGUF's UnetLoaderGGUF (models/unet_gguf)."""

    def _gguf_only(self, comfy, files=GGUF_WAN):
        comfy.models["diffusion_models"] = []
        comfy.models["unet_gguf"] = list(files)
        comfy.nodes += ["UnetLoaderGGUF", "CLIPLoaderGGUF"]

    def test_gguf_only_server_makes_videos(self, comfy):
        self._gguf_only(comfy)
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert modes["video"]["ok"], modes["video"]
        assert modes["video"]["models"]["unet_name"] == "wan2.1-t2v-14b-Q5_K_S.gguf"
        assert modes["animate"]["models"]["unet_name"] == "wan22RemixT2VI2V_i2vHighV30-Q6_K.gguf"
        assert "/models/unet_gguf" in comfy.paths()

    def test_tag_matched_as_a_word_not_inside_t2vi2v(self, comfy):
        # "wan22RemixT2VI2V_..." holds both tags; the one after "_" says what the file is.
        self._gguf_only(comfy, GGUF_WAN[1:])
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert modes["video"]["models"]["unet_name"] == "wan22RemixT2VI2V_t2vHighV20-Q6_K.gguf"
        assert modes["animate"]["models"]["unet_name"] == "wan22RemixT2VI2V_i2vHighV30-Q6_K.gguf"

    def test_safetensors_preferred_over_gguf(self, comfy):
        comfy.models["unet_gguf"] = list(GGUF_WAN)
        comfy.nodes += ["UnetLoaderGGUF"]
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert modes["video"]["models"]["unet_name"] == "wan2.1_t2v_1.3B_fp16.safetensors"

    def test_gguf_ignored_without_the_gguf_loader(self, comfy):
        comfy.models["diffusion_models"] = []
        comfy.models["unet_gguf"] = list(GGUF_WAN)
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert not modes["video"]["ok"]
        assert "/models/unet_gguf" not in comfy.paths()

    def test_no_wan_reason_names_both_folders(self, comfy):
        self._gguf_only(comfy, [])
        reason = capabilities.detect(ComfyClient(comfy.url))["modes"]["video"]["reason"]
        assert "models/diffusion_models" in reason and "GGUF" in reason

    def test_old_server_without_models_endpoint(self, comfy):
        self._gguf_only(comfy)
        comfy.has_models_endpoint = False
        modes = capabilities.detect(ComfyClient(comfy.url))["modes"]
        assert modes["video"]["models"]["unet_name"] == "wan2.1-t2v-14b-Q5_K_S.gguf"


class TestCache:
    def test_cached_after_first_use(self, backend, comfy):
        client = ComfyClient(comfy.url)
        capabilities.load(client)
        n = len(comfy.requests)
        capabilities.load(client)
        assert len(comfy.requests) == n

    def test_url_change_rereads(self, backend, comfy):
        capabilities.load(ComfyClient(comfy.url))
        n = len(comfy.requests)
        capabilities.load(ComfyClient(comfy.url + "/api"))  # a different base URL
        assert len(comfy.requests) > n

    def test_forget_rereads(self, backend, comfy):
        client = ComfyClient(comfy.url)
        capabilities.load(client)
        capabilities.forget()
        n = len(comfy.requests)
        capabilities.load(client)
        assert len(comfy.requests) > n
