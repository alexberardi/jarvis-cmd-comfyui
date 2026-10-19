"""What the user's ComfyUI can make, read from ``/object_info`` and ``/models``.

Three modes, each offered only when the server has what its workflow needs:

- ``video``   text to video: a Wan 2.x text-to-video model (``*t2v*`` in
              diffusion_models), a umt5 text encoder and a Wan VAE.
- ``animate`` image to video: a Wan 2.x image-to-video model (``*i2v*``), the
              same encoder and VAE, and the ``WanImageToVideo`` node. A CLIP
              vision model (``clip_vision_h``) is used when there is one.
- ``image``   text to image: any checkpoint in models/checkpoints.

The result is a plain dict (cached in node storage, see :func:`load`):
``{"url", "checked_at", "modes": {mode: {"ok": True, "models": {...}} |
{"ok": False, "reason": "..."}}}``. Reasons are sentences meant to be read to
the user.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jarvis_command_sdk import JarvisStorage

from comfyui_shared.client import ComfyClient, ComfyError

STATE_STORAGE = "comfyui_state"
CAPS_KEY = "capabilities"
CACHE_TTL = timedelta(hours=6)

MODES = ("video", "animate", "image")
MODE_NOUNS = {"video": "video", "animate": "video", "image": "picture"}

VIDEO_NODES = (
    "UNETLoader", "CLIPLoader", "VAELoader", "ModelSamplingSD3", "CLIPTextEncode",
    "KSampler", "VAEDecode", "CreateVideo", "SaveVideo",
)
REQUIRED_NODES: dict[str, tuple[str, ...]] = {
    "video": VIDEO_NODES + ("EmptyHunyuanLatentVideo",),
    "animate": VIDEO_NODES + ("LoadImage", "WanImageToVideo"),
    "image": ("CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"),
}

# Model folders (current name first, then the legacy alias) and, for servers
# without the /models endpoint, the loader input that lists the same files.
_FOLDERS: dict[str, tuple[tuple[str, ...], tuple[str, str]]] = {
    "diffusion_models": (("diffusion_models", "unet"), ("UNETLoader", "unet_name")),
    "text_encoders": (("text_encoders", "clip"), ("CLIPLoader", "clip_name")),
    "vae": (("vae",), ("VAELoader", "vae_name")),
    "clip_vision": (("clip_vision",), ("CLIPVisionLoader", "clip_name")),
    "checkpoints": (("checkpoints",), ("CheckpointLoaderSimple", "ckpt_name")),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _combo_options(spec: Any) -> list[str]:
    """Options of an object_info combo input: ``[[...]]`` or ``["COMBO", {"options": [...]}]``."""
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            return [str(x) for x in spec[0]]
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return [str(x) for x in spec[1].get("options") or []]
    return []


def _from_object_info(object_info: dict[str, Any], node: str, field: str) -> list[str]:
    inputs = (object_info.get(node) or {}).get("input") or {}
    spec = (inputs.get("required") or {}).get(field) or (inputs.get("optional") or {}).get(field)
    return _combo_options(spec)


def fetch_models(client: ComfyClient, object_info: dict[str, Any]) -> dict[str, list[str]]:
    """Model files per folder: from /models, or from the loaders' inputs if it is missing."""
    try:
        available = set(client.model_folders())
    except ComfyError as e:
        if e.kind != "not_found":
            raise
        available = None
    out: dict[str, list[str]] = {}
    for key, (folders, (node, field)) in _FOLDERS.items():
        if available is None:
            out[key] = _from_object_info(object_info, node, field)
            continue
        files: list[str] = []
        for folder in folders:
            if folder in available:
                files += [f for f in client.models(folder) if f not in files]
        out[key] = files
    return out


def _low(name: str) -> str:
    return name.lower().replace("\\", "/")


def _is_moe_half(name: str) -> bool:
    low = _low(name)
    return "high_noise" in low or "low_noise" in low


def _pick_wan(files: list[str], tag: str) -> tuple[str | None, bool]:
    """(model, only_moe): the first single-file Wan model with ``tag`` in its name."""
    tagged = [f for f in files if tag in _low(f) and "ti2v" not in _low(f)]
    single = [f for f in tagged if not _is_moe_half(f)]
    single.sort(key=lambda f: (0 if "wan" in _low(f) else 1, _low(f)))
    return (single[0] if single else None), bool(tagged) and not single


def _pick_encoder(files: list[str]) -> str | None:
    return next((f for f in sorted(files, key=_low) if "umt5" in _low(f)), None)


def _pick_vae(files: list[str]) -> str | None:
    # Wan 2.1's VAE also serves the Wan 2.2 14B models; wan2.2_vae is for the 5B only.
    wan = [f for f in sorted(files, key=_low) if "wan" in _low(f) and "2.2" not in _low(f) and "2_2" not in _low(f)]
    return wan[0] if wan else None


def _pick_clip_vision(files: list[str]) -> str | None:
    ordered = sorted(files, key=lambda f: (0 if "clip_vision_h" in _low(f) else 1, _low(f)))
    return ordered[0] if ordered else None


_VIDEO_WORDS = ("wan", "svd", "video", "hunyuan", "ltx", "mochi", "cosmos")


def _pick_checkpoint(files: list[str]) -> str | None:
    images = [f for f in sorted(files, key=_low) if not any(w in _low(f) for w in _VIDEO_WORDS)]
    return images[0] if images else None


def image_size_for(ckpt: str) -> int:
    """Square side for a checkpoint: 512 for SD 1.x, 1024 for everything newer."""
    low = _low(ckpt)
    return 512 if any(t in low for t in ("v1-5", "sd15", "sd1.5", "1.5", "v1_5")) else 1024


def _missing_nodes(object_info: dict[str, Any], mode: str) -> list[str]:
    return [n for n in REQUIRED_NODES[mode] if n not in object_info]


def _missing_nodes_reason(missing: list[str]) -> str:
    return (
        f"Your ComfyUI is missing the {', '.join(missing)} node{'s' if len(missing) > 1 else ''}; "
        "updating ComfyUI should add them."
    )


def _video_mode(object_info: dict[str, Any], models: dict[str, list[str]], mode: str) -> dict[str, Any]:
    missing = _missing_nodes(object_info, mode)
    if missing:
        return {"ok": False, "reason": _missing_nodes_reason(missing)}
    tag, what = ("t2v", "text-to-video") if mode == "video" else ("i2v", "image-to-video")
    unet, only_moe = _pick_wan(models["diffusion_models"], tag)
    if unet is None:
        if only_moe:
            return {"ok": False, "reason": (
                f"Your ComfyUI only has Wan 2.2's two-part (high and low noise) {what} models, "
                "which I can't drive yet. A single-file Wan model would work."
            )}
        return {"ok": False, "reason": (
            f"Your ComfyUI has no Wan {what} model (a file with '{tag}' in its name in models/diffusion_models)."
        )}
    encoder = _pick_encoder(models["text_encoders"])
    if encoder is None:
        return {"ok": False, "reason": "Your ComfyUI has no umt5 text encoder for Wan (in models/text_encoders)."}
    vae = _pick_vae(models["vae"])
    if vae is None:
        return {"ok": False, "reason": "Your ComfyUI has no Wan VAE (in models/vae)."}
    picked: dict[str, Any] = {"unet_name": unet, "clip_name": encoder, "vae_name": vae}
    if mode == "animate":
        clip_vision = None
        if "CLIPVisionLoader" in object_info and "CLIPVisionEncode" in object_info:
            clip_vision = _pick_clip_vision(models["clip_vision"])
        picked["clip_vision_name"] = clip_vision
    return {"ok": True, "models": picked}


def _image_mode(object_info: dict[str, Any], models: dict[str, list[str]]) -> dict[str, Any]:
    missing = _missing_nodes(object_info, "image")
    if missing:
        return {"ok": False, "reason": _missing_nodes_reason(missing)}
    ckpt = _pick_checkpoint(models["checkpoints"])
    if ckpt is None:
        return {"ok": False, "reason": "Your ComfyUI has no image model (a checkpoint in models/checkpoints)."}
    return {"ok": True, "models": {"ckpt_name": ckpt}}


def detect(client: ComfyClient) -> dict[str, Any]:
    """Read the server and work out the modes. Raises ComfyError if it can't be read."""
    object_info = client.object_info()
    models = fetch_models(client, object_info)
    return {
        "url": client.base_url,
        "checked_at": _now().isoformat(),
        "modes": {
            "video": _video_mode(object_info, models, "video"),
            "animate": _video_mode(object_info, models, "animate"),
            "image": _image_mode(object_info, models),
        },
    }


def load(client: ComfyClient, *, refresh: bool = False) -> dict[str, Any]:
    """Cached capabilities for this server (re-read after CACHE_TTL, a URL change or ``refresh``)."""
    store = JarvisStorage(STATE_STORAGE)
    cached = None if refresh else store.get(CAPS_KEY)
    if cached and cached.get("url") == client.base_url:
        try:
            fresh = _now() - datetime.fromisoformat(cached["checked_at"]) < CACHE_TTL
        except (KeyError, TypeError, ValueError):
            fresh = False
        if fresh:
            return cached
    caps = detect(client)
    store.save(CAPS_KEY, caps)
    return caps


def forget() -> None:
    """Drop the cache (e.g. after ComfyUI refused a model we picked from it)."""
    JarvisStorage(STATE_STORAGE).delete(CAPS_KEY)
