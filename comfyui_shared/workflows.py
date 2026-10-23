"""API-format workflow templates and how they are filled in.

Templates live in ``workflows/*.json`` (ComfyUI's "Save (API)" format). A
placeholder is a whole string value ``"{{name}}"``, replaced by a typed value
(so ``"{{seed}}"`` becomes an int). Swapping a template for another one with
the same placeholders needs no code change.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

TEMPLATE_DIR = Path(__file__).parent / "workflows"
TEMPLATES = {"video": "wan_t2v.json", "animate": "wan_i2v.json", "image": "sd_t2i.json"}

# The Wan templates' model loader, and its stand-in for GGUF files (ComfyUI-GGUF),
# which takes no weight_dtype.
UNET_NODE = "1"
GGUF_LOADER = "UnetLoaderGGUF"

# The i2v template's CLIP-vision branch, dropped when the server has no CLIP vision model.
CLIP_VISION_NODES = ("13", "14")

FPS = 16
DEFAULT_SECONDS = 3
MAX_SECONDS = 5
VIDEO_SIZES = {"landscape": (832, 480), "portrait": (480, 832), "square": (640, 640)}
VIDEO_LONG_SIDE = 832

DEFAULT_NEGATIVE = (
    "blurry, low quality, low resolution, distorted, deformed, extra limbs, "
    "watermark, text, subtitles, jpeg artifacts, static, frozen frame"
)


def load_template(mode: str) -> dict[str, Any]:
    return json.loads((TEMPLATE_DIR / TEMPLATES[mode]).read_text())


def fill(template: Any, values: dict[str, Any]) -> Any:
    """A copy of ``template`` with every ``"{{name}}"`` replaced. KeyError if a value is missing."""
    if isinstance(template, dict):
        return {k: fill(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [fill(v, values) for v in template]
    if isinstance(template, str) and template.startswith("{{") and template.endswith("}}"):
        return values[template[2:-2].strip()]
    return template


def drop_nodes(workflow: dict[str, Any], node_ids: tuple[str, ...]) -> dict[str, Any]:
    """A copy without ``node_ids`` and without any input linked to them."""
    out = copy.deepcopy(workflow)
    for node_id in node_ids:
        out.pop(node_id, None)
    for node in out.values():
        inputs = node.get("inputs") or {}
        for name in [n for n, v in inputs.items() if isinstance(v, list) and v and v[0] in node_ids]:
            del inputs[name]
    return out


def video_length(seconds: Any) -> tuple[int, int]:
    """(seconds, frames) — clamped to 1..MAX_SECONDS; Wan wants 4k+1 frames."""
    try:
        secs = int(round(float(seconds)))
    except (TypeError, ValueError):
        secs = DEFAULT_SECONDS
    secs = max(1, min(MAX_SECONDS, secs))
    return secs, FPS * secs + 1


def _round16(x: float) -> int:
    return max(256, int(round(x / 16.0)) * 16)


def size_for_photo(width: int | None, height: int | None) -> tuple[int, int]:
    """Video size keeping the photo's aspect, long side VIDEO_LONG_SIDE, multiples of 16."""
    if not width or not height:
        return VIDEO_SIZES["landscape"]
    scale = VIDEO_LONG_SIDE / max(width, height)
    return _round16(width * scale), _round16(height * scale)


def build(
    mode: str,
    models: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    length: int = 0,
    image: str | None = None,
    negative: str = DEFAULT_NEGATIVE,
    filename_prefix: str = "jarvis/comfyui",
) -> dict[str, Any]:
    template = load_template(mode)
    if mode == "animate" and not models.get("clip_vision_name"):
        template = drop_nodes(template, CLIP_VISION_NODES)
    if str(models.get("unet_name") or "").lower().endswith(".gguf"):
        template[UNET_NODE] = {"class_type": GGUF_LOADER, "inputs": {"unet_name": "{{unet_name}}"}}
    values: dict[str, Any] = {
        **{k: v for k, v in models.items() if v is not None},
        "prompt": prompt,
        "negative": negative,
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "length": int(length),
        "fps": float(FPS),
        "filename_prefix": filename_prefix,
    }
    if image is not None:
        values["image"] = image
    return fill(template, values)
