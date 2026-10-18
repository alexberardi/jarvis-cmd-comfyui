"""Make videos and pictures on the user's own ComfyUI, delivered to the inbox.

- ``video``   text to video (Wan 2.x): "make a video of a fox running through snow"
- ``animate`` image to video: a photo sent in chat + "make this move"
- ``image``   text to image: "make a picture of a lighthouse at dusk"

Only the modes the server can do are offered (see comfyui_shared.capabilities,
read from ``/object_info`` and ``/models`` on first use). ``run()`` expands the
user's words into a model prompt with the node's LLM, queues the workflow and
answers at once; the ``comfyui_jobs`` agent polls ComfyUI and posts the result
(or a readable failure) to the user's inbox with a push.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Any, List

from jarvis_command_sdk import (
    CommandExample,
    CommandResponse,
    IJarvisCommand,
    IJarvisParameter,
    IJarvisSecret,
    JarvisParameter,
    JarvisSecret,
    RequestInformation,
    llm_chat_text,
)

from comfyui_shared import capabilities
from comfyui_shared.capabilities import MODE_NOUNS, MODES, image_size_for
from comfyui_shared.client import ComfyClient, ComfyError
from comfyui_shared.errors import describe_request_error, is_missing_model
from comfyui_shared.jobs import COMMAND_NAME, KEY_SECRET, URL_SECRET, save_job, settings
from comfyui_shared.log import logger
from comfyui_shared.workflows import VIDEO_SIZES, build, size_for_photo, video_length

_ORIENTATIONS = list(VIDEO_SIZES)
_KINDS = {
    "video": "text-to-video",
    "animate": "image-to-video (the user's photo is the first frame)",
    "image": "text-to-image",
}
_DEFAULT_ANIMATION = "gentle, natural motion with a slow camera push-in"
MAX_PROMPT_CHARS = 1200

READY_LATER = "I'll send it to your inbox when it's ready, usually a few minutes."


def expand_prompt(words: str, mode: str) -> str:
    """The user's words as a detailed model prompt; the words themselves if the LLM has nothing."""
    ask = (
        f"Rewrite this request as a single prompt for a {_KINDS[mode]} model. "
        "Describe the subject, "
        + ("the motion, the camera movement, " if mode != "image" else "the composition, ")
        + "the setting, the lighting and the style in one paragraph of at most 80 words. "
        "Keep everything the request asks for and add nothing that contradicts it. "
        "Reply with the prompt only, no preamble and no quotes.\n\n"
        f"Request: {words}"
    )
    reply = llm_chat_text(ask)
    text = " ".join((reply or "").split()).strip().strip('"').strip()
    if not text:
        return words
    return text[:MAX_PROMPT_CHARS]


def _offered(caps: dict[str, Any]) -> list[str]:
    return [m for m in MODES if (caps.get("modes") or {}).get(m, {}).get("ok")]


def _offer_sentence(offered: list[str]) -> str:
    labels = {"video": "videos from a description", "animate": "videos from a photo", "image": "pictures"}
    if not offered:
        return "Right now it can't make anything I know how to ask for."
    return "It can make " + " and ".join(labels[m] for m in offered) + "."


class ComfyUICommand(IJarvisCommand):
    """Generate a video or a picture on the user's ComfyUI and deliver it to the inbox."""

    @property
    def command_name(self) -> str:
        return COMMAND_NAME

    @property
    def description(self) -> str:
        return (
            "Make a short video or a picture with the user's own ComfyUI server and send it to "
            "their inbox when it's done (it takes a few minutes). Use for 'make a video of…', "
            "'animate this photo', 'bring this picture to life', 'generate an image of…', "
            "'draw me…'."
        )

    @property
    def keywords(self) -> List[str]:
        return [
            "comfyui", "make a video", "generate a video", "video of", "animate", "bring to life",
            "make a picture", "generate an image", "picture of", "image of", "draw",
        ]

    @property
    def parameters(self) -> List[IJarvisParameter]:
        return [
            JarvisParameter(
                "mode", "string", required=False,
                description=(
                    "video=a video from a description, animate=a video from a photo the user "
                    "attached, image=a still picture"
                ),
                enum_values=list(MODES),
            ),
            JarvisParameter(
                "prompt", "string", required=False,
                description="What to make, in the user's words, e.g. 'a fox running through fresh snow at dawn'",
            ),
            JarvisParameter(
                "photos", "image", required=False,
                description="The photo to animate (mode=animate)",
            ),
            JarvisParameter(
                "seconds", "int", required=False,
                description="Video length in seconds (1-5, default 3)",
            ),
            JarvisParameter(
                "orientation", "string", required=False,
                description="Video shape: landscape (default), portrait or square",
                enum_values=_ORIENTATIONS,
            ),
        ]

    @property
    def required_secrets(self) -> List[IJarvisSecret]:
        return [
            JarvisSecret(
                URL_SECRET,
                "Your ComfyUI server's address, e.g. http://192.168.1.50:8188 or a hosted ComfyUI URL",
                "integration", "string", is_sensitive=False, friendly_name="ComfyUI URL",
            ),
            JarvisSecret(
                KEY_SECRET,
                "API key for a hosted ComfyUI (sent as X-API-Key and as a Bearer token). Leave empty for a server on your network.",
                "integration", "string", required=False, friendly_name="API key",
            ),
        ]

    @property
    def rules(self) -> List[str]:
        return [
            "If the user attached a photo and wants it to move -> mode='animate'",
            "'make/generate a video of X' -> mode='video', prompt=X",
            "'make/generate/draw a picture/image of X' -> mode='image', prompt=X",
            "Pass the user's own description as prompt; don't embellish it, the command expands it",
        ]

    def generate_prompt_examples(self) -> List[CommandExample]:
        return [
            CommandExample(
                "Make a video of a fox running through fresh snow",
                {"mode": "video", "prompt": "a fox running through fresh snow"}, is_primary=True,
            ),
            CommandExample("Animate this photo so the waves move", {"mode": "animate", "prompt": "the waves move"}),
            CommandExample(
                "Generate a picture of a lighthouse at dusk",
                {"mode": "image", "prompt": "a lighthouse at dusk"},
            ),
        ]

    def generate_adapter_examples(self) -> List[CommandExample]:
        items: list[tuple[str, dict[str, Any], bool]] = [
            ("Make a video of a fox running through fresh snow", {"mode": "video", "prompt": "a fox running through fresh snow"}, True),
            ("Generate a five second video of a rocket launch", {"mode": "video", "prompt": "a rocket launch", "seconds": 5}, False),
            ("Make a vertical video of rain on a window", {"mode": "video", "prompt": "rain on a window", "orientation": "portrait"}, False),
            ("Use ComfyUI to make a clip of a cat playing piano", {"mode": "video", "prompt": "a cat playing piano"}, False),
            ("Animate this photo", {"mode": "animate"}, False),
            ("Bring this picture to life, make her wave", {"mode": "animate", "prompt": "she waves"}, False),
            ("Make the clouds in this photo move", {"mode": "animate", "prompt": "the clouds move"}, False),
            ("Generate a picture of a lighthouse at dusk", {"mode": "image", "prompt": "a lighthouse at dusk"}, False),
            ("Draw me a dragon made of autumn leaves", {"mode": "image", "prompt": "a dragon made of autumn leaves"}, False),
            ("Make an image of a cozy cabin in the snow", {"mode": "image", "prompt": "a cozy cabin in the snow"}, False),
        ]
        return [CommandExample(voice, params, is_primary) for voice, params, is_primary in items]

    def post_process_tool_call(self, args: dict[str, Any], voice_command: str) -> dict[str, Any]:
        if not args.get("mode"):
            args["mode"] = "animate" if args.get("photos") else "video"
        return args

    # ── Execution ─────────────────────────────────────────────────────────

    def run(self, request_info: RequestInformation, **kwargs: Any) -> CommandResponse:
        url, key = settings(kwargs.get("secrets"))
        if not url:
            return CommandResponse.error_response(
                "ComfyUI isn't set up yet. Add your ComfyUI URL in the ComfyUI package settings."
            )
        photos = kwargs.get("photos") or []
        mode = kwargs.get("mode") or ("animate" if photos else "video")
        if mode not in MODES:
            mode = "video"
        if mode == "animate" and not photos:
            return self._say("Send me the photo in chat with what should happen in it, and I'll animate it.")

        client = ComfyClient(url, key)
        try:
            caps = capabilities.load(client)
        except ComfyError as e:
            return CommandResponse.error_response(describe_request_error(e, client.base_url))
        mode_caps = caps["modes"][mode]
        if not mode_caps.get("ok"):
            others = [m for m in _offered(caps) if m != mode]
            return self._say(f"{mode_caps.get('reason')} {_offer_sentence(others)}".strip())

        words = " ".join(str(kwargs.get("prompt") or "").split())
        if not words and mode != "animate":
            words = " ".join(str(getattr(request_info, "voice_command", "") or "").split())
        if not words:
            words = _DEFAULT_ANIMATION if mode == "animate" else ""
        if not words:
            return self._say(f"What should the {MODE_NOUNS[mode]} show?")

        try:
            return self._submit(client, request_info, mode, mode_caps["models"], words, photos, kwargs)
        except ComfyError as e:
            if e.kind == "rejected" and is_missing_model(e.body):
                capabilities.forget()  # the cached model list is stale; re-read it next time
            logger.warning("comfyui: submit failed", kind=e.kind, status=e.status)
            return CommandResponse.error_response(describe_request_error(e, client.base_url))

    def _submit(
        self,
        client: ComfyClient,
        request_info: RequestInformation,
        mode: str,
        models: dict[str, Any],
        words: str,
        photos: list[Any],
        kwargs: dict[str, Any],
    ) -> CommandResponse:
        prompt = expand_prompt(words, mode)
        seed = random.randint(0, 2**32 - 1)
        job_id = uuid.uuid4().hex
        seconds, length = video_length(kwargs.get("seconds"))
        image_name = None
        if mode == "image":
            side = image_size_for(models["ckpt_name"])
            width, height = side, side
        elif mode == "animate":
            photo = photos[0]
            width, height = size_for_photo(photo.width, photo.height)
            ext = {"image/png": "png", "image/webp": "webp"}.get(photo.mime, "jpg")
            image_name = client.upload_image(photo.data, f"jarvis_{job_id}.{ext}", photo.mime)
        else:
            width, height = VIDEO_SIZES.get(kwargs.get("orientation") or "landscape", VIDEO_SIZES["landscape"])

        workflow = build(
            mode, models, prompt=prompt, seed=seed, width=width, height=height,
            length=length, image=image_name, filename_prefix=f"jarvis/{mode}_{job_id[:8]}",
        )
        prompt_id = client.queue_prompt(workflow, client_id=f"jarvis-{job_id}")
        save_job({
            "prompt_id": prompt_id,
            "mode": mode,
            "request": words,
            "prompt": prompt,
            "seed": seed,
            "user_id": getattr(request_info, "user_id", None),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("comfyui: job queued", prompt_id=prompt_id, mode=mode)
        noun = MODE_NOUNS[mode]
        what = f"a {seconds}-second {noun}" if mode != "image" else f"your {noun}"
        return CommandResponse.success_response(
            context_data={
                "message": f"I'm making {what} on your ComfyUI. {READY_LATER}",
                "mode": mode,
                "prompt": prompt,
            },
            wait_for_input=False,
        )

    @staticmethod
    def _say(message: str) -> CommandResponse:
        return CommandResponse.success_response(context_data={"message": message}, wait_for_input=False)
