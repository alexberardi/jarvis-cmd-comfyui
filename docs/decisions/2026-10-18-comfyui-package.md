# 2026-10-18 — ComfyUI package (piece 5 of the ComfyUI plan)

## Context

The user asked (2026-10-17, low priority) for media made on their own ComfyUI
and delivered through the inbox. Pieces 1–4 have landed: inbox attachments in
jarvisd (jarvis-server `2026-10-17-inbox-attachments.md`), the mobile display
(#185), the SDK's `JarvisInbox.post(attachments=...)` (SDK 0.13.0) and the
Python node's pass-through (jarvis-node-setup `2026-10-18-inbox-attachments-node.md`).
This is piece 5: the package.

## Decisions

1. **Async without threads: queue in the command, poll in an agent.**
   The SDK forbids commands from starting threads or sleeping and has no job
   API. `run()` posts the workflow (`POST /prompt` answers in well under a
   second), saves `{prompt_id, mode, request, prompt, seed, user_id,
   submitted_at}` in `JarvisStorage("comfyui_jobs")` and replies at once. The
   `comfyui_jobs` agent (every 20 s, `run_on_startup`) polls
   `GET /history/{prompt_id}`, downloads the output through `/view` and posts the
   inbox item with the file attached, a push, `target_type="user"` (household if
   the request had no user). Jobs survive node restarts. A tick with no jobs
   makes no request. **No websocket**: `/history` polling is enough for
   minute-long jobs, works through any proxy and needs no long-lived connection.
2. **Timeout 30 minutes** from submission: a job still queued, or a server
   unreachable for that long, becomes a failure item. An auth failure while
   polling fails the job at once. A failed inbox post (`http_error`, …) is
   retried on the next ticks, five attempts, then dropped with an error log.
3. **Errors.** At submit time the user is in the conversation, so server down,
   a refused key, a missing model or node (`/prompt` 400 `node_errors`) are
   **spoken**, and nothing is queued. After that every failure is an **inbox
   item** ("Couldn't make your video" + the reason): out of memory
   (`OutOfMemoryError` / "CUDA out of memory"), any other `execution_error`
   (node + message), cancelled, timed out, lost the server, finished with no
   media, too large for the inbox.
4. **Capabilities from `/object_info` + `/models`**, cached in
   `JarvisStorage("comfyui_state")` for 6 hours per URL; dropped when `/prompt`
   says a picked model is not in its list, so the next request re-reads.
   Before refusing a mode the cache says is unavailable, the command re-reads
   the server once, so a model installed after a "no" works straight away.
   Servers without `/models` fall back to the loader nodes' combo lists in
   `/object_info` (both the old `[[...]]` and the `["COMBO", {"options"}]` shapes).
   A mode is offered only when its nodes and models are there; otherwise the
   reply says what is missing and what the server can make instead.
5. **Model choice by filename**: `*t2v*` / `*i2v*` in diffusion_models (legacy
   `unet` too), `umt5*` text encoder, a `wan` VAE that is not the 2.2 (5B) one,
   `clip_vision_h` when present (the i2v template drops its CLIP-vision branch
   otherwise), the first non-video checkpoint for images (512 px for SD 1.x,
   1024 otherwise). Wan 2.2's two-part high/low-noise 14B models and the 5B
   `ti2v` model are **not driven in v1** (they need different graphs); the reply
   says so instead of making a bad video.
   **GGUF Wan (2026-10-22, #516)**: with ComfyUI-GGUF's `UnetLoaderGGUF` in
   `/object_info`, `unet_gguf` is read too and searched after diffusion_models
   (safetensors first, the least surprising pick). A `.gguf` model gets
   `UnetLoaderGGUF` (only `unet_name`) as the template's node "1"; the choice is
   made from the file extension at build time, so capabilities cached before
   this change still build. The text encoder stays on `CLIPLoader` (GGUF
   encoders are not read). A tag only counts as a word start when names hold
   both (`wan22RemixT2VI2V_i2vHighV30` is image-to-video). Unknown: whether
   "Remix ...High" files are high-noise halves of a two-part model, which
   `_is_moe_half` (only `high_noise`/`low_noise`) would not catch.
6. **Templates** are ComfyUI API-format JSON in `comfyui_shared/workflows/`
   (`wan_t2v`, `wan_i2v`, `sd_t2i`), core nodes only. Placeholders are whole
   string values (`"{{seed}}"`) replaced by typed values, so a template can be
   swapped for a hand-tuned one with the same placeholders. Video ends in
   `CreateVideo` → `SaveVideo` (mp4, h264): mp4 plays on both phone platforms
   and is in jarvisd's allowed types. 16 fps, 1–5 s (16·s+1 frames), 832×480 /
   480×832 / 640×640; a photo keeps its aspect (long side 832, multiples of 16).
7. **Prompt expansion** with `llm_chat_text`; on `None` (no LLM, a failure) the
   user's words are used as they are. The negative prompt is fixed.
8. **Auth**: with a key, every request sends both `X-API-Key: <key>` (Comfy
   Cloud) and `Authorization: Bearer <key>` (most other hosts and auth
   proxies); with none, nothing (LAN). One setting, no header picker. Not checked
   against a real hosted service.
9. **Inbox fit**: the file is checked against the SDK's 25 MiB cap before the
   post; over it, the failure item names the file on the ComfyUI server.
   `attachments_unsupported` (a node from before SDK 0.13.0) gets a text-only
   "ready" item naming the file. `min_sdk_version: 0.13.0` in the manifest.
10. **stdlib only** (`urllib`, a hand-built multipart body for `/upload/image`):
    no package dependencies.

## Not done / open

- No real ComfyUI was available to the build; the templates follow ComfyUI's
  core Wan 2.1 examples and are tested against a fake server only.
- Hosted-service auth headers are unverified (see 8).
- Downloads run inside the agent's tick (blocking, 120 s timeout), like other
  packages' agents do their HTTP.
