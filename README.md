# jarvis-cmd-comfyui

Make short videos and pictures with **your own ComfyUI** and get them in your
[Jarvis](https://github.com/alexberardi/jarvis-node-setup) inbox.

Any ComfyUI works: the GPU box on your network or a hosted ComfyUI. Jarvis
bundles and runs nothing; it sends workflows to your server and delivers what
comes back.

- "Make a video of a fox running through fresh snow" → text to video (Wan 2.x)
- a photo in chat + "animate this, make the waves move" → photo to video (Wan 2.x)
- "Generate a picture of a lighthouse at dusk" → text to image (any checkpoint)

Jarvis answers at once ("I'll send it to your inbox when it's ready, usually a
few minutes"), and the result arrives in the inbox with a push. If something
goes wrong (the server is down, it ran out of GPU memory, a model is missing)
you get a readable failure in the inbox instead.

## Install

```bash
python scripts/command_store.py install --url https://github.com/alexberardi/jarvis-cmd-comfyui
```

Needs a node with jarvis-command-sdk **0.13.0** or newer (inbox attachments).
An older node still gets an "it's ready" item, without the file.

## Settings

| Setting | | |
|---|---|---|
| `COMFYUI_URL` | required | e.g. `http://192.168.1.50:8188`, or a hosted ComfyUI's base URL |
| `COMFYUI_API_KEY` | optional, secret | sent as `X-API-Key: <key>` and `Authorization: Bearer <key>`; leave empty for a LAN server |

## What your ComfyUI needs

On first use the package reads `/object_info` and `/models` and offers only
what the server can do (re-read every 6 hours, when a model it picked is
gone, and before saying a mode isn't available, so a model you just installed
is picked up at once):

| Mode | Models | Nodes |
|---|---|---|
| text to video | a Wan `*t2v*` model in `diffusion_models`, a `umt5` text encoder, a Wan VAE | core Wan nodes + `CreateVideo`/`SaveVideo` |
| photo to video | a Wan `*i2v*` model, the same encoder and VAE (`clip_vision_h` used if present) | + `WanImageToVideo` |
| text to image | any checkpoint in `checkpoints` | core |

Videos are 1–5 s (default 3) at 16 fps, 832×480 landscape (or portrait/square),
saved as mp4. Wan 2.2's two-part (high/low noise) 14B models and the 5B
`ti2v` model are not driven yet.

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `comfyui` | command | Checks what the server can do, expands the prompt with the LLM, queues the workflow, answers at once |
| `comfyui_jobs` | agent | Every 20 s: polls `/history`, downloads the result, posts it to the inbox (or a readable failure); 30-min timeout |

## Layout

```
commands/comfyui/command.py        # voice/chat surface, queues the job
agents/comfyui_jobs/agent.py       # delivers finished jobs to the inbox
comfyui_shared/
  client.py                        # stdlib HTTP client for the ComfyUI API
  capabilities.py                  # /object_info + /models → offered modes (cached)
  workflows.py                     # template loading and filling
  workflows/*.json                 # API-format templates (wan_t2v, wan_i2v, sd_t2i)
  jobs.py                          # job records, polling, inbox delivery
  errors.py                        # ComfyUI failures as readable sentences
tests/                             # against a fake ComfyUI server (tests/fake_comfyui.py)
docs/decisions/                    # design decisions
```

## Tests

```bash
.venv/bin/pytest -q
.venv/bin/jdt test .
```
