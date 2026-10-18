# jarvis-cmd-comfyui

A Jarvis command package: videos and pictures from the user's own ComfyUI,
delivered to the inbox. Read `README.md` and `docs/decisions/` first.

## Rules

- Commands never start threads or sleep (SDK rule). `run()` queues a workflow and
  returns; the `comfyui_jobs` agent does the waiting. Keep it that way.
- stdlib HTTP only (`urllib`); `packages: []` in the manifest.
- Shared code lives in `comfyui_shared/` (a generic name like `shared/` collides
  between installed packages; `jdt test` rejects it).
- Every failure the user can hit ends as a readable sentence (`comfyui_shared/errors.py`):
  spoken at submit time, an inbox item after that.
- Workflow templates are ComfyUI "Save (API)" JSON; placeholders are whole string
  values `"{{name}}"`. A template change needs a test in `tests/test_workflows.py`.

## Test

```bash
.venv/bin/pytest -q          # unit + end-to-end against tests/fake_comfyui.py
.venv/bin/jdt test .         # Pantry checks (manifest, static analysis, imports)
```

New behaviour gets a test against the fake server first; extend the fake to
match real ComfyUI responses rather than mocking the client.
