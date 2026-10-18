"""Shared test fixtures: in-memory storage, inbox and LLM backends, and a fake
ComfyUI server (tests/fake_comfyui.py) the client talks to over real HTTP."""

import pytest
from jarvis_command_sdk.inbox import InboxBackend, get_inbox_backend, set_inbox_backend
from jarvis_command_sdk.llm import LLMBackend, get_llm_backend, set_llm_backend
from jarvis_command_sdk.storage import StorageBackend, get_backend, set_backend

from tests.fake_comfyui import FakeComfyUI


class FakeStorageBackend(StorageBackend):
    """In-memory (command_name, key) -> dict store. Copies on read/write so
    callers can't mutate stored records by reference (matches the real DB)."""

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], dict] = {}
        self.secrets: dict[tuple[str, str, int | None], str] = {}

    def save(self, command_name, data_key, data, expires_at=None):
        self.data[(command_name, data_key)] = dict(data)

    def get(self, command_name, data_key):
        rec = self.data.get((command_name, data_key))
        return dict(rec) if rec is not None else None

    def get_all(self, command_name):
        return [dict(v) for (c, _k), v in self.data.items() if c == command_name]

    def delete(self, command_name, data_key):
        return self.data.pop((command_name, data_key), None) is not None

    def delete_all(self, command_name):
        keys = [(c, k) for (c, k) in self.data if c == command_name]
        for key in keys:
            del self.data[key]
        return len(keys)

    def get_secret(self, key, scope, user_id=None):
        return self.secrets.get((key, scope, user_id))

    def set_secret(self, key, value, scope, value_type="string", user_id=None):
        self.secrets[(key, scope, user_id)] = value

    def delete_secret(self, key, scope, user_id=None):
        self.secrets.pop((key, scope, user_id), None)


@pytest.fixture
def backend():
    prev = get_backend()
    fake = FakeStorageBackend()
    set_backend(fake)
    yield fake
    set_backend(prev)  # type: ignore[arg-type]


class FakeInboxBackend(InboxBackend):
    """Captures posted inbox items (with their attachments). ``tag`` is what post returns."""

    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.tag = "ok"

    def post_inbox_item(
        self, command_name, *, title, summary="", body="", category="general",
        metadata=None, user_id=None, create_push_notification=False, target_type="household",
        attachments=None,
    ):
        self.posts.append({
            "command_name": command_name, "title": title, "summary": summary,
            "body": body, "category": category, "metadata": metadata,
            "user_id": user_id, "push": create_push_notification, "target_type": target_type,
            "attachments": list(attachments or []),
        })
        return self.tag


class OldInboxBackend(InboxBackend):
    """A node backend from before SDK 0.13.0: no ``attachments`` parameter."""

    def __init__(self) -> None:
        self.posts: list[dict] = []

    def post_inbox_item(  # type: ignore[override]
        self, command_name, *, title, summary="", body="", category="general",
        metadata=None, user_id=None, create_push_notification=False, target_type="household",
    ):
        self.posts.append({"title": title, "summary": summary, "body": body, "user_id": user_id})
        return "ok"


def _swap_inbox(fake):
    prev = get_inbox_backend()
    set_inbox_backend(fake)
    return prev


@pytest.fixture
def inbox():
    fake = FakeInboxBackend()
    prev = _swap_inbox(fake)
    yield fake
    set_inbox_backend(prev)  # type: ignore[arg-type]


@pytest.fixture
def old_inbox():
    fake = OldInboxBackend()
    prev = _swap_inbox(fake)
    yield fake
    set_inbox_backend(prev)  # type: ignore[arg-type]


class FakeLLM(LLMBackend):
    def __init__(self) -> None:
        self.reply: str | None = "A red fox sprints through deep fresh snow at dawn, cinematic tracking shot."
        self.prompts: list[str] = []

    def chat_text(self, prompt):
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture
def llm():
    prev = get_llm_backend()
    fake = FakeLLM()
    set_llm_backend(fake)
    yield fake
    set_llm_backend(prev)


@pytest.fixture
def comfy(monkeypatch):
    # The client honours *_proxy env vars like any urllib caller; the fake is local.
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    server = FakeComfyUI().start()
    yield server
    server.stop()


@pytest.fixture
def configured(backend, comfy):
    """Package settings pointing at the fake server, no API key (a LAN server)."""
    backend.secrets[("COMFYUI_URL", "integration", None)] = comfy.url
    return comfy
