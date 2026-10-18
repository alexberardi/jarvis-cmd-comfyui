"""jarvis_package.yaml and the command declare the same secrets."""

from pathlib import Path

import yaml

from commands.comfyui.command import ComfyUICommand

MANIFEST = Path(__file__).resolve().parent.parent / "jarvis_package.yaml"


def _shape(key, required, is_sensitive, friendly_name, scope, value_type):
    return {"key": key, "required": required, "is_sensitive": is_sensitive,
            "friendly_name": friendly_name, "scope": scope, "value_type": value_type}


def test_manifest_secrets_match_command():
    manifest = yaml.safe_load(MANIFEST.read_text())
    declared = [
        _shape(s["key"], s["required"], s["is_sensitive"], s["friendly_name"], s["scope"], s["value_type"])
        for s in manifest["secrets"]
    ]
    code = [
        _shape(s.key, s.required, s.is_sensitive, s.friendly_name, s.scope, s.value_type)
        for s in ComfyUICommand().required_secrets
    ]
    assert declared == code


def test_api_key_optional_url_required():
    by_key = {s["key"]: s for s in yaml.safe_load(MANIFEST.read_text())["secrets"]}
    assert by_key["COMFYUI_URL"]["required"] is True
    assert by_key["COMFYUI_API_KEY"]["required"] is False
