from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


@lru_cache(maxsize=1)
def initialize_hf_environment() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    env_values = _parse_env_file(root / ".env")
    for key, value in env_values.items():
        os.environ.setdefault(key, value)

    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )

    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    hf_home = os.getenv("HF_HOME") or env_values.get("HF_HOME")
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)

    transformers_cache = os.getenv("TRANSFORMERS_CACHE") or env_values.get("TRANSFORMERS_CACHE")
    if transformers_cache:
        os.environ.setdefault("TRANSFORMERS_CACHE", transformers_cache)

    return {
        "token": token,
        "hf_home": os.getenv("HF_HOME"),
        "transformers_cache": os.getenv("TRANSFORMERS_CACHE"),
    }


def get_hf_token() -> str | None:
    return initialize_hf_environment()["token"]


def hf_hub_download_with_auth(repo_id: str, filename: str) -> str:
    initialize_hf_environment()
    from huggingface_hub import hf_hub_download

    kwargs: dict[str, Any] = {}
    token = get_hf_token()
    if token:
        kwargs["token"] = token
    return hf_hub_download(repo_id, filename, **kwargs)


def snapshot_download_with_auth(repo_id: str) -> str:
    initialize_hf_environment()
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "resume_download": True,
        "max_workers": 16,
    }
    token = get_hf_token()
    if token:
        kwargs["token"] = token
    return snapshot_download(repo_id, **kwargs)
