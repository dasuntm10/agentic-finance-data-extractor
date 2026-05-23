"""Config loading helpers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Optional .env loading — only effective if python-dotenv is installed and a
# .env file is present. Used for non-secret tunables like AFDE_LLM_BASE_URL.
try:
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"


@lru_cache(maxsize=4)
def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_labels() -> dict[str, Any]:
    return load_yaml("canonical_labels.yaml")


def scoring_config() -> dict[str, Any]:
    return load_yaml("scoring.yaml")


def llm_base_url() -> str:
    return os.environ.get("AFDE_LLM_BASE_URL", "http://localhost:11434")


def llm_model() -> str:
    return os.environ.get("AFDE_LLM_MODEL", "qwen2.5:7b-instruct")
