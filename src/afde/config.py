"""Config loading helpers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

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


def is_offline() -> bool:
    return os.environ.get("AFDE_OFFLINE", "0") not in ("0", "", "false", "False")


def anthropic_api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


def google_api_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY")
