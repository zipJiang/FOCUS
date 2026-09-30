"""Optional OmegaConf helpers for loading experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        msg = "install focus[config] to load OmegaConf YAML configs"
        raise RuntimeError(msg) from exc

    raw = OmegaConf.load(path)
    loaded = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError("top-level config must be a mapping")
    return dict(loaded)


__all__ = [
    "load_yaml",
]
