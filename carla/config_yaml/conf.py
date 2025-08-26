# carla/conf.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf, DictConfig

DEFAULT_CFG_PATH = Path(__file__).with_name("config.yaml")

def load_cfg(path: Optional[str] = None,
             overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
    """
    Load CARLA manager config from YAML and merge optional overrides.
    """
    base = OmegaConf.load(path or DEFAULT_CFG_PATH)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.create(overrides))
    _validate(base)
    return base

def _validate(cfg: DictConfig) -> None:
    if not (0 < int(cfg.port) < 65536):
        raise ValueError(f"Invalid port: {cfg.port}")
    if float(cfg.timeout) <= 0:
        raise ValueError(f"timeout must be > 0 (got {cfg.timeout})")
    # paths validated later (OS-specific) in the manager
