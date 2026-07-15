"""
Simple config loader for bb_metrics.

Experiment configs live OUTSIDE this repo (in your per-year data/working folders)
and are loaded by path:

    import bb_metrics
    cfg = bb_metrics.load_config("berlin2025.py")   # path to your season config
    bb_metrics.set_config(cfg)
    # other modules can fetch via bb_metrics.get_config()
"""

from typing import Any, Optional

_CFG: Optional[Any] = None


def set_config(cfg: Any) -> None:
    """Set the active config object (module or namespace)."""
    global _CFG
    _CFG = cfg


def get_config() -> Any:
    """Return the active config, or raise if unset."""
    if _CFG is None:
        raise RuntimeError("bb_metrics config not set. Call bb_metrics.set_config(cfg) first.")
    return _CFG


# UID helpers for reused tag IDs
from .uid import assign_uid, build_reuse_intervals  # noqa: E402

# External experiment-config loader (configs live outside this repo; see config/__init__.py)
from .config import load_config, default_label_config_path  # noqa: E402
