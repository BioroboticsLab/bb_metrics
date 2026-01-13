"""
Simple config loader for bb_metrics.

Usage in notebooks/scripts:
    from bb_metrics.config import berlin2025 as cfg
    import bb_metrics
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
