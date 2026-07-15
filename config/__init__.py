"""
bb_metrics.config — loader for EXTERNAL experiment configs.

Experiment configs (berlin2025.py, konstanz2025.py, ...) are NOT stored in this
public repo. They live in your per-year data/working folders. Load one by path:

    import bb_metrics
    cfg = bb_metrics.load_config("berlin2026.py")            # relative to cwd
    cfg = bb_metrics.load_config("/path/to/data/berlin2026.py")   # or an absolute path
    bb_metrics.set_config(cfg)

The shared, non-secret label_classes.json still ships with this package; configs
resolve it via default_label_config_path() below.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_CACHE: dict[str, ModuleType] = {}
_SYS_PREFIX = "bb_metrics._configs"  # private; never masquerade as bb_metrics.config.<name>


def default_label_config_path() -> Path:
    """Path to the shared label_classes.json bundled with this package."""
    return Path(__file__).parent / "label_classes.json"


def load_config(path, reload: bool = False) -> ModuleType:
    """Load a config .py file as a module and return it (cached by resolved path).

    `path` may be absolute or relative-to-cwd; `~` is expanded. Pass reload=True
    to re-exec after editing a config in a live kernel.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    key = str(p)
    if not reload and key in _CACHE:
        return _CACHE[key]
    if not p.is_file():
        raise FileNotFoundError(
            f"bb_metrics config not found: {p}\n"
            f"  Pass the path to your season config .py "
            f"(template: bb_metrics/config/example_config.py)."
        )
    mod_name = f"{_SYS_PREFIX}.{p.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # register before exec (self-reference safety)
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keep the data folders free of __pycache__
    try:
        spec.loader.exec_module(module)  # runs the config top-to-bottom
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    finally:
        sys.dont_write_bytecode = prev
    _CACHE[key] = module
    return module


def __getattr__(name: str):  # helpful error for any missed old-style import
    if name.startswith("__"):
        raise AttributeError(name)
    raise AttributeError(
        f"bb_metrics.config.{name} was moved out of this repo. "
        f"Load it by path: bb_metrics.load_config('/path/to/{name}.py')."
    )
