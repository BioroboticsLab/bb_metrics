"""Find and read shared, versioned tables under saved_output/, and locate question output folders.

Each experiment keeps hand-entered records (layer 0) and shared results (layer 1) in their own
folder under ``<basedir>/saved_output/``. A table gets a new version instead of being overwritten,
in one of three layouts:

    bees/LATEST + bees/v2/lifetime_reference_v2.parquet    LATEST names the current version
    <folder>/v1/, v2/ ... with <name>_vN.<ext> inside        no LATEST: highest vN
    treatment/treat_df.csv, treatment/treat_df_v2.csv        files side by side: highest _vN
                                                             (the unversioned file counts as v1)

"Newest" is the version number, never the file date. At equal version the extension order in
``EXTS`` decides (parquet before csv).

Layer 2 (questions) output never goes to saved_output/. It goes to
``<basedir>/questions_output/<question>/`` (see ``questions_dir``).

Where saved_output/ is: ``cfg.saved_output_dir`` if a config is passed, else the active
bb_metrics config (``bb_metrics.set_config``), else ``$BB_DATA_ROOT/saved_output``, else the first
existing path in ``candidates`` + ``/saved_output``. Mount points are experiment-specific, so
experiment repos pass ``candidates``; this public module holds no storage paths.

    from bb_metrics import saved_output as so
    tags = so.load("dftags", "tags")                       # newest version
    path = so.find("dftags", "tags", missing_ok=True)      # path, or None
    old = so.load("treat_df", "treatment", version="v1")
    v, d = so.next_version("bees")                         # ('v3', .../bees/v3), created
    so.set_latest("bees", v)
    out = so.questions_dir("hygienic_paper", "figdata", create=True)
"""
import os
import re
from pathlib import Path

import pandas as pd

EXTS = (".parquet", ".csv", ".xlsx", ".json")   # order = preference at equal version


def saved_dir(cfg=None, candidates=None):
    """saved_output/ as a Path. Never raises, so it can set module constants; a wrong root shows
    up on the first file access."""
    if cfg is None:
        try:
            from . import get_config
            cfg = get_config()
        except RuntimeError:
            cfg = None
    if cfg is not None and getattr(cfg, "saved_output_dir", None):
        return Path(cfg.saved_output_dir)
    if os.environ.get("BB_DATA_ROOT"):
        return Path(os.environ["BB_DATA_ROOT"]) / "saved_output"
    candidates = [Path(p) for p in (candidates or [])]
    if not candidates:
        raise RuntimeError("saved_output: no config set, no BB_DATA_ROOT and no candidates given")
    root = next((p for p in candidates if p.exists()), candidates[0])
    return root / "saved_output"


def questions_dir(question, *sub, cfg=None, candidates=None, create=False):
    """<basedir>/questions_output/<question>[/<sub>...], the place for layer 2 output.
    basedir is the parent of saved_output/."""
    d = saved_dir(cfg, candidates).parent / "questions_output" / question
    for s in sub:
        d = d / s
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _vnum(v):
    m = re.fullmatch(r"v(\d+)", str(v))
    if not m:
        raise ValueError(f"version must look like 'v2', got {v!r}")
    return int(m.group(1))


def _version_dirs(base):
    if not base.is_dir():
        return {}
    return {_vnum(p.name): p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)}


def _file_in(d, name, v, exts):
    """<name>_<v>.<ext> (or <name>.<ext>) in folder d, in extension preference order."""
    for stem in (f"{name}_{v}", name):
        for ext in exts:
            if (d / f"{stem}{ext}").exists():
                return d / f"{stem}{ext}"
    return None


def find(name, folder, cfg=None, version=None, verbose=True, missing_ok=False,
         candidates=None, exts=EXTS):
    """Path of the newest version (or `version`, e.g. 'v1') of table `name` in saved_output/<folder>/.
    missing_ok=True returns None instead of raising when there is none."""
    base = saved_dir(cfg, candidates) / folder
    if not base.is_dir():
        if missing_ok:
            return None
        raise FileNotFoundError(f"no folder {base}")
    subdirs = _version_dirs(base)
    path = None
    if subdirs:
        # version folders: pinned, else LATEST, else the highest vN
        if version is not None:
            v = version
        elif (base / "LATEST").exists():
            v = (base / "LATEST").read_text().strip()
        else:
            v = f"v{max(subdirs)}"
        if _vnum(v) in subdirs:
            path = _file_in(subdirs[_vnum(v)], name, v, exts)
    else:
        # files side by side: <name>.<ext> = v1, <name>_vN.<ext> = vN
        pat = re.compile(rf"{re.escape(name)}(?:_v(\d+))?({'|'.join(map(re.escape, exts))})")
        found = []
        for p in base.iterdir():
            m = pat.fullmatch(p.name)
            if m and p.is_file():
                found.append((int(m.group(1) or 1), -exts.index(m.group(2)), p))
        if version is not None:
            found = [f for f in found if f[0] == _vnum(version)]
        if found:
            path = max(found)[2]
    if path is None:
        if missing_ok:
            return None
        have = sorted(p.name for p in base.iterdir())
        raise FileNotFoundError(f"no table {name!r}{f' {version}' if version else ''} in {base}; it holds: {have}")
    if verbose:
        print(f"{name}: {path}")
    return path


def read(path, **read_kw):
    """Read a table file by its extension (parquet, csv, xlsx, json)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **read_kw)
    if path.suffix == ".csv":
        return pd.read_csv(path, **read_kw)
    if path.suffix == ".xlsx":
        return pd.read_excel(path, **read_kw)
    if path.suffix == ".json":
        import json
        with open(path) as f:
            return json.load(f, **read_kw)
    raise ValueError(f"cannot read {path}")


def load(name, folder, cfg=None, version=None, verbose=True, candidates=None, exts=EXTS, **read_kw):
    """The table found by find(), read by its extension (extra keywords go to the reader).
    A .json file is returned as the parsed object, not a DataFrame."""
    return read(find(name, folder, cfg, version, verbose, candidates=candidates, exts=exts), **read_kw)


def next_version(folder, cfg=None, candidates=None, create=True):
    """('vN', path) for the next version folder saved_output/<folder>/vN/ (v1 if none). LATEST is
    not changed; call set_latest once the version is complete."""
    base = saved_dir(cfg, candidates) / folder
    n = max(_version_dirs(base), default=0) + 1
    d = base / f"v{n}"
    if create:
        d.mkdir(parents=True, exist_ok=False)
    return f"v{n}", d


def set_latest(folder, version, cfg=None, candidates=None):
    """Point saved_output/<folder>/LATEST at `version` (which must exist)."""
    base = saved_dir(cfg, candidates) / folder
    if not (base / version).is_dir():
        raise FileNotFoundError(f"no version folder {base / version}")
    _vnum(version)
    (base / "LATEST").write_text(version + "\n")
    return base / "LATEST"
