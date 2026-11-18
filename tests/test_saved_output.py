import json
from types import SimpleNamespace

import pandas as pd
import pytest

from bb_metrics import saved_output as so


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "saved_output").mkdir()
    return SimpleNamespace(saved_output_dir=str(tmp_path / "saved_output"))


def _csv(path, val):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"a": [val]}).to_csv(path, index=False)


def test_side_by_side_highest_version(cfg):
    base = so.saved_dir(cfg) / "treatment"
    _csv(base / "treat_df.csv", 1)
    _csv(base / "treat_df_v2.csv", 2)
    assert so.load("treat_df", "treatment", cfg, verbose=False).a[0] == 2
    assert so.load("treat_df", "treatment", cfg, version="v1", verbose=False).a[0] == 1


def test_version_dirs_latest_wins(cfg):
    base = so.saved_dir(cfg) / "bees"
    _csv(base / "v1" / "bees_v1.csv", 1)
    _csv(base / "v2" / "bees_v2.csv", 2)
    assert so.load("bees", "bees", cfg, verbose=False).a[0] == 2
    so.set_latest("bees", "v1", cfg)
    assert so.load("bees", "bees", cfg, verbose=False).a[0] == 1


def test_parquet_before_csv(cfg):
    base = so.saved_dir(cfg) / "tags"
    _csv(base / "dftags.csv", 1)
    pd.DataFrame({"a": [9]}).to_parquet(base / "dftags.parquet")
    assert so.find("dftags", "tags", cfg, verbose=False).suffix == ".parquet"


def test_json_and_missing(cfg):
    base = so.saved_dir(cfg) / "tags"
    base.mkdir()
    (base / "params.json").write_text(json.dumps({"x": 1}))
    assert so.load("params", "tags", cfg, verbose=False) == {"x": 1}
    assert so.find("nope", "tags", cfg, missing_ok=True) is None
    assert so.find("nope", "nofolder", cfg, missing_ok=True) is None
    with pytest.raises(FileNotFoundError):
        so.find("nope", "tags", cfg)


def test_next_version(cfg):
    v, d = so.next_version("free_tags", cfg)
    assert v == "v1" and d.is_dir()
    v, d = so.next_version("free_tags", cfg)
    assert v == "v2"


def test_candidates_and_questions_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("BB_DATA_ROOT", raising=False)
    import bb_metrics
    monkeypatch.setattr(bb_metrics, "_CFG", None)
    root = tmp_path / "share"
    root.mkdir()
    assert so.saved_dir(candidates=[tmp_path / "missing", root]) == root / "saved_output"
    q = so.questions_dir("hygienic_paper", "figdata", candidates=[root], create=True)
    assert q == root / "questions_output" / "hygienic_paper" / "figdata" and q.is_dir()
    with pytest.raises(RuntimeError):
        so.saved_dir()
