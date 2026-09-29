"""Exact lifespan posteriors and tag-status calls (bb_metrics.lifespan), synthetic data only.

Covers the closed form against numerical integration, masking, right-censoring, the
joint birth-death model, the legacy series construction against metrics_pipeline (with
the PyMC fit replaced by the exact posterior, so no sampling), uid generations against
uid.build_reuse_intervals, twin-aware binarization, and a command-line smoke run.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, norm

from bb_metrics import lifespan as ls
from bb_metrics import uid as uidmod

_G = (np.arange(3000) + 0.5) / 3000
_WH = beta.pdf(_G, 5, 1) / beta.pdf(_G, 5, 1).sum()
_WL = beta.pdf(_G, 1, 5) / beta.pdf(_G, 1, 5).sum()


def _lik(k, n, w):
    return (_G ** k * (1 - _G) ** (n - k) * w).sum()


def _quad_death(y, sp, A, mu=21, sigma=25):
    W = len(y)
    M = ~np.isnan(y)
    yz = np.nan_to_num(y)
    pri = norm.pdf(np.arange(A), mu, sigma)
    post = np.zeros(A)
    for a in range(A):
        al = (np.arange(W) >= sp) & (np.arange(W) <= sp + a)
        post[a] = pri[a] * _lik((yz * M * al).sum(), (M * al).sum(), _WH) * _lik(
            (yz * M * ~al).sum(), (M * ~al).sum(), _WL)
    return post / post.sum()


def _series(rng, W, on_from, on_to, p_on=0.9, p_off=0.08, p_nan=0.1):
    y = (rng.random(W) < p_off).astype(float)
    y[on_from:on_to + 1] = rng.random(on_to - on_from + 1) < p_on
    y[rng.random(W) < p_nan] = np.nan
    return y


def test_death_posterior_matches_quadrature():
    rng = np.random.default_rng(0)
    for _ in range(4):
        W = int(rng.integers(15, 40))
        y = _series(rng, W, 0, int(rng.integers(3, W)))
        sp, A = int(rng.integers(0, 3)), W + 10
        exact = ls.death_posterior(y[None], [sp], n_support=A)[0]
        assert np.abs(exact - _quad_death(y, sp, A)).max() < 1e-5


def test_birth_posterior_matches_quadrature():
    rng = np.random.default_rng(1)
    y = _series(rng, 14, 4, 13)
    L = len(y)
    M, yz = ~np.isnan(y), np.nan_to_num(y)
    pri = norm.pdf(np.arange(L), 0, 5)
    q = np.array([pri[c] * _lik((yz * M * (np.arange(L) >= c)).sum(), (M * (np.arange(L) >= c)).sum(), _WH)
                  * _lik((yz * M * (np.arange(L) < c)).sum(), (M * (np.arange(L) < c)).sum(), _WL)
                  for c in range(L)])
    assert np.abs(ls.birth_posterior(y[None])[0] - q / q.sum()).max() < 1e-5


def test_joint_reduces_to_two_step_when_birth_is_certain():
    y = np.r_[np.zeros(1), np.ones(30), np.zeros(20)][None]
    j, t = ls.birth_death_posterior(y), ls.two_step_posterior(y)
    grid = np.arange(j["death"].shape[1])
    assert abs((j["death"][0] * grid).sum() - (t["death"][0] * grid).sum()) < 0.05


def test_masked_day_contributes_nothing():
    # a masked (NaN) day adds no likelihood: the posterior equals that of a series in
    # which the day is dropped, up to the index shift after it
    c = np.r_[np.ones(10), [np.nan], np.zeros(10)]
    d = np.r_[np.ones(10), np.zeros(10)]
    pc = ls.death_posterior(c[None], n_support=60)[0]
    pd_ = ls.death_posterior(d[None], n_support=60)[0]
    assert np.allclose(pc[:10] / pc[:10].sum(), pd_[:10] / pd_[:10].sum())
    # zero-filling the same day instead (legacy) is a phantom 'not seen' day
    z = np.r_[np.ones(5), [0.0], np.ones(5)]
    m = np.r_[np.ones(5), [np.nan], np.ones(5)]
    assert ls.prob_at_most(ls.death_posterior(z[None], n_support=60), 0, 4)[0] > \
        ls.prob_at_most(ls.death_posterior(m[None], n_support=60), 0, 4)[0]


def test_counts_on_bad_days_are_ignored():
    D = 12
    c = np.zeros((ls.N_IDS, D))
    c[:400, :] = 200000.0
    c[:, 6] *= 0.01                             # recording failed on day 6
    a, b = c.copy(), c.copy()
    b[5, 6] = 150000.0                          # whatever id 5 shows on the bad day
    Ba = ls.binarize(ls.load_day_counts(_daydata(a)), cut=15000)
    Bb = ls.binarize(ls.load_day_counts(_daydata(b)), cut=15000)
    assert np.array_equal(np.isnan(Ba), np.isnan(Bb)) and np.array_equal(np.nan_to_num(Ba), np.nan_to_num(Bb))


def test_right_censored_bee_stays_alive():
    y = np.ones(40)[None]
    post = ls.death_posterior(y, n_support=150)
    assert 1 - ls.prob_at_most(post, 0, 38)[0] > 0.95       # alive on the last day
    # legacy zero padding (support = series) would force death near the end instead
    legacy = ls.death_posterior(np.r_[np.ones(40), np.zeros(25)][None])
    assert ls.prob_at_most(legacy, 0, 45)[0] > 0.99


def test_p_real_separates_bee_from_noise():
    rng = np.random.default_rng(2)
    noise = (rng.random((20, 60)) < 0.03).astype(float)
    bee = np.zeros((1, 60))
    bee[0, 5:30] = 1
    assert np.median(ls.birth_death_posterior(noise, birth_window=None)["p_real"]) < 0.2
    assert ls.birth_death_posterior(bee, birth_window=None)["p_real"][0] > 0.999


def _daydata(counts: np.ndarray, start="2026-07-01") -> pd.DataFrame:
    """(4096, D) count matrix -> long daydatamat rows (only counts > 0)."""
    days = pd.date_range(start, periods=counts.shape[1])
    i, j = np.nonzero(counts)
    return pd.DataFrame({"hive": "A", "bee_id": i, "daynum": j, "day": days[j].strftime("%Y-%m-%d"),
                         "num_detections": counts[i, j]})


def test_twin_aware_binarization_and_day_quality():
    D = 20
    c = np.zeros((ls.N_IDS, D))
    c[:400, :] = 200000.0                       # a colony of real bees
    c[10 + ls.ODD_OFFSET, :] = 200000.0         # odd 10 alive
    c[10, :] = 20000.0                          # even 10: 10 % leakage from its twin
    c[:, 7] *= 0.01                             # a broken recording day
    dc = ls.load_day_counts(_daydata(c))
    assert not dc.good[7] and dc.good[6]
    B0 = ls.binarize(dc, cut=15000, twin_alpha=0.0)
    B1 = ls.binarize(dc, cut=15000, twin_alpha=0.25)
    assert np.isnan(B0[:, 7]).all()
    assert np.nanmean(B0[10]) == 1.0 and np.nanmean(B1[10]) == 0.0
    assert np.nanmean(B1[10 + ls.ODD_OFFSET]) == 1.0


def test_intro_table_matches_build_reuse_intervals():
    dftags = pd.DataFrame({"tag_start": [0, 2048, 0], "tag_end": [9, 2050, 4],
                           "tag_start2": np.nan, "tag_end2": np.nan, "Hive": "A",
                           "Date": ["2026-06-01", "2026-06-02", "2026-07-01"], "row_id": ["1", "2", "3"]})
    a = ls.intro_table(dftags)[["hive", "bee_id", "generation", "uid"]].reset_index(drop=True)
    b = uidmod.build_reuse_intervals(dftags)[["hive", "bee_id", "generation", "uid"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a.astype("int64", errors="ignore"), b.astype("int64", errors="ignore"),
                                  check_dtype=False)


def test_log_to_dftags_parity_and_validation():
    log = pd.DataFrame({"row_id": ["1", "2"], "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
                        "num_start": [0, 5], "num_end": [3, 7], "parity": ["odd", "even"],
                        "hive": "A", "include": [True, True]})
    d = ls.log_to_dftags(log)
    assert d["tag_start"].tolist() == [2048, 5] and d["tag_end"].tolist() == [2051, 7]
    bad = log.assign(num_end=[3, 2100])
    with pytest.raises(ValueError):
        ls.log_to_dftags(bad)


def test_free_runs_and_compress():
    st = pd.DataFrame({"parity": "even", "number": np.arange(10),
                       "status": ["free", "free", "occupied", "free", "never_used",
                                  "free", "uncertain", "free", "free", "free"],
                       "twin_status": "free"})
    runs = ls.free_runs(st)
    assert runs[["num_start", "num_end"]].values.tolist() == [[0, 1], [3, 5], [7, 9]]
    assert ls.free_runs(st, max_gap=1)["n_free"].tolist() == [8]
    assert ls.compress_numbers([1, 2, 3, 7, 9, 10]) == "1-3, 7, 9-10"


def test_legacy_construction_matches_metrics_pipeline(monkeypatch):
    mp = pytest.importorskip("bb_metrics.metrics_pipeline")

    class _Trace:
        def __init__(self, name, value):
            self.posterior = {name: pd.Series([value])}

    def fake_death_fit(self, num_detect, *, switchpoint_emerged=0, **_):
        y = (num_detect > ls._legacy_cut(self)).astype(float)[None]
        post = ls.death_posterior(y, [switchpoint_emerged], mu=self.mu_days_alive, sigma=self.sigma_days_alive)
        return None, _Trace("switchpoint_died", switchpoint_emerged + (post[0] * np.arange(post.shape[1])).sum()), None

    def fake_birth_fit(self, num_detect, **_):
        y = (num_detect > ls._legacy_cut(self)).astype(float)[None]
        post = ls.birth_posterior(y, mu=self.mu_emergence, sigma=self.sigma_emergence)
        return None, _Trace("switchpoint_emerged", (post[0] * np.arange(post.shape[1])).sum()), None

    monkeypatch.setattr(mp.LifetimeEstimator, "fit", fake_death_fit)
    monkeypatch.setattr(mp.BirthEstimator, "fit", fake_birth_fit)
    rng = np.random.default_rng(3)
    rows = []
    for b in range(6):
        start, stop = int(rng.integers(0, 10)), int(rng.integers(15, 40))
        for d in range(start, stop):
            if rng.random() < 0.85:  # gaps are zero-filled by both
                rows.append({"hive": "A", "bee_id": b, "daynum": d,
                             "num_detections": float(rng.choice([500, 4000, 60000]))})
    dfday = pd.DataFrame(rows)
    ref_b = mp.estimate_birth_days(dfday, hives=["A"], estimator=mp.BirthEstimator())
    ref_d = mp.estimate_death_days(dfday, hives=["A"], estimator=mp.LifetimeEstimator(), birth_days=ref_b)
    ex_b = ls.estimate_birth_days_exact(dfday, hives=["A"], estimator=mp.BirthEstimator())
    ex_d = ls.estimate_death_days_exact(dfday, hives=["A"], estimator=mp.LifetimeEstimator(), birth_days=ex_b)
    m = ref_b.merge(ex_b, on=["hive", "bee_id"]).merge(ref_d.merge(ex_d, on=["hive", "bee_id"]), on=["hive", "bee_id"])
    assert (m["estimated_birth_daynum_x"] == m["estimated_birth_daynum_y"]).all()
    assert (m["estimated_death_daynum_x"] == m["estimated_death_daynum_y"]).all()


def test_cli_smoke(tmp_path):
    rng = np.random.default_rng(4)
    D = 40
    c = (rng.random((ls.N_IDS, D)) < 0.3) * rng.integers(100, 3000, (ls.N_IDS, D)).astype(float)
    c[:300, 5:] = 150000.0                              # even 0-299 tagged day 5
    c[:100, 25:] = 300.0                                # even 0-99 died day 25
    c[2048:2148, 10:] = 150000.0                        # odd 0-99 tagged day 10, alive
    path = tmp_path / "daydatamat.csv"
    _daydata(c).to_csv(path, index=False)
    log = pd.DataFrame({"row_id": ["1", "2"], "source": "test", "date": ["2026-07-06", "2026-07-11"],
                        "num_start": [0, 0], "num_end": [299, 99], "parity": ["even", "odd"],
                        "hive": "A", "include": [1, 1]})
    log.to_csv(tmp_path / "log.csv", index=False)
    out = tmp_path / "out"
    assert ls.main([str(path), "--log", str(tmp_path / "log.csv"), "--out", str(out), "--no-figures",
                    "--cut", "15000"]) == 0
    st = pd.read_csv(out / "tag_status.csv")
    ev = st[st["parity"] == "even"].set_index("number")["status"]
    od = st[st["parity"] == "odd"].set_index("number")["status"]
    assert (ev.loc[0:99] == "free").mean() > 0.95
    assert (ev.loc[100:299] == "occupied").mean() > 0.95
    assert (od.loc[0:99] == "occupied").mean() > 0.95
    for f in ("free_tags.txt", "free_runs.csv", "log_discrepancies.csv", "params.json"):
        assert (out / f).exists()


def test_block_table_labels_and_free_numbers():
    n = np.arange(ls.ODD_OFFSET)
    status = np.full(len(n), "occupied", dtype=object)
    status[0:32] = "free"                       # block 0-31: all dead
    status[32:40] = "free"                      # block 32-63: 8 of 32 free (25 %)
    status[64:96] = "never_used"                # block 64-95: never used
    status[96:100] = "uncertain"
    status[100:128] = "free"                    # block 96-127: dead except uncertain
    st = pd.DataFrame({"parity": "even", "number": n, "bee_id": n, "status": status, "flags": "",
                       "n_days_detected": 5, "source": "log", "intro_date": pd.Timestamp("2026-07-01"),
                       "onset_day": pd.Timestamp("2026-07-02"), "death_q50": pd.Timestamp("2026-07-20")})
    st = pd.concat([st, st.assign(parity="odd", bee_id=n + ls.ODD_OFFSET, status="occupied")], ignore_index=True)
    intros = pd.DataFrame({"hive": "A", "bee_id": np.r_[0:64, 96:ls.ODD_OFFSET],
                           "intro_date": pd.Timestamp("2026-07-01")})
    t = ls.block_table(st, intros, block=32).set_index(["parity", "numbers"])
    assert t.loc[("even", "0-31"), "label"] == "used, all dead now"
    assert t.loc[("even", "0-31"), "whole_16s_free"] == "0-15, 16-31"
    assert t.loc[("even", "32-63"), "label"] == "mostly in use" and t.loc[("even", "32-63"), "free_numbers"] == "32-39"
    assert t.loc[("even", "64-95"), "label"] == "never used"
    assert t.loc[("even", "96-127"), "label"] == "dead except uncertain"
    assert t.loc[("odd", "0-31"), "label"] == "in use"
    assert t.loc[("even", "0-31"), "intro_inferred_median"] == "2026-07-02"


def test_neighbor_leak_flags_onset_right_after_a_neighbour():
    ids = np.arange(ls.N_IDS)
    st = pd.DataFrame({"bee_id": ids, "source": "log", "status": "free", "flags": "",
                       "onset_day": pd.NaT, "intro_date": pd.NaT, "recent_median": 1000.0})
    st.loc[4, ["source", "status", "intro_date", "onset_day", "recent_median"]] = \
        ["log", "occupied", pd.Timestamp("2026-09-02"), pd.Timestamp("2026-09-03"), 200000.0]
    st.loc[5, ["source", "status", "onset_day", "recent_median"]] = \
        ["data", "occupied", pd.Timestamp("2026-09-05"), 40000.0]   # 5 = 4 ^ 1, came on 2 days later
    st.loc[9, ["source", "status", "onset_day", "recent_median"]] = \
        ["data", "occupied", pd.Timestamp("2026-08-01"), 40000.0]   # no neighbour started near its onset
    flags = ls.neighbor_leak(st)
    assert flags[5] and not flags[9] and not flags[4]
