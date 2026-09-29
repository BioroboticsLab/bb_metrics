"""
Exact per-bee lifespan (birth / death) estimation and tag-status calls.

Self-contained: needs only numpy, pandas and scipy (matplotlib for figures, imported
lazily), and imports nothing from bb_metrics, so it also runs as a plain script:

    python lifespan.py daydatamat.csv --log tag_log.csv --out outdir/

Why exact instead of MCMC
-------------------------
The PyMC changepoint models in ``metrics_pipeline`` (``BirthEstimator``,
``LifetimeEstimator``) observe one Bernoulli "detected" flag per day, with a single
discrete changepoint and two Beta-distributed daily detection rates:

    p_hi ~ Beta(5, 1)   (alive)        p_lo ~ Beta(1, 5)   (not alive)

Both rates are conjugate, so they integrate out in closed form (Beta-Binomial) and the
posterior over the changepoint day is prior * marginal likelihood, normalised over every
candidate day. Enumerating all days gives the exact posterior the sampler approximates,
for every bee at once, in well under a second.

Differences from the legacy series construction (all switchable, see ``mode='legacy'``
in the drop-in estimators): days with no or partial recording are *masked* (they add no
likelihood) instead of zero-filled, and the death support extends past the last observed
day instead of padding the series with zero "not seen" days, so a bee still alive at the
end of the data is right-censored rather than forced to die there.

Tag numbers
-----------
A printed 12-bit tag number k (0-2047) maps to ``bee_id = k`` (even sheet) or
``bee_id = k + 2048`` (odd sheet). ``twin(bee_id)`` is the same number on the other
parity; decoding errors leak detections between twins, which is why a dead tag whose twin
is alive never drops to zero detections.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp
from scipy.stats import norm as _norm

N_IDS = 4096
ODD_OFFSET = 2048

# Beta priors on the daily detection probability, identical to the PyMC estimators
# (probability_higher ~ Beta(5, 1), probability_lower ~ Beta(1, 5)).
HI_PRIOR = (5.0, 1.0)
LO_PRIOR = (1.0, 5.0)

STATUS_ORDER = ["occupied", "uncertain", "free", "never_used"]
STATUS_COLORS = {
    "occupied": "#e41a1c",
    "uncertain": "#ff7f00",
    "free": "#4daf4a",
    "never_used": "#bdbdbd",
}

DEFAULT_PARAMS = {
    # day quality
    "top_k": 300,
    "quality_window": 7,
    "bad_q": 0.3,
    "rescale_below": 0.75,
    "manual_bad_days": [],
    # strict rule used only to find onsets (cohort structure), not for the death fits
    "onset_cut": 30000.0,
    "onset_twin_alpha": 0.25,
    "onset_tol": [-2, 3],
    # detection rule for the death model (None -> calibrated on the last
    # calibration_days days, the period the current generations cover)
    "cut": None,
    "twin_alpha": None,
    "calibration_days": 56,
    # death prior (legacy LifetimeEstimator values) and support length
    "mu_days_alive": 21.0,
    "sigma_days_alive": 25.0,
    "n_support": 150,
    # status
    "margin_days": 3,
    "p_free": 0.95,
    "p_occupied": 0.05,
    "strong_cut": 20000.0,
    "strong_twin_ratio": 0.5,
    "veto_days": [2, 3],
    # a 'free' call must survive a looser rule (lower cut; leakage below
    # sensitive_twin_alpha x twin still ignored), else it becomes 'uncertain'
    "sensitive_cut_factor": 0.5,
    "sensitive_twin_alpha": 0.25,
    # a 'free' tag whose median count over the last low_signal_days good days exceeds
    # low_signal_cut, and is not explained by leakage from its twin, becomes 'uncertain'
    # (leakage from other one-bit neighbours looks the same as a live bee seen rarely)
    "low_signal_cut": 5000.0,
    "low_signal_days": 7,
    "last_day": None,
    # ages of the bees on the tags are given as of this day (None = today)
    "age_date": None,
}


# ---------------------------------------------------------------------------
# tag number <-> bee_id
# ---------------------------------------------------------------------------
def to_bee_id(number, parity) -> np.ndarray:
    """Tag number (0-2047) + parity ('even'/'odd') -> ferwar bee_id (0-4095)."""
    number = np.asarray(number, dtype=int)
    odd = np.char.lower(np.asarray(parity).astype(str)) == "odd"
    return number + ODD_OFFSET * odd


def from_bee_id(bee_id):
    """bee_id -> (tag number, parity)."""
    bee_id = np.asarray(bee_id, dtype=int)
    return bee_id % ODD_OFFSET, np.where(bee_id >= ODD_OFFSET, "odd", "even")


def twin(bee_id) -> np.ndarray:
    """Same tag number on the other parity sheet."""
    return np.asarray(bee_id, dtype=int) ^ ODD_OFFSET


# ---------------------------------------------------------------------------
# daily detection counts
# ---------------------------------------------------------------------------
@dataclass
class DayCounts:
    """Per-id daily detection counts for one hive on a gap-free calendar grid.

    raw[i, j]  summed detections of bee_id i on calendar day j; NaN if day j is absent
               from the input altogether, 0 if the id had no row on a present day.
    norm       raw divided by the day's recording-fraction scale; NaN on bad days.
    quality    one row per calendar day (see :func:`day_quality`).
    """

    hive: str
    days: pd.DatetimeIndex
    daynum: np.ndarray
    raw: np.ndarray
    norm: np.ndarray
    quality: pd.DataFrame

    @property
    def good(self) -> np.ndarray:
        return self.quality["good"].to_numpy()

    def day_index(self, day) -> int:
        """Calendar index of *day*; may be negative or >= len(days) (off-grid)."""
        return int((pd.Timestamp(day).normalize() - self.days[0]).days)

    def day_at(self, idx) -> pd.Timestamp:
        return self.days[0] + pd.Timedelta(days=int(idx))

    def last_good_day(self) -> pd.Timestamp:
        return self.days[np.where(self.good)[0][-1]]


def _read_daydata(daydata, value_col: str) -> pd.DataFrame:
    if isinstance(daydata, pd.DataFrame):
        return daydata
    wanted = {"hive", "bee_id", "day", "daynum", value_col}
    return pd.read_csv(daydata, usecols=lambda c: c in wanted)


def day_counts_from_metrics(df_metrics: pd.DataFrame, *, tz: str = "Europe/Berlin",
                            value_col: str = "num_detections") -> pd.DataFrame:
    """Aggregate hourly metrics (metrics-60min parquet rows) to (hive, bee_id, day)."""
    ts = df_metrics["timestamp_start"]
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    day = ts.dt.tz_convert(tz).dt.strftime("%Y-%m-%d")
    out = (df_metrics.assign(day=day)
           .groupby(["hive", "bee_id", "day"], as_index=False)[value_col].sum())
    return out


def day_quality(raw: np.ndarray, days: pd.DatetimeIndex, *, top_k: int = 300,
                window: int = 7, bad_q: float = 0.3, rescale_below: float = 0.75,
                manual_bad_days: Iterable = ()) -> pd.DataFrame:
    """Flag days with missing or partial recording.

    The day's level is the median of its top-k id counts (robust to colony size, unlike
    the day total), compared with the rolling median level of the surrounding days:
    q < bad_q -> bad (masked); bad_q <= q < rescale_below -> counts divided by q.
    """
    D = raw.shape[1]
    present = ~np.all(np.isnan(raw), axis=0)
    level = np.full(D, np.nan)
    n_ids = np.zeros(D, dtype=int)
    total = np.zeros(D)
    for j in np.where(present)[0]:
        col = np.nan_to_num(raw[:, j])
        k = min(top_k, len(col))
        level[j] = np.median(np.partition(col, len(col) - k)[-k:])
        n_ids[j] = int((col > 0).sum())
        total[j] = col.sum()
    lv = pd.Series(level)
    # reference from the upper half of nearby days, so a run of low days at the start
    # of the season cannot pull the reference down to its own level
    ref = lv.rolling(2 * window + 1, center=True, min_periods=1).quantile(0.75).to_numpy()
    q = level / ref
    manual = {pd.Timestamp(d).normalize() for d in manual_bad_days}
    reason = np.array([""] * D, dtype=object)
    reason[~present] = "missing"
    low = present & (q < bad_q)
    reason[low] = "partial"
    is_manual = np.array([d in manual for d in days])
    reason[is_manual] = "manual"
    good = present & ~low & ~is_manual
    scale = np.where(good & (q < rescale_below), q, 1.0)
    reason[good & (scale < 1)] = "rescaled"
    return pd.DataFrame({
        "day": days, "present": present, "n_ids": n_ids, "total": total,
        "level": level, "ref": ref, "q": q, "good": good, "scale": scale, "reason": reason,
    })


def load_day_counts(daydata, *, hive: Optional[str] = None,
                    value_col: str = "num_detections", top_k: int = 300,
                    quality_window: int = 7, bad_q: float = 0.3,
                    rescale_below: float = 0.75, manual_bad_days: Iterable = (),
                    **_ignored) -> DayCounts:
    """Load a day data matrix (path or DataFrame) into a :class:`DayCounts` grid."""
    df = _read_daydata(daydata, value_col)
    if "hive" in df.columns:
        hives = sorted(df["hive"].dropna().astype(str).unique())
        if hive is None:
            if len(hives) != 1:
                raise ValueError(f"several hives in input {hives}; pass hive=")
            hive = hives[0]
        df = df[df["hive"].astype(str) == str(hive)]
    else:
        hive = hive or "A"
    day = pd.to_datetime(df["day"].astype(str), format="%Y-%m-%d")
    days = pd.date_range(day.min(), day.max(), freq="D")
    j = ((day - days[0]).dt.days).to_numpy()
    bee = df["bee_id"].to_numpy(dtype=int)
    ok = (bee >= 0) & (bee < N_IDS)
    raw = np.full((N_IDS, len(days)), np.nan)
    raw[:, np.unique(j)] = 0.0
    np.add.at(raw, (bee[ok], j[ok]), df[value_col].to_numpy(dtype=float)[ok])

    if "daynum" in df.columns:
        off = (df["daynum"].to_numpy() - j)
        if len(np.unique(off)) != 1:
            warnings.warn("daynum is not a fixed offset of the calendar day; using the mode")
        daynum = np.arange(len(days)) + int(pd.Series(off).mode().iloc[0])
    else:
        daynum = np.arange(len(days))

    quality = day_quality(raw, days, top_k=top_k, window=quality_window, bad_q=bad_q,
                          rescale_below=rescale_below, manual_bad_days=manual_bad_days)
    normed = raw / quality["scale"].to_numpy()[None, :]
    normed[:, ~quality["good"].to_numpy()] = np.nan
    return DayCounts(hive=str(hive), days=days, daynum=daynum, raw=raw, norm=normed,
                     quality=quality)


def binarize(dc: DayCounts, *, cut: float, twin_alpha: float = 0.0) -> np.ndarray:
    """Daily detected flag per id: 1/0, NaN on bad days.

    Detected = count > cut, and (if twin_alpha > 0) count > twin_alpha * twin's count, so
    decoding leakage from a live twin does not keep a dead tag "alive".
    """
    n = dc.norm
    with np.errstate(invalid="ignore"):
        B = n > cut
        if twin_alpha and twin_alpha > 0:
            B &= n > twin_alpha * n[twin(np.arange(N_IDS))]
    out = B.astype(float)
    out[:, ~dc.good] = np.nan
    return out


# ---------------------------------------------------------------------------
# exact changepoint posteriors
# ---------------------------------------------------------------------------
def _cumulative(Y: np.ndarray):
    """(hits, observed days) cumulative along axis 1, with a leading zero column."""
    M = ~np.isnan(Y)
    Yz = np.where(M, Y, 0.0)
    S = Y.shape[0]
    CK = np.concatenate([np.zeros((S, 1)), np.cumsum(Yz, axis=1)], axis=1)
    CN = np.concatenate([np.zeros((S, 1)), np.cumsum(M, axis=1)], axis=1)
    return CK, CN


def _log_marginal(k_in, n_in, k_out, n_out, hi=HI_PRIOR, lo=LO_PRIOR):
    """log p(data | changepoint) with p_hi / p_lo integrated out (Beta-Binomial)."""
    return (betaln(hi[0] + k_in, hi[1] + n_in - k_in) - betaln(*hi)
            + betaln(lo[0] + k_out, lo[1] + n_out - k_out) - betaln(*lo))


def _log_prior(n: int, mu: float, sigma: float) -> np.ndarray:
    """Discretised normal prior over 0..n-1, normalised (as the PyMC Categorical)."""
    p = _norm.pdf(np.arange(n), mu, sigma)
    return np.log(p / p.sum())


def death_log_joint(Y, sp=0, *, mu: float = 21.0, sigma: float = 25.0,
                    n_support: Optional[int] = None, hi=HI_PRIOR, lo=LO_PRIOR) -> np.ndarray:
    """Unnormalised log posterior over days alive, shape (S, A).

    Y (S, W): 1 detected / 0 not / NaN unobserved. sp: index of the first alive day
    (emergence) per row. Alive on [sp, sp + a]; every other observed day is 'not alive'.
    ``n_support=None`` uses A = W, the legacy LifetimeEstimator support.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    S, W = Y.shape
    A = W if n_support is None else int(n_support)
    sp = np.clip(np.broadcast_to(np.asarray(sp, dtype=int), (S,)), 0, W - 1)
    CK, CN = _cumulative(Y)
    K, N = CK[:, -1:], CN[:, -1:]
    end = np.minimum(sp[:, None] + np.arange(A)[None, :], W - 1)
    rows = np.arange(S)[:, None]
    k_in = CK[rows, end + 1] - CK[rows, sp[:, None]]
    n_in = CN[rows, end + 1] - CN[rows, sp[:, None]]
    return _log_marginal(k_in, n_in, K - k_in, N - n_in, hi, lo) + _log_prior(A, mu, sigma)[None, :]


def death_posterior(Y, sp=0, **kw) -> np.ndarray:
    """Exact posterior pmf over days alive (last alive index = sp + a), shape (S, A)."""
    lj = death_log_joint(Y, sp, **kw)
    return np.exp(lj - logsumexp(lj, axis=1, keepdims=True))


def birth_log_joint(Y, *, mu: float = 0.0, sigma: float = 5.0,
                    hi=HI_PRIOR, lo=LO_PRIOR) -> np.ndarray:
    """Unnormalised log posterior over the emergence offset c (alive for day >= c)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    S, L = Y.shape
    CK, CN = _cumulative(Y)
    K, N = CK[:, -1:], CN[:, -1:]
    k_out, n_out = CK[:, :L], CN[:, :L]
    return _log_marginal(K - k_out, N - n_out, k_out, n_out, hi, lo) + _log_prior(L, mu, sigma)[None, :]


def birth_posterior(Y, **kw) -> np.ndarray:
    """Exact posterior pmf over the emergence offset, shape (S, L)."""
    lj = birth_log_joint(Y, **kw)
    return np.exp(lj - logsumexp(lj, axis=1, keepdims=True))


def null_log_evidence(Y, *, lo=LO_PRIOR) -> np.ndarray:
    """log p(data | never alive): every observed day at the 'not alive' rate."""
    CK, CN = _cumulative(np.atleast_2d(np.asarray(Y, dtype=float)))
    K, N = CK[:, -1], CN[:, -1]
    return betaln(lo[0] + K, lo[1] + N - K) - betaln(*lo)


def birth_death_posterior(Y, *, mu_birth: float = 0.0, sigma_birth: float = 5.0,
                          birth_window: Optional[int] = 14, mu_days_alive: float = 21.0,
                          sigma_days_alive: float = 25.0, n_support: int = 150,
                          hi=HI_PRIOR, lo=LO_PRIOR, chunk: int = 512) -> dict:
    """Joint exact posterior over emergence b and days alive a (alive on [b, b + a]).

    Independent priors, the legacy ones: b ~ N(mu_birth, sigma_birth) over the first
    `birth_window` days (None = whole series), a ~ N(mu_days_alive, sigma_days_alive)
    over 0..n_support-1. Same Beta-Bernoulli likelihood as the separate models.

    Returns birth pmf (S, Bw) over b, death pmf (S, Bw + A - 1) over the last alive index
    d = b + a, and p_real = P(alive model | data) against the never-alive model at 1:1
    prior odds.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    S, W = Y.shape
    Bw = W if birth_window is None else max(1, min(int(birth_window), W))
    A = int(n_support)
    lp = _log_prior(Bw, mu_birth, sigma_birth)[:, None] + _log_prior(A, mu_days_alive, sigma_days_alive)[None, :]
    b = np.arange(Bw)
    end = np.minimum(b[:, None] + np.arange(A)[None, :], W - 1)            # (Bw, A)
    birth = np.empty((S, Bw))
    death = np.zeros((S, Bw + A - 1))
    logz = np.empty(S)
    for s0 in range(0, S, chunk):
        CK, CN = _cumulative(Y[s0:s0 + chunk])
        K, N = CK[:, -1][:, None, None], CN[:, -1][:, None, None]
        k_in = CK[:, end + 1] - CK[:, b][:, :, None]
        n_in = CN[:, end + 1] - CN[:, b][:, :, None]
        lj = _log_marginal(k_in, n_in, K - k_in, N - n_in, hi, lo) + lp[None]
        z = logsumexp(lj.reshape(len(lj), -1), axis=1)
        post = np.exp(lj - z[:, None, None])
        birth[s0:s0 + chunk] = post.sum(axis=2)
        for bi in range(Bw):
            death[s0:s0 + chunk, bi:bi + A] += post[:, bi, :]
        logz[s0:s0 + chunk] = z
    p_real = 1.0 / (1.0 + np.exp(null_log_evidence(Y, lo=lo) - logz))
    return {"birth": birth, "death": death, "p_real": p_real, "log_evidence": logz}


def two_step_posterior(Y, *, mu_birth: float = 0.0, sigma_birth: float = 5.0,
                       birth_window: Optional[int] = 14, mu_days_alive: float = 21.0,
                       sigma_days_alive: float = 25.0, n_support: int = 150,
                       hi=HI_PRIOR, lo=LO_PRIOR) -> dict:
    """The legacy chain, exactly: birth on the first `birth_window` days, round its
    posterior mean, then death with the alive window starting there. Same output layout
    as :func:`birth_death_posterior` (death pmf offset by the rounded birth)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    S, W = Y.shape
    Bw = W if birth_window is None else max(1, min(int(birth_window), W))
    birth = birth_posterior(Y[:, :Bw], mu=mu_birth, sigma=sigma_birth, hi=hi, lo=lo)
    sp = np.round((birth * np.arange(Bw)).sum(axis=1)).astype(int)
    dpost = death_posterior(Y, sp, mu=mu_days_alive, sigma=sigma_days_alive,
                            n_support=n_support, hi=hi, lo=lo)
    A = dpost.shape[1]
    death = np.zeros((S, Bw + A - 1))
    rows = np.arange(S)[:, None]
    death[rows, sp[:, None] + np.arange(A)[None, :]] = dpost
    lz = logsumexp(death_log_joint(Y, sp, mu=mu_days_alive, sigma=sigma_days_alive,
                                   n_support=n_support, hi=hi, lo=lo), axis=1)
    p_real = 1.0 / (1.0 + np.exp(null_log_evidence(Y, lo=lo) - lz))
    return {"birth": birth, "death": death, "p_real": p_real, "log_evidence": lz, "sp": sp}


def posterior_summary(post: np.ndarray, offset=0, probs=(0.05, 0.5, 0.95)) -> pd.DataFrame:
    """Mean, quantiles and MAP of a pmf over offset + 0..A-1 (per row)."""
    S, A = post.shape
    offset = np.broadcast_to(np.asarray(offset, dtype=float), (S,))
    vals = offset[:, None] + np.arange(A)[None, :]
    out = {"mean": (post * vals).sum(axis=1)}
    cdf = np.cumsum(post, axis=1)
    for p in probs:
        out[f"q{int(round(p * 100)):02d}"] = offset + np.argmax(cdf >= p - 1e-12, axis=1)
    out["map"] = offset + np.argmax(post, axis=1)
    return pd.DataFrame(out)


def prob_at_most(post: np.ndarray, offset, cutoff) -> np.ndarray:
    """P(offset + a <= cutoff) per row."""
    S, A = post.shape
    offset = np.broadcast_to(np.asarray(offset, dtype=float), (S,))
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=float), (S,))
    vals = offset[:, None] + np.arange(A)[None, :]
    return (post * (vals <= cutoff[:, None])).sum(axis=1)


# ---------------------------------------------------------------------------
# tag log -> dftags -> generations
# ---------------------------------------------------------------------------
def read_tag_log(path) -> pd.DataFrame:
    """Read a tag-introduction log CSV (raw + corrected columns, one row per range)."""
    log = pd.read_csv(path, dtype={"row_id": str})
    log["date"] = pd.to_datetime(log["date"])
    for c in ("num_start", "num_end"):
        log[c] = pd.to_numeric(log[c], errors="coerce")
    log["include"] = log["include"].fillna(0).astype(int).astype(bool)
    log["parity"] = log["parity"].astype(str).str.strip().str.lower()
    if "hive" not in log.columns:
        log["hive"] = "A"
    return log


def log_to_dftags(log: pd.DataFrame, *, min_gap_days: int = 5) -> pd.DataFrame:
    """Included log rows -> dftags in bee_id space (the format uid.assign_uid expects).

    Parity is taken per row, so one hive can hold both sheets.
    """
    use = log[log["include"] & log["num_start"].notna() & log["num_end"].notna()].copy()
    bad = use[(use["num_start"] > use["num_end"]) | (use["num_start"] < 0)
              | (use["num_end"] >= ODD_OFFSET) | ~use["parity"].isin(["even", "odd"])]
    if len(bad):
        raise ValueError(f"invalid log rows (range or parity): {bad['row_id'].tolist()}")
    use["tag_start"] = to_bee_id(use["num_start"].astype(int), use["parity"])
    use["tag_end"] = to_bee_id(use["num_end"].astype(int), use["parity"])
    dftags = pd.DataFrame({
        "tag_start": use["tag_start"].to_numpy(), "tag_end": use["tag_end"].to_numpy(),
        "tag_start2": np.nan, "tag_end2": np.nan,
        "Hive": use["hive"].astype(str).to_numpy(),
        "Date": use["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "parity": use["parity"].to_numpy(),
        "num_start": use["num_start"].astype(int).to_numpy(),
        "num_end": use["num_end"].astype(int).to_numpy(),
        "row_id": use["row_id"].to_numpy(),
    })
    it = intro_table(dftags)
    gaps = it.groupby(["hive", "bee_id"])["intro_date"].diff().dt.days
    close = it[gaps < min_gap_days]
    if len(close):
        warnings.warn(f"{len(close)} bee_ids re-introduced < {min_gap_days} days after the "
                      f"previous intro (rows {sorted(close['row_id'].unique())})")
    return dftags


def intro_table(dftags: pd.DataFrame) -> pd.DataFrame:
    """One row per (hive, bee_id, introduction); generation/uid as uid.build_reuse_intervals."""
    recs = []
    for r in dftags.itertuples(index=False):
        for a, b in (("tag_start", "tag_end"), ("tag_start2", "tag_end2")):
            s, e = getattr(r, a), getattr(r, b)
            if pd.notna(s) and pd.notna(e):
                ids = np.arange(int(s), int(e) + 1)
                recs.append(pd.DataFrame({"hive": r.Hive, "bee_id": ids,
                                          "intro_date": pd.Timestamp(r.Date),
                                          "row_id": getattr(r, "row_id", None)}))
    if not recs:
        return pd.DataFrame(columns=["hive", "bee_id", "intro_date", "row_id", "generation", "uid"])
    it = pd.concat(recs, ignore_index=True)
    it["intro_date"] = pd.to_datetime(it["intro_date"]).dt.tz_localize(None).dt.normalize()
    it = it.sort_values(["hive", "bee_id", "intro_date"], kind="stable").reset_index(drop=True)
    it["generation"] = it.groupby(["hive", "bee_id"]).cumcount()
    it["uid"] = it["bee_id"] + it["generation"] * N_IDS
    return it


# ---------------------------------------------------------------------------
# onsets and log reconciliation
# ---------------------------------------------------------------------------
def detect_onsets(B: np.ndarray, dc: DayCounts, *, window: int = 3, min_on: int = 2,
                  max_before: int = 0) -> pd.DataFrame:
    """Days where an id switches from absent to present (bad days skipped).

    Onset at good day j: detected on j, on >= min_on of the next `window` good days and
    on <= max_before of the previous `window` good days. Onsets with fewer than `window`
    good days before them are flagged left_censored (the id may have been present already).
    """
    gidx = np.where(dc.good)[0]
    Bg = np.nan_to_num(B[:, gidx])
    G = len(gidx)
    C = np.concatenate([np.zeros((Bg.shape[0], 1)), np.cumsum(Bg, axis=1)], axis=1)
    j = np.arange(G)
    prev = C[:, j] - C[:, np.maximum(j - window, 0)]
    nxt = C[:, np.minimum(j + 1 + window, G)] - C[:, j + 1]
    on = (Bg == 1) & (nxt >= min_on) & (prev <= max_before)
    ids, jj = np.nonzero(on)
    return pd.DataFrame({
        "bee_id": ids, "day": dc.days[gidx[jj]], "day_idx": gidx[jj],
        "left_censored": jj < window,
    }).sort_values(["bee_id", "day"]).reset_index(drop=True)


def _present_before(B: np.ndarray, dc: DayCounts, ids, day, *, window: int = 3,
                    min_on: int = 2) -> np.ndarray:
    """Detected on >= min_on of the last `window` good days before `day`."""
    gidx = np.where(dc.good)[0]
    before = gidx[gidx < dc.day_index(day)][-window:]
    if len(before) == 0:
        return np.zeros(len(ids), dtype=bool)
    return np.nan_to_num(B[np.ix_(ids, before)]).sum(axis=1) >= min_on


def explain_onsets(onsets: pd.DataFrame, intros: pd.DataFrame, *, tol=(-2, 3)) -> pd.DataFrame:
    """Attach the log introduction (if any) that explains each onset."""
    on = onsets.copy()
    on["intro_date"] = pd.NaT
    on["row_id"] = None
    if len(intros) and len(on):
        m = on.reset_index().merge(intros[["bee_id", "intro_date", "row_id"]], on="bee_id",
                                   how="inner", suffixes=("_x", ""))
        off = (m["day"] - m["intro_date"]).dt.days
        m = m[(off >= tol[0]) & (off <= tol[1])]
        m = m.assign(_abs=off.loc[m.index].abs()).sort_values("_abs").drop_duplicates("index")
        on.loc[m["index"].to_numpy(), "intro_date"] = m["intro_date"].to_numpy()
        on.loc[m["index"].to_numpy(), "row_id"] = m["row_id"].to_numpy()
    on["explained"] = on["intro_date"].notna()
    return on


def reconcile_log(dc: DayCounts, log: pd.DataFrame, B: np.ndarray, onsets: pd.DataFrame,
                  *, tol=(-2, 3), search_days: int = 12) -> pd.DataFrame:
    """Check every log row with a numeric range against the onsets seen in the data.

    For the logged parity and the twin parity: the fraction of ids with an onset within
    `tol` days of the row date, the median offset, and the tag-number span that actually
    rose. The verdict and suggested_correction are hints for editing the log CSV.
    """
    first_good = dc.days[np.where(dc.good)[0][0]]
    out = []
    rows = log[log["num_start"].notna() & log["num_end"].notna()]
    by_id = {k: g for k, g in onsets.groupby("bee_id")}

    def _onsets_near(ids, d, lo, hi):
        res = []
        for i in ids:
            g = by_id.get(int(i))
            if g is None:
                continue
            off = (g["day"] - d).dt.days
            hit = g[(off >= lo) & (off <= hi)]
            if len(hit):
                k = (hit["day"] - d).dt.days.abs().idxmin()
                res.append((int(i), int((hit.loc[k, "day"] - d).days)))
        return res

    for r in rows.itertuples(index=False):
        a, b = int(min(r.num_start, r.num_end)), int(max(r.num_start, r.num_end))
        nums = np.arange(a, b + 1)
        d = pd.Timestamp(r.date)
        rec = {"row_id": r.row_id, "date": d.date(), "parity": r.parity,
               "num_start": a, "num_end": b, "n_ids": len(nums), "include": bool(r.include)}
        for label, par in (("logged", r.parity), ("twin", "odd" if r.parity == "even" else "even")):
            ids = to_bee_id(nums, [par] * len(nums))
            hits = _onsets_near(ids, d, *tol)
            rec[f"frac_onset_{label}"] = len(hits) / len(nums)
            if label == "logged":
                rec["median_offset"] = float(np.median([h[1] for h in hits])) if hits else np.nan
                # offset in good days, so masked days right after tagging do not count
                gdays = dc.days[dc.good]
                rec["median_good_offset"] = float(np.median(
                    [np.sum((gdays >= min(d, d + pd.Timedelta(days=h[1])))
                            & (gdays < max(d, d + pd.Timedelta(days=h[1])))) * np.sign(h[1])
                     for h in hits])) if hits else np.nan
                rn = [from_bee_id(h[0])[0] for h in hits]
                rec["num_min_rising"] = int(min(rn)) if rn else np.nan
                rec["num_max_rising"] = int(max(rn)) if rn else np.nan
                if dc.day_index(d) >= 0:
                    rec["n_prev_present"] = int(_present_before(B, dc, ids, d).sum())
                    after = np.nan_to_num(B[ids][:, max(dc.day_index(d), 0):])
                    rec["n_never_seen"] = int((after.sum(axis=1) == 0).sum())
                else:
                    rec["n_prev_present"] = np.nan
                    rec["n_never_seen"] = np.nan
                wide = _onsets_near(ids, d, -search_days, search_days)
                rec["best_date"] = (d + pd.Timedelta(days=int(np.median([h[1] for h in wide])))).date() if wide else None
                rec["frac_onset_wide"] = len(wide) / len(nums)
        fl, ft = rec["frac_onset_logged"], rec["frac_onset_twin"]
        if fl >= 0.5 and abs(np.nan_to_num(rec["median_good_offset"])) <= 1.5:
            verdict = "confirmed"
        elif d <= first_good:
            verdict = "left_censored"
        elif ft >= 0.5 and fl < 0.2:
            verdict = "parity_swapped"
        elif fl < 0.2 and rec["frac_onset_wide"] >= 0.5:
            verdict = "date_shift"
        elif fl >= 0.1:
            verdict = "partial"
        else:
            verdict = "no_onset"
        rec["verdict"] = verdict
        sug = ""
        if verdict == "parity_swapped":
            sug = f"parity -> {'odd' if r.parity == 'even' else 'even'}"
        elif verdict == "date_shift":
            sug = f"date -> {rec['best_date']}"
        elif verdict in ("confirmed", "partial") and pd.notna(rec["num_min_rising"]):
            if rec["num_min_rising"] > a + 3 or rec["num_max_rising"] < b - 3:
                sug = f"rising numbers span {int(rec['num_min_rising'])}-{int(rec['num_max_rising'])}"
        rec["suggested_correction"] = sug
        out.append(rec)
    return pd.DataFrame(out)


def unexplained_onset_clusters(onsets_x: pd.DataFrame, *, day_tol: int = 1, max_gap: int = 8,
                               min_ids: int = 5) -> pd.DataFrame:
    """Group onsets no log row explains into (parity, date, number-run) clusters."""
    u = onsets_x[~onsets_x["explained"] & ~onsets_x["left_censored"]].copy()
    if u.empty:
        return pd.DataFrame(columns=["parity", "day_min", "day_max", "day_median",
                                     "num_start", "num_end", "n_ids"])
    u["number"], u["parity"] = from_bee_id(u["bee_id"].to_numpy())
    out = []
    for par, g in u.groupby("parity"):
        g = g.sort_values("day")
        grp = (g["day"].diff().dt.days.fillna(0) > day_tol).cumsum()
        for _, gd in g.groupby(grp):
            gd = gd.sort_values("number")
            run = (gd["number"].diff().fillna(0) > max_gap).cumsum()
            for _, gr in gd.groupby(run):
                if len(gr) < min_ids:
                    continue
                out.append({"parity": par, "day_min": gr["day"].min().date(),
                            "day_max": gr["day"].max().date(),
                            "day_median": gr["day"].sort_values().iloc[len(gr) // 2].date(),
                            "num_start": int(gr["number"].min()), "num_end": int(gr["number"].max()),
                            "n_ids": len(gr)})
    return pd.DataFrame(out).sort_values(["day_median", "parity", "num_start"]).reset_index(drop=True) \
        if out else pd.DataFrame(columns=["parity", "day_min", "day_max", "day_median",
                                          "num_start", "num_end", "n_ids"])


# ---------------------------------------------------------------------------
# segments (one bee = one tag between two introductions) and death fits
# ---------------------------------------------------------------------------
def _segment_matrix(B: np.ndarray, bee_ids, start_idx, end_idx) -> np.ndarray:
    """Rows of B cut to [start_idx, end_idx] (calendar indices, start may be < 0),
    left-aligned, NaN outside the grid and past each row's end."""
    bee_ids = np.asarray(bee_ids, dtype=int)
    start_idx = np.asarray(start_idx, dtype=int)
    end_idx = np.asarray(end_idx, dtype=int)
    W = int((end_idx - start_idx).max()) + 1 if len(bee_ids) else 1
    D = B.shape[1]
    Y = np.full((len(bee_ids), W), np.nan)
    cols = start_idx[:, None] + np.arange(W)[None, :]
    valid = (cols >= 0) & (cols < D) & (cols <= end_idx[:, None])
    r, c = np.nonzero(valid)
    Y[r, c] = B[bee_ids[r], cols[r, c]]
    return Y


def fit_segments(B: np.ndarray, bee_ids, start_idx, end_idx, sp=0, *, mu=21.0, sigma=25.0,
                 n_support=150):
    """Exact death posterior for each (bee_id, [start, end]) segment.

    Returns (post, first_alive_idx) with post over a = 0..n_support-1 and the last alive
    calendar index = first_alive_idx + a.
    """
    Y = _segment_matrix(B, bee_ids, start_idx, end_idx)
    sp = np.broadcast_to(np.asarray(sp, dtype=int), (len(Y),))
    post = death_posterior(Y, sp, mu=mu, sigma=sigma, n_support=n_support)
    return post, np.asarray(start_idx, dtype=int) + sp


def build_segments(dc: DayCounts, intros: pd.DataFrame, onsets_x: pd.DataFrame, *,
                   last_day=None) -> pd.DataFrame:
    """The current generation of every bee_id as of `last_day`.

    Starts at the later of the latest log introduction and the latest onset that no log
    row explains; ids with neither span the whole season. sp = offset of the explaining
    onset (tagging in the afternoon shows up the next day), else 0.
    """
    T = pd.Timestamp(last_day) if last_day is not None else dc.last_good_day()
    ids = np.arange(N_IDS)
    seg = pd.DataFrame({"bee_id": ids})
    seg["number"], seg["parity"] = from_bee_id(ids)

    it = intros[intros["intro_date"] <= T]
    last_intro = it.sort_values("intro_date").groupby("bee_id").tail(1).set_index("bee_id")
    # reindex, not map: map of an empty datetime Series fails on pandas 3
    seg["intro_date"] = pd.to_datetime(last_intro["intro_date"].reindex(ids).to_numpy())
    seg["generation"] = seg["bee_id"].map(last_intro["generation"])
    seg["uid"] = seg["bee_id"].map(last_intro["uid"])
    seg["row_id"] = seg["bee_id"].map(last_intro["row_id"])
    seg["n_intros"] = seg["bee_id"].map(intros.groupby("bee_id").size()).fillna(0).astype(int)

    ux = onsets_x[~onsets_x["explained"] & ~onsets_x["left_censored"] & (onsets_x["day"] <= T)]
    last_ux = ux.groupby("bee_id")["day"].max()
    seg["unexplained_onset"] = pd.to_datetime(last_ux.reindex(ids).to_numpy())

    use_data = seg["unexplained_onset"].notna() & (
        seg["intro_date"].isna() | (seg["unexplained_onset"] > seg["intro_date"]))
    seg["source"] = np.where(use_data, "data", np.where(seg["intro_date"].notna(), "log", "season"))
    seg["start_day"] = seg["intro_date"].fillna(dc.days[0]).where(~use_data, seg["unexplained_onset"])
    seg["start_idx"] = (seg["start_day"] - dc.days[0]).dt.days.astype(int)

    ex = onsets_x[onsets_x["explained"]].copy()
    ex = ex.merge(seg[["bee_id", "intro_date"]], on="bee_id", suffixes=("_on", ""))
    ex = ex[ex["intro_date_on"] == ex["intro_date"]]
    sp_map = ((ex["day"] - ex["intro_date"]).dt.days.clip(lower=0)).groupby(ex["bee_id"]).min()
    seg["sp"] = np.where(seg["source"] == "log", seg["bee_id"].map(sp_map).fillna(0), 0).astype(int)
    # the day the current bee was first seen: its explaining onset, or the unexplained one
    on_log = pd.to_datetime(ex.groupby("bee_id")["day"].min().reindex(ids).to_numpy())
    seg["onset_day"] = pd.Series(on_log, index=seg.index).where(seg["source"] == "log",
                                                                 seg["unexplained_onset"])
    return seg


def reuse_conflicts(dc: DayCounts, B: np.ndarray, intros: pd.DataFrame, onsets_x: pd.DataFrame,
                    *, mu=21.0, sigma=25.0, n_support=150, p_alive: float = 0.5) -> pd.DataFrame:
    """For every re-introduction: was the previous bee on that tag still alive?"""
    it = intros.sort_values(["bee_id", "intro_date"]).copy()
    it["prev_intro"] = it.groupby("bee_id")["intro_date"].shift()
    re = it[it["prev_intro"].notna()].copy()
    if re.empty:
        return re.assign(p_prev_alive=[], conflict=[])
    start = (re["prev_intro"] - dc.days[0]).dt.days.to_numpy()
    end = (re["intro_date"] - dc.days[0]).dt.days.to_numpy() - 1
    ex = onsets_x[onsets_x["explained"]]
    sp_lookup = {(b, d): int(max((o - d).days, 0))
                 for b, d, o in zip(ex["bee_id"], ex["intro_date"], ex["day"])}
    sp = np.array([sp_lookup.get((b, d), 0) for b, d in zip(re["bee_id"], re["prev_intro"])])
    post, first = fit_segments(B, re["bee_id"].to_numpy(), start, end, sp,
                               mu=mu, sigma=sigma, n_support=n_support)
    re["p_prev_alive"] = 1.0 - prob_at_most(post, first, end - 1)
    re["prev_present_before"] = False
    for d, g in re.groupby("intro_date"):
        if dc.day_index(d) > 0:
            re.loc[g.index, "prev_present_before"] = _present_before(B, dc, g["bee_id"].to_numpy(), d)
    re["conflict"] = re["p_prev_alive"] > p_alive
    re["number"], re["parity"] = from_bee_id(re["bee_id"].to_numpy())
    return re.reset_index(drop=True)


# ---------------------------------------------------------------------------
# calibration of the detection rule
# ---------------------------------------------------------------------------
_TWIN_BINS = [-np.inf, 10000, 50000, 150000, np.inf]
_TWIN_LABELS = ["<10k", "10-50k", "50-150k", ">150k"]
_AGE_BINS = [-np.inf, 7, 21, 35, np.inf]
_AGE_LABELS = ["0-7", "8-21", "22-35", ">35"]


def calibration_controls(dc: DayCounts, intros: pd.DataFrame, onsets_x: pd.DataFrame, *,
                         strong_cut: float = 50000, strong_twin_ratio: float = 0.5,
                         later=(2, 10), young=(1, 7), pre_gap: int = 3,
                         B_strict: Optional[np.ndarray] = None, since=None, last_day=None,
                         post_gap: int = 5, mu: float = 21.0, sigma: float = 25.0,
                         n_support: int = 150) -> pd.DataFrame:
    """Id-days whose true state is known without any detection threshold.

    Positives: 'young' = days 1-7 after an introduction confirmed by an onset;
    'alive_later' = days followed within 2-10 days by a strong detection of the same
    generation (the bee was certainly alive in between). Negatives: 'pre_intro' = days
    more than `pre_gap` days before an id's first introduction, for ids with no onset of
    their own before it; 'post_death' (needs B_strict) = days from `post_gap` days after
    the 95% upper death bound of a generation that is confidently dead (fit on the strict
    onset rule) until 3 days before the tag's next introduction, for generations with no
    strong detection in that window. `since` keeps only days on/after that date, so the
    rule is calibrated on the period the current generations actually cover.
    """
    n = dc.norm
    tw = n[twin(np.arange(N_IDS))]
    good = dc.good
    D = n.shape[1]
    recs = []

    # generation start (latest intro <= day) per (id, day), as a calendar index
    gen_start = np.full((N_IDS, D), -10 ** 6, dtype=int)
    next_intro = np.full((N_IDS, D), 10 ** 6, dtype=int)
    for b, g in intros.groupby("bee_id"):
        idx = np.sort(((g["intro_date"] - dc.days[0]).dt.days).to_numpy())
        pos = np.searchsorted(idx, np.arange(D), side="right")
        gen_start[b] = np.where(pos > 0, idx[np.maximum(pos - 1, 0)], -10 ** 6)
        next_intro[b] = np.where(pos < len(idx), idx[np.minimum(pos, len(idx) - 1)], 10 ** 6)

    # young positives
    conf = onsets_x[onsets_x["explained"]]
    for b, d in zip(conf["bee_id"], conf["intro_date"]):
        j0 = dc.day_index(d)
        for j in range(j0 + young[0], min(j0 + young[1], D - 1) + 1):
            if 0 <= j < D and good[j]:
                recs.append((b, j, "pos", "young", j - j0))

    # alive-later positives
    with np.errstate(invalid="ignore"):
        strong = (n > strong_cut) & (n > strong_twin_ratio * tw)
    for j in np.where(good)[0]:
        lo, hi = j + later[0], min(j + later[1], D - 1)
        if lo > hi:
            continue
        s = strong[:, lo:hi + 1]
        # the strong day must precede the next re-introduction of the tag
        cols = np.arange(lo, hi + 1)[None, :]
        s = s & (cols < next_intro[:, [j]])
        ok = s.any(axis=1) & (gen_start[:, j] > -10 ** 6)
        for b in np.where(ok)[0]:
            recs.append((b, j, "pos", "alive_later", j - gen_start[b, j]))

    # pre-intro negatives
    first = intros.groupby("bee_id")["intro_date"].min()
    ux_first = onsets_x[~onsets_x["explained"]].groupby("bee_id")["day"].min()
    for b, d in first.items():
        if b in ux_first.index and ux_first[b] < d:
            continue
        jmax = dc.day_index(d) - pre_gap
        for j in np.where(good[:max(jmax, 0)])[0]:
            recs.append((b, j, "neg", "pre_intro", np.nan))

    # post-death negatives
    if B_strict is not None and len(intros):
        T_idx = dc.day_index(last_day if last_day is not None else dc.last_good_day())
        g = intros.sort_values(["bee_id", "intro_date"]).copy()
        g["start"] = (g["intro_date"] - dc.days[0]).dt.days
        g["end"] = g.groupby("bee_id")["start"].shift(-1).fillna(T_idx + 1).astype(int) - 1
        g["end"] = g["end"].clip(upper=T_idx)
        g = g[g["end"] - g["start"] >= 14]
        ex = onsets_x[onsets_x["explained"]]
        sp_lookup = {(b, d): int(max((o - d).days, 0))
                     for b, d, o in zip(ex["bee_id"], ex["intro_date"], ex["day"])}
        sp = np.array([sp_lookup.get((b, d), 0) for b, d in zip(g["bee_id"], g["intro_date"])])
        post, first = fit_segments(B_strict, g["bee_id"].to_numpy(), g["start"].to_numpy(),
                                   g["end"].to_numpy(), sp, mu=mu, sigma=sigma, n_support=n_support)
        q95 = posterior_summary(post, first)["q95"].to_numpy().astype(int)
        sure = prob_at_most(post, first, g["end"].to_numpy() - 10) >= 0.99
        for b, lo, hi in zip(g["bee_id"].to_numpy()[sure], q95[sure] + post_gap,
                             g["end"].to_numpy()[sure] - 3):
            if hi < lo:
                continue
            if strong[b, max(lo, 0):hi + 1].any():
                continue
            for j in range(max(lo, 0), min(hi, D - 1) + 1):
                if good[j]:
                    recs.append((b, j, "neg", "post_death", np.nan))

    ctl = pd.DataFrame(recs, columns=["bee_id", "j", "label", "kind", "age"])
    if since is not None:
        ctl = ctl[ctl["j"] >= dc.day_index(since)]
    ctl = ctl.drop_duplicates(["bee_id", "j", "label"])
    ctl["count"] = n[ctl["bee_id"], ctl["j"]]
    ctl["twin_count"] = tw[ctl["bee_id"], ctl["j"]]
    ctl["twin_bin"] = pd.cut(ctl["twin_count"], _TWIN_BINS, labels=_TWIN_LABELS)
    ctl["age_bin"] = pd.cut(ctl["age"], _AGE_BINS, labels=_AGE_LABELS)
    return ctl.reset_index(drop=True)


def _rule(count, twin_count, cut, alpha):
    with np.errstate(invalid="ignore"):
        on = count > cut
        if alpha > 0:
            on &= count > alpha * twin_count
    return on


def calibrate(ctl: pd.DataFrame, dc: DayCounts, *,
              cuts=(2500, 5000, 10000, 15000, 20000, 30000), alphas=(0.0, 0.05, 0.1, 0.2),
              max_miss_old: float = 0.05, max_miss_twin: float = 0.07, min_n: int = 200,
              tie: float = 0.005):
    """Miss / false-detection rates of each (cut, twin_alpha) rule on the controls.

    Chosen rule: among rules missing <= max_miss_old of 'alive_later' days of bees older
    than 21 days (and <= max_miss_twin with a >150k twin), the lowest worst-stratum
    false-detection rate; within `tie`, the lower cut (missing a live bee risks handing
    out a tag that is still on a bee).
    """
    pos = ctl[ctl["label"] == "pos"]
    neg = ctl[ctl["label"] == "neg"]
    old = pos[(pos["kind"] == "alive_later") & (pos["age"] > 21)]
    tw_hi = pos[pos["twin_bin"] == ">150k"]
    rows = []
    for cut in cuts:
        for a in alphas:
            r = {"cut": cut, "twin_alpha": a}
            r["miss_young"] = 1 - _rule(pos.loc[pos.kind == "young", "count"],
                                        pos.loc[pos.kind == "young", "twin_count"], cut, a).mean()
            r["miss_alive_later"] = 1 - _rule(pos.loc[pos.kind == "alive_later", "count"],
                                              pos.loc[pos.kind == "alive_later", "twin_count"], cut, a).mean()
            r["miss_old"] = 1 - _rule(old["count"], old["twin_count"], cut, a).mean()
            r["miss_twin_gt150k"] = 1 - _rule(tw_hi["count"], tw_hi["twin_count"], cut, a).mean()
            fd = _rule(neg["count"], neg["twin_count"], cut, a)
            r["false_all"] = fd.mean()
            worst = 0.0
            for lab in _TWIN_LABELS:
                m = (neg["twin_bin"] == lab).to_numpy()
                if m.sum() >= min_n:
                    v = fd[m].mean()
                    r[f"false_twin_{lab}"] = v
                    worst = max(worst, v)
            r["false_worst"] = worst
            # ids with >= 3 consecutive false good days (what fools a changepoint model)
            nd = neg.assign(on=fd.to_numpy()).sort_values(["bee_id", "j"])
            runs = nd.groupby("bee_id")["on"].apply(
                lambda s: int(pd.Series(s.to_numpy()).groupby((~s.to_numpy()).cumsum()).sum().max()))
            r["false_run3_ids"] = float((runs >= 3).mean()) if len(runs) else np.nan
            rows.append(r)
    table = pd.DataFrame(rows)
    ok = table[(table["miss_old"] <= max_miss_old) & (table["miss_twin_gt150k"] <= max_miss_twin)]
    if ok.empty:
        warnings.warn("no rule meets the miss-rate limits; choosing the lowest miss_old")
        ok = table.nsmallest(1, "miss_old")
    best = ok["false_worst"].min()
    pick = ok[ok["false_worst"] <= best + tie].sort_values(["cut", "twin_alpha"]).iloc[0]
    chosen = {"cut": float(pick["cut"]), "twin_alpha": float(pick["twin_alpha"])}
    return table, chosen


# ---------------------------------------------------------------------------
# tag status and free runs
# ---------------------------------------------------------------------------
def tag_status(dc: DayCounts, B: np.ndarray, segments: pd.DataFrame, *, last_day=None,
               margin_days: int = 3, p_free: float = 0.95, p_occupied: float = 0.05,
               mu_days_alive: float = 21.0, sigma_days_alive: float = 25.0,
               n_support: int = 150, strong_cut: float = 20000, strong_twin_ratio: float = 0.5,
               veto_days=(2, 3), conflicts: Optional[pd.DataFrame] = None,
               low_signal_cut: float = 5000.0, low_signal_days: int = 7,
               sensitive_twin_alpha: float = 0.25, **_ignored) -> pd.DataFrame:
    """One row per bee_id: is the tag on a living bee now?

    Death of the current generation from the exact posterior; T = last good day.
    free:       posterior-mean death <= T - margin AND P(death <= T - margin) >= p_free
    occupied:   P(death <= T - margin) <= p_occupied, or introduced after T - margin
    uncertain:  anything else
    never_used: never introduced, no onset, and at most one detected day
    A strong detection on veto_days[0] of the last veto_days[1] good days blocks 'free', and
    so does a recent median count above low_signal_cut that twin leakage cannot explain.
    """
    T = pd.Timestamp(last_day) if last_day is not None else dc.last_good_day()
    T_idx = dc.day_index(T)
    cutoff = T_idx - margin_days
    st = segments.copy()
    end = np.full(len(st), T_idx)
    post, first = fit_segments(B, st["bee_id"].to_numpy(), st["start_idx"].to_numpy(), end,
                               st["sp"].to_numpy(), mu=mu_days_alive, sigma=sigma_days_alive,
                               n_support=n_support)
    summ = posterior_summary(post, first)
    st["p_dead_by_cutoff"] = prob_at_most(post, first, cutoff)
    st["p_alive_last"] = 1.0 - prob_at_most(post, first, T_idx - 1)
    st["death_mean_idx"] = summ["mean"].to_numpy()
    for c in ("mean", "q05", "q50", "q95"):
        v = summ[c].to_numpy()
        st[f"death_{c}"] = [dc.day_at(int(np.floor(x))).date() if np.isfinite(x) else None for x in v]

    Y = _segment_matrix(B, st["bee_id"].to_numpy(), st["start_idx"].to_numpy(), end)
    st["n_days_detected"] = np.nansum(Y, axis=1).astype(int)
    st["n_days_observed"] = np.sum(~np.isnan(Y), axis=1).astype(int)
    det_idx = np.where(np.nan_to_num(B) > 0, np.arange(B.shape[1])[None, :], -1).max(axis=1)
    st["last_detected"] = [dc.day_at(j).date() if j >= 0 else None for j in det_idx]

    # veto: strong, twin-dominant detections on the most recent good days
    gidx = np.where(dc.good)[0]
    recent = gidx[gidx <= T_idx][-veto_days[1]:]
    n = dc.norm
    with np.errstate(invalid="ignore"):
        strong = (n[:, recent] > strong_cut) & (n[:, recent] > strong_twin_ratio * n[twin(np.arange(N_IDS))][:, recent])
    st["recent_strong_days"] = strong.sum(axis=1)[st["bee_id"].to_numpy()]
    veto = st["recent_strong_days"] >= veto_days[0]
    lastk = gidx[gidx <= T_idx][-low_signal_days:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(n[:, lastk], axis=1)
        med_tw = med[twin(np.arange(N_IDS))]
    st["recent_median"] = med[st["bee_id"].to_numpy()]
    low_signal = pd.Series((med > low_signal_cut) & (med > sensitive_twin_alpha * med_tw),
                           index=np.arange(N_IDS))[st["bee_id"].to_numpy()].to_numpy()
    veto = veto | low_signal

    new = st["start_idx"] > cutoff
    p = st["p_dead_by_cutoff"]
    mean_ok = st["death_mean_idx"] <= cutoff
    never = (st["source"] == "season") & (st["n_intros"] == 0) & (st["n_days_detected"] <= 1)
    status = np.where(new | (p <= p_occupied), "occupied",
                      np.where((p >= p_free) & mean_ok & ~veto, "free", "uncertain"))
    status = np.where(never & ~veto, "never_used", status)
    st["status"] = status

    flags = [[] for _ in range(len(st))]
    for i in np.where((st["source"] == "log") & (st["n_days_detected"] == 0))[0]:
        flags[i].append("logged_never_seen")
    for i in np.where(st["source"] == "data")[0]:
        flags[i].append("unlogged_onset")
    for i in np.where((st["recent_strong_days"] >= veto_days[0]) & (p >= p_free) & mean_ok)[0]:
        flags[i].append("veto_recent_detection")
    for i in np.where(low_signal & (p >= p_free) & mean_ok)[0]:
        flags[i].append("recent_low_signal")
    if conflicts is not None and len(conflicts):
        c = conflicts[conflicts["conflict"]]
        shared = set(zip(c["bee_id"], c["intro_date"]))
        for i, (b, d) in enumerate(zip(st["bee_id"], st["intro_date"])):
            if (b, d) in shared:
                flags[i].append("shared_id")
    st["flags"] = [";".join(f) for f in flags]
    tw_status = st.set_index("bee_id")["status"]
    st["twin_status"] = tw_status.reindex(twin(st["bee_id"].to_numpy())).to_numpy()
    return st


def downgrade_sensitive(st: pd.DataFrame, st_alt: pd.DataFrame) -> pd.DataFrame:
    """'free' ids that are not free under a looser detection rule become 'uncertain'."""
    st = st.copy()
    alt = st_alt.set_index("bee_id")["status"].reindex(st["bee_id"]).to_numpy()
    m = (st["status"] == "free").to_numpy() & ~np.isin(alt, ["free", "never_used"])
    st.loc[m, "status"] = "uncertain"
    st.loc[m, "flags"] = [";".join([f for f in (x, "threshold_sensitive") if f]) for x in st.loc[m, "flags"]]
    return st


def free_runs(status: pd.DataFrame, *, statuses=("free", "never_used"), max_gap: int = 0) -> pd.DataFrame:
    """Runs of consecutive free tag numbers per parity.

    max_gap > 0 merges runs separated by up to that many unavailable numbers (listed in
    'skip'), which is how tags are taken from a sheet in practice.
    """
    out = []
    for par, g in status.groupby("parity"):
        g = g.sort_values("number")
        nums = g.loc[g["status"].isin(statuses), "number"].to_numpy()
        twin_free = set(g.loc[g["twin_status"].isin(statuses), "number"])
        if len(nums) == 0:
            continue
        brk = np.where(np.diff(nums) > max_gap + 1)[0]
        for seg in np.split(nums, brk + 1):
            a, b = int(seg[0]), int(seg[-1])
            skip = sorted(set(range(a, b + 1)) - set(seg.tolist()))
            out.append({"parity": par, "num_start": a, "num_end": b, "n_free": len(seg),
                        "n_skip": len(skip), "skip": " ".join(map(str, skip)),
                        "n_twin_also_free": int(sum(x in twin_free for x in seg))})
    return pd.DataFrame(out)


def compress_numbers(nums) -> str:
    """[1, 2, 3, 7, 9, 10] -> '1-3, 7, 9-10'."""
    nums = sorted(int(x) for x in nums)
    if not nums:
        return ""
    out, a, b = [], nums[0], nums[0]
    for x in nums[1:]:
        if x == b + 1:
            b = x
            continue
        out.append(f"{a}" if a == b else f"{a}-{b}")
        a = b = x
    out.append(f"{a}" if a == b else f"{a}-{b}")
    return ", ".join(out)


def format_blocks(status: pd.DataFrame, *, block: int = 128,
                  statuses=("free", "never_used")) -> str:
    """Per parity and block of `block` numbers: counts by status and the free numbers."""
    lines = []
    for par in ("odd", "even"):
        g = status[status["parity"] == par]
        nfree = int(g["status"].isin(statuses).sum())
        lines.append(f"{par.upper()} tags: {nfree} free, "
                     f"{int((g['status'] == 'uncertain').sum())} uncertain, "
                     f"{int((g['status'] == 'occupied').sum())} occupied")
        for b0 in range(0, ODD_OFFSET, block):
            gb = g[(g["number"] >= b0) & (g["number"] < b0 + block)]
            free = gb.loc[gb["status"].isin(statuses), "number"]
            lines.append(f"  {par} {b0}-{b0 + block - 1}: {len(free)} free, "
                         f"{int((gb['status'] == 'uncertain').sum())} uncertain, "
                         f"{int((gb['status'] == 'occupied').sum())} occupied")
            if len(free):
                lines.append(f"      free: {compress_numbers(free)}")
        lines.append("")
    return "\n".join(lines)


BLOCK_LABELS = {
    "never used": "no tag in the block was ever introduced or seen",
    "used, all dead now": "every tag free (the last bee on each is dead)",
    "dead except uncertain": "no live bee, but some tags still uncertain",
    "mostly dead": ">= 75 % of the tags free",
    "partly in use": "25-75 % of the tags free",
    "mostly in use": "< 25 % of the tags free",
    "in use": "no free tag",
}


def neighbor_leak(status: pd.DataFrame, *, max_ratio: float = 0.6, window=(-5, 1)) -> pd.Series:
    """Flag live tags whose bee is probably a misread of a one-bit neighbour.

    Applies to tags whose current 'bee' came on without a log entry (source 'data'): the
    onset falls within `window` days of the start of a one-bit neighbour's current bee
    (any of the 11 number bits, or the twin), and the tag's recent count is below
    `max_ratio` x that neighbour's. Such a tag reads as occupied but most likely carries
    no bee of its own.
    """
    st = status.set_index("bee_id")
    start = pd.to_datetime(st["onset_day"]).fillna(pd.to_datetime(st["intro_date"]))
    rec = st["recent_median"]
    out = pd.Series(False, index=st.index)
    cand = st.index[(st["source"] == "data") & st["status"].isin(["occupied", "uncertain"])]
    for b in cand:
        on = pd.to_datetime(st.at[b, "onset_day"])
        if pd.isna(on):
            continue
        for k in range(12):
            x = b ^ (1 << k)
            if x not in st.index or pd.isna(start.get(x)):
                continue
            lag = (on - start[x]).days          # neighbour started `lag` days before the onset
            if -window[1] <= lag <= -window[0] and rec[b] < max_ratio * rec[x]:
                out[b] = True
                break
    return out.reindex(status["bee_id"]).to_numpy()


def tag_birth_date(status: pd.DataFrame) -> pd.Series:
    """Birth of the bee now on each tag: the day it was first seen after its introduction
    (tags are put on newly emerged bees), else the logged introduction date."""
    on = pd.to_datetime(status["onset_day"]) if "onset_day" in status else pd.Series(pd.NaT, index=status.index)
    return on.fillna(pd.to_datetime(status["intro_date"]))


def _live_columns(gb: pd.DataFrame, age_date, _d, _median_day) -> dict:
    live = gb[gb["status"].isin(["occupied", "uncertain"])].sort_values("number")
    age = (age_date - live["birth_date"]).dt.days
    recent = live["recent_median"] if "recent_median" in live else pd.Series(np.nan, index=live.index)
    leak = live["flags"].fillna("").str.contains("neighbor_leak")
    items = [f"{int(nb)}:{'' if pd.isna(a) else int(a)}d({'' if pd.isna(r) else round(r / 1000)}k)"
             + ("?" if stt == "uncertain" else "") + ("*" if lk else "")
             for nb, a, r, stt, lk in zip(live["number"], age, recent, live["status"], leak)]
    return {
        "n_live": len(live),
        "n_live_likely_leak": int(leak.sum()),
        "live_birth_median": _d(_median_day(live["birth_date"])),
        "live_age_median": int(age.median()) if age.notna().any() else np.nan,
        "live_recent_k_median": round(float(recent.median()) / 1000) if recent.notna().any() else np.nan,
        "live_bees": " ".join(items),
    }


def block_table(status: pd.DataFrame, intros: pd.DataFrame, *, block: int = 32,
                sub_block: Optional[int] = 16, min_onsets: int = 5, age_date=None) -> pd.DataFrame:
    """One row per parity and block of `block` tag numbers (tags are punched in runs).

    Introductions from the log (all generations, and the current one), the median day the
    current bees were first seen, status counts, when the dead ones died, a summary label
    (see BLOCK_LABELS) and the free numbers. `sub_block` lists the sub-ranges that are
    entirely free.

    For the bees still on the block (occupied + uncertain): median birth date and age on
    `age_date` (default today), and one entry per bee, ``number:age d(recent k/day)``
    with '?' for uncertain, where recent = median daily detections over the last good
    days. A real bee in the hive is typically at 50-300k/day; a tag near the cut (15k)
    whose count is low is the one to look at twice.
    """
    age_date = pd.Timestamp(age_date if age_date is not None else pd.Timestamp.today()).normalize()
    def _d(x):
        return pd.Timestamp(x).strftime("%Y-%m-%d") if pd.notna(x) else ""

    def _median_day(s):
        s = pd.to_datetime(pd.Series(s)).dropna()
        return s.sort_values().iloc[len(s) // 2] if len(s) else pd.NaT

    it = intros.copy()
    it["number"], it["parity"] = from_bee_id(it["bee_id"].to_numpy())
    st = status.copy()
    for c in ("intro_date", "onset_day", "death_q50", "last_detected"):
        if c in st.columns:
            st[c] = pd.to_datetime(st[c])
    st["birth_date"] = tag_birth_date(st)
    free_set = ("free", "never_used")
    rows = []
    for par in ("odd", "even"):
        g = st[st["parity"] == par]
        ip = it[it["parity"] == par]
        for b0 in range(0, ODD_OFFSET, block):
            gb = g[(g["number"] >= b0) & (g["number"] < b0 + block)]
            ib = ip[(ip["number"] >= b0) & (ip["number"] < b0 + block)]
            fl = gb["flags"].fillna("")
            n = len(gb)
            n_free = int((gb["status"] == "free").sum())
            n_never = int((gb["status"] == "never_used").sum())
            n_unc = int((gb["status"] == "uncertain").sum())
            n_occ = int((gb["status"] == "occupied").sum())
            frac_free = (n_free + n_never) / n
            if n_never == n:
                label = "never used"
            elif frac_free == 1:
                label = "used, all dead now"
            elif n_occ == 0:
                label = "dead except uncertain"
            elif frac_free >= 0.75:
                label = "mostly dead"
            elif frac_free > 0.25:
                label = "partly in use"
            elif frac_free > 0:
                label = "mostly in use"
            else:
                label = "in use"
            hist = ib.groupby("intro_date").size()
            cur = gb["intro_date"].dropna().dt.strftime("%Y-%m-%d").value_counts()
            seen = gb[gb["n_days_detected"] > 0]
            on = gb.loc[gb["source"] == "log", "onset_day"].dropna() if "onset_day" in gb else pd.Series(dtype="datetime64[ns]")
            dead = gb[(gb["status"] == "free") & (gb["n_days_detected"] > 0)]
            free_nums = gb.loc[gb["status"].isin(free_set), "number"]
            row = {
                "parity": par, "numbers": f"{b0}-{b0 + block - 1}", "label": label,
                "n_free": n_free + n_never, "n_uncertain": n_unc, "n_occupied": n_occ,
                "intro_log_current": " / ".join(f"{d}" + (f" ({k})" if len(cur) > 1 else "")
                                                for d, k in cur.items()),
                # first-seen day of the bees of the current log introduction (blank when too few
                # were seen, e.g. tagged before recording started)
                "intro_inferred_median": _d(_median_day(on)) if len(on) >= min_onsets else "",
                "n_seen_current": len(seen),
                "death_median": _d(_median_day(dead["death_q50"])),
                "death_first": _d(dead["death_q50"].min()),
                "death_last": _d(dead["death_q50"].max()),
                "n_logged_never_seen": int(fl.str.contains("logged_never_seen").sum()),
                "n_shared_id": int(fl.str.contains("shared_id").sum()),
                "n_unlogged_onset": int(fl.str.contains("unlogged_onset").sum()),
                "intro_history_log": "; ".join(f"{d:%m-%d}" + (f" ({k})" if k != n else "")
                                               for d, k in hist.items()),
                "free_numbers": compress_numbers(free_nums),
                **_live_columns(gb, age_date, _d, _median_day),
            }
            if sub_block and sub_block < block:
                whole = [f"{s}-{s + sub_block - 1}" for s in range(b0, b0 + block, sub_block)
                         if set(range(s, s + sub_block)) <= set(free_nums)]
                row[f"whole_{sub_block}s_free"] = ", ".join(whole)
            rows.append(row)
    return pd.DataFrame(rows)


def format_runs(runs: pd.DataFrame, *, min_len: int = 1) -> str:
    lines = []
    for par in ("odd", "even"):
        r = runs[(runs["parity"] == par) & (runs["n_free"] >= min_len)]
        lines.append(f"{par.upper()} tags: {int(r['n_free'].sum())} free in {len(r)} runs")
        for x in r.itertuples(index=False):
            rng = f"{x.num_start}" if x.num_start == x.num_end else f"{x.num_start}-{x.num_end}"
            s = f"  {par} {rng}: {x.n_free} free"
            if x.n_skip:
                s += f" (skip {x.skip})"
            lines.append(s)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# birth / death per generation, and drop-ins for metrics_pipeline's PyMC estimators
# ---------------------------------------------------------------------------
def _legacy_cut(estimator, default_min: float = 1000.0, default_max: float = 3000.0) -> float:
    """The PyMC estimators' binarisation, clip(n - min, 0, max) / max > 0.5, is n > min + max / 2."""
    mn = float(getattr(estimator, "min_detections", default_min))
    mx = float(getattr(estimator, "max_detections", default_max))
    return mn + mx / 2.0


def _legacy_matrix(dfday: pd.DataFrame, id_col: str, hive, value_col: str = "num_detections"):
    """Per-id series as the legacy per-bee loop builds them: first to last daynum of the id,
    missing days zero-filled, left-aligned. Ids in order of first appearance."""
    dfsel = dfday[dfday["hive"] == hive]
    ids = pd.unique(dfsel[id_col])
    code = pd.Categorical(dfsel[id_col], categories=ids).codes
    dn = dfsel["daynum"].to_numpy(dtype=int)
    first = np.full(len(ids), np.iinfo(np.int64).max, dtype=np.int64)
    last = np.full(len(ids), np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(first, code, dn)
    np.maximum.at(last, code, dn)
    span = last - first + 1
    X = np.zeros((len(ids), int(span.max()) if len(ids) else 1))
    np.add.at(X, (code, dn - first[code]), dfsel[value_col].to_numpy(dtype=float))
    if id_col == "bee_id":
        bee = np.asarray(ids)
    else:
        bee = dfsel.groupby(id_col)["bee_id"].agg(lambda s: s.mode().iloc[0]).reindex(ids).to_numpy()
    return np.asarray(ids), first, span, X, bee


def _chunked(fn, Y, *args, chunk: int = 256, **kw):
    """Apply a row-wise posterior function in row chunks (bounded memory)."""
    parts = [fn(Y[i:i + chunk], *(a[i:i + chunk] if isinstance(a, np.ndarray) and a.ndim and len(a) == len(Y) else a
                                     for a in args), **kw) for i in range(0, len(Y), chunk)]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 1))


def _write(out: pd.DataFrame, output_path) -> pd.DataFrame:
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
    return out


def estimate_birth_days_exact(dfday: pd.DataFrame, *, hives=None, estimator=None,
                              output_path=None, **_ignored) -> pd.DataFrame:
    """Exact drop-in for ``metrics_pipeline.estimate_birth_days``.

    Same series construction (legacy mode: gaps zero-filled, first
    ``min(birth_window, longest span)`` days, detected = n > min + max/2) and the same
    output columns, plus birth_q05 / birth_q95. ``estimator`` may be a
    ``BirthEstimator`` (its prior and detection settings are used; sampler settings are
    irrelevant). The file is rewritten every call: there is nothing to resume.
    """
    has_uid = "uid" in dfday.columns
    id_col = "uid" if has_uid else "bee_id"
    mu = float(getattr(estimator, "mu_emergence", 0.0))
    sigma = float(getattr(estimator, "sigma_emergence", 5.0))
    window = int(getattr(estimator, "birth_window", 14))
    cut = _legacy_cut(estimator)
    spans = dfday.groupby(id_col)["daynum"]
    max_span = int((spans.max() - spans.min() + 1).max()) if len(dfday) else 0
    L = max(1, min(window, max_span) if max_span else window)
    hives = list(pd.unique(dfday["hive"])) if hives is None else list(hives)
    parts = []
    for hive in hives:
        ids, first, span, X, bee = _legacy_matrix(dfday, id_col, hive)
        if not len(ids):
            continue
        Xb = np.zeros((len(ids), L))
        Xb[:, :min(L, X.shape[1])] = X[:, :L]
        post = _chunked(birth_posterior, (Xb > cut).astype(float), mu=mu, sigma=sigma)
        s = posterior_summary(post, 0)
        df = pd.DataFrame({"hive": hive, "bee_id": bee,
                           "estimated_birth_daynum": first + np.round(s["mean"].to_numpy()).astype(int),
                           "birth_q05": first + s["q05"].to_numpy().astype(int),
                           "birth_q95": first + s["q95"].to_numpy().astype(int)})
        if has_uid:
            df.insert(1, "uid", ids)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["hive"] + (["uid"] if has_uid else []) + ["bee_id", "estimated_birth_daynum"])
    return _write(out, output_path)


def estimate_death_days_exact(dfday: pd.DataFrame, *, hives=None, estimator=None,
                              extra_days_after: int = 25, birth_days: Optional[pd.DataFrame] = None,
                              output_path=None, **_ignored) -> pd.DataFrame:
    """Exact drop-in for ``metrics_pipeline.estimate_death_days`` (legacy series
    construction, including the ``extra_days_after`` zero padding and the optional
    ``birth_days`` alive-window start). Adds death_q05 / death_q95 and p_alive_last
    (probability the bee is still alive on its last observed day)."""
    has_uid = "uid" in dfday.columns
    id_col = "uid" if has_uid else "bee_id"
    mu = float(getattr(estimator, "mu_days_alive", 21.0))
    sigma = float(getattr(estimator, "sigma_days_alive", 25.0))
    cut = _legacy_cut(estimator)
    spans = dfday.groupby(id_col)["daynum"]
    max_span = int((spans.max() - spans.min() + 1).max()) if len(dfday) else 0
    fixed_len = max_span + extra_days_after
    lookup = {}
    if birth_days is not None and len(birth_days) and {"hive", id_col, "estimated_birth_daynum"} <= set(birth_days.columns):
        for h, v, b in zip(birth_days["hive"], birth_days[id_col], birth_days["estimated_birth_daynum"]):
            if pd.notna(h) and pd.notna(v) and pd.notna(b):
                lookup[(str(h), int(v))] = float(b)
    hives = list(pd.unique(dfday["hive"])) if hives is None else list(hives)
    parts = []
    for hive in hives:
        ids, first, span, X, bee = _legacy_matrix(dfday, id_col, hive)
        if not len(ids):
            continue
        Xd = np.zeros((len(ids), fixed_len))
        Xd[:, :min(fixed_len, X.shape[1])] = X[:, :fixed_len]
        sp = np.zeros(len(ids), dtype=int)
        if lookup:
            for i, (v, f) in enumerate(zip(ids, first)):
                b = lookup.get((str(hive), int(v)))
                if b is not None:
                    sp[i] = max(0, min(int(np.round(b)) - int(f), fixed_len - 1))
        post = _chunked(death_posterior, (Xd > cut).astype(float), sp, mu=mu, sigma=sigma, n_support=None)
        s = posterior_summary(post, sp)
        df = pd.DataFrame({"hive": hive, "bee_id": bee,
                           "estimated_death_daynum": first + np.round(s["mean"].to_numpy()).astype(int),
                           "death_q05": first + s["q05"].to_numpy().astype(int),
                           "death_q95": first + s["q95"].to_numpy().astype(int),
                           "p_alive_last": 1.0 - prob_at_most(post, sp, span - 2)})
        if has_uid:
            df.insert(1, "uid", ids)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["hive"] + (["uid"] if has_uid else []) + ["bee_id", "estimated_death_daynum"])
    return _write(out, output_path)


def generation_segments(dc: DayCounts, intros: pd.DataFrame, *, last_day=None,
                        pre_intro_days: int = 2) -> pd.DataFrame:
    """One segment per (bee_id, generation): [intro - pre_intro_days, next intro - 1].

    This is the uid of ``uid.assign_uid`` with the misread days before a generation's
    introduction cut off. Ids never introduced get one whole-season segment (uid =
    bee_id). Days before the first introduction of an introduced id are not fitted.
    """
    T_idx = dc.day_index(last_day if last_day is not None else dc.last_good_day())
    it = intros.sort_values(["bee_id", "intro_date"]).copy()
    it["intro_idx"] = (it["intro_date"] - dc.days[0]).dt.days
    it["next_idx"] = it.groupby("bee_id")["intro_idx"].shift(-1)
    seg = pd.DataFrame({
        "hive": dc.hive, "bee_id": it["bee_id"].to_numpy(), "uid": it["uid"].to_numpy(),
        "generation": it["generation"].to_numpy(), "intro_date": it["intro_date"].to_numpy(),
        "start_idx": it["intro_idx"].to_numpy() - pre_intro_days,
        "end_idx": np.minimum(it["next_idx"].fillna(T_idx + 1).to_numpy() - 1, T_idx).astype(int),
        "intro_offset": pre_intro_days,
    })
    never = np.setdiff1d(np.arange(N_IDS), it["bee_id"].unique())
    seg = pd.concat([seg, pd.DataFrame({
        "hive": dc.hive, "bee_id": never, "uid": never, "generation": 0, "intro_date": pd.NaT,
        "start_idx": 0, "end_idx": T_idx, "intro_offset": 0})], ignore_index=True)
    seg = seg[seg["end_idx"] >= seg["start_idx"]]
    return seg.sort_values(["bee_id", "generation"]).reset_index(drop=True)


def estimate_birth_death(daydata, *, dftags: Optional[pd.DataFrame] = None, model: str = "two_step",
                         params: Optional[dict] = None, hive: Optional[str] = None,
                         pre_intro_days: int = 2, birth_window: int = 14, sigma_birth: float = 5.0,
                         dc: Optional[DayCounts] = None, cutoff_day=None,
                         output_path=None) -> pd.DataFrame:
    """Birth and death of every generation (uid) of every tag, exact.

    model='two_step': the legacy chain (birth on the first `birth_window` days of the
    segment, rounded, then death). model='joint': birth and death estimated together.
    Detection rule and masking from `params` (see :data:`DEFAULT_PARAMS`; typically the
    params.json written by :func:`run_free_tags`). The birth prior is centred on the
    introduction day when the segment has one.

    Output columns keep the estimate_birth_days / estimate_death_days schema
    (hive, uid, bee_id, estimated_birth_daynum, estimated_death_daynum) plus quantiles,
    p_alive_last (alive on the last good day), right_censored and p_real (evidence for a
    real bee against 'never alive'; the inclusion gate when tags were not filtered).
    With `cutoff_day`, also p_dead_by_cutoff = P(last alive day <= cutoff_day).
    """
    if model not in ("two_step", "joint"):
        raise ValueError("model must be 'two_step' or 'joint'")
    p = {**DEFAULT_PARAMS, **(params or {})}
    if dc is None:
        dc = load_day_counts(daydata, hive=hive, top_k=p["top_k"], quality_window=p["quality_window"],
                             bad_q=p["bad_q"], rescale_below=p["rescale_below"],
                             manual_bad_days=p["manual_bad_days"])
    if p["cut"] is None:
        raise ValueError("params['cut'] is None: calibrate first (run_free_tags) or set a cut")
    B = binarize(dc, cut=p["cut"], twin_alpha=p["twin_alpha"] or 0.0)
    T = pd.Timestamp(p["last_day"]) if p["last_day"] else dc.last_good_day()
    intros = intro_table(dftags) if dftags is not None and len(dftags) else intro_table(pd.DataFrame(
        columns=["tag_start", "tag_end", "tag_start2", "tag_end2", "Hive", "Date"]))
    intros = intros[intros["hive"].astype(str) == dc.hive]
    seg = generation_segments(dc, intros, last_day=T, pre_intro_days=pre_intro_days)
    Y = _segment_matrix(B, seg["bee_id"].to_numpy(), seg["start_idx"].to_numpy(), seg["end_idx"].to_numpy())
    kw = dict(sigma_birth=sigma_birth, birth_window=birth_window, mu_days_alive=p["mu_days_alive"],
              sigma_days_alive=p["sigma_days_alive"], n_support=p["n_support"])
    out = []
    for mu_b, m in ((float(pre_intro_days), seg["intro_offset"].to_numpy() > 0),
                    (0.0, seg["intro_offset"].to_numpy() == 0)):
        if not m.any():
            continue
        fn = birth_death_posterior if model == "joint" else two_step_posterior
        r = {k: [] for k in ("birth", "death", "p_real")}
        for i in range(0, int(m.sum()), 1024):
            rr = fn(Y[m][i:i + 1024], mu_birth=mu_b, **kw)
            for k in r:
                r[k].append(rr[k])
        birth, death, p_real = (np.concatenate(r[k]) for k in ("birth", "death", "p_real"))
        s0 = seg[m].reset_index(drop=True)
        off = s0["start_idx"].to_numpy()
        bs, ds = posterior_summary(birth, off), posterior_summary(death, off)
        T_idx = dc.day_index(T)
        o = s0[["hive", "uid", "bee_id", "generation", "intro_date"]].copy()
        d0 = int(dc.daynum[0])
        o["estimated_birth_daynum"] = d0 + np.round(bs["mean"].to_numpy()).astype(int)
        o["estimated_death_daynum"] = d0 + np.round(ds["mean"].to_numpy()).astype(int)
        for c in ("q05", "q50", "q95"):
            o[f"birth_{c}"] = d0 + bs[c].to_numpy().astype(int)
            o[f"death_{c}"] = d0 + ds[c].to_numpy().astype(int)
        o["p_alive_last"] = 1.0 - prob_at_most(death, off, np.minimum(s0["end_idx"].to_numpy(), T_idx) - 1)
        o["right_censored"] = (o["p_alive_last"] > 0.5) & (s0["end_idx"].to_numpy() >= T_idx)
        o["p_real"] = p_real
        if cutoff_day is not None:
            o["p_dead_by_cutoff"] = prob_at_most(death, off, dc.day_index(cutoff_day))
            o["death_mean_idx"] = ds["mean"].to_numpy()
        o["n_days_detected"] = np.nansum(Y[m], axis=1).astype(int)
        out.append(o)
    res = pd.concat(out, ignore_index=True).sort_values(["bee_id", "generation"]).reset_index(drop=True)
    res["model"] = model
    return _write(res, output_path)


def model_agreement(daydata=None, *, dftags: pd.DataFrame, params: dict, dc: Optional[DayCounts] = None,
                    margin_days: int = 3, p_free: float = 0.95, p_occupied: float = 0.05,
                    real_p: float = 0.95, min_days: int = 3, max_day_diff: int = 1,
                    min_within: float = 0.95, max_flip_frac: float = 0.01) -> dict:
    """Gate for the joint model: does it change results against the two-step chain?

    Both models on identical inputs. Adopt the joint model only if birth and death daynums
    agree within `max_day_diff` for >= `min_within` of real uids AND the current-tag
    status (free / uncertain / occupied, same rule as :func:`tag_status` without veto and
    sensitivity downgrade) changes for <= `max_flip_frac` of the tags.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    if dc is None:
        dc = load_day_counts(daydata, top_k=p["top_k"], quality_window=p["quality_window"],
                             bad_q=p["bad_q"], rescale_below=p["rescale_below"],
                             manual_bad_days=p["manual_bad_days"])
    T = pd.Timestamp(p["last_day"]) if p["last_day"] else dc.last_good_day()
    cutoff = T - pd.Timedelta(days=margin_days)
    ci = dc.day_index(cutoff)
    fits = {m: estimate_birth_death(None, dc=dc, dftags=dftags, model=m, params=p, cutoff_day=cutoff)
            for m in ("two_step", "joint")}
    key = ["hive", "uid", "bee_id", "generation"]
    m = fits["two_step"].merge(fits["joint"], on=key, suffixes=("_2s", "_j"))
    real = (m["p_real_2s"] >= real_p) & (m["p_real_j"] >= real_p) & (m["n_days_detected_2s"] >= min_days)
    m["real"] = real
    m["birth_diff"] = m["estimated_birth_daynum_j"] - m["estimated_birth_daynum_2s"]
    m["death_diff"] = m["estimated_death_daynum_j"] - m["estimated_death_daynum_2s"]
    status = {}
    for mod, f in fits.items():
        cur = f[f["intro_date"].isna() | (f["intro_date"] <= T)].sort_values(["bee_id", "generation"]) \
            .groupby("bee_id").tail(1).set_index("bee_id")
        pd_ = cur["p_dead_by_cutoff"]
        new = cur["intro_date"] > cutoff
        status[mod] = pd.Series(np.where(new | (pd_ <= p_occupied), "occupied",
                                         np.where((pd_ >= p_free) & (cur["death_mean_idx"] <= ci), "free",
                                                  "uncertain")), index=cur.index)
    st = pd.DataFrame(status)
    out = {
        "table": m,
        "n_real": int(real.sum()),
        "birth_within": float((m.loc[real, "birth_diff"].abs() <= max_day_diff).mean()),
        "death_within": float((m.loc[real, "death_diff"].abs() <= max_day_diff).mean()),
        "status": st,
        "status_crosstab": pd.crosstab(st["two_step"], st["joint"]),
        "status_flip_frac": float((st["two_step"] != st["joint"]).mean()),
    }
    out["adopt_joint"] = (out["birth_within"] >= min_within and out["death_within"] >= min_within
                          and out["status_flip_frac"] <= max_flip_frac)
    return out


# ---------------------------------------------------------------------------
# figures (matplotlib imported lazily)
# ---------------------------------------------------------------------------
def plot_evidence_heatmap(dc: DayCounts, status: pd.DataFrame, dftags: Optional[pd.DataFrame],
                          *, parity: str, reconcile: Optional[pd.DataFrame] = None,
                          path=None, title_extra: str = ""):
    """Tag number x day, log10 daily detections; intro ranges as vertical bars; the
    median death of the current generation as dots; status strip on the right."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ids = to_bee_id(np.arange(ODD_OFFSET), [parity] * ODD_OFFSET)
    img = np.log10(dc.norm[ids] + 1.0)
    fig, (ax, axs) = plt.subplots(1, 2, figsize=(16, 11), sharey=True,
                                  gridspec_kw={"width_ratios": [40, 1], "wspace": 0.02})
    x0 = 0
    im = ax.imshow(img, aspect="auto", origin="lower", interpolation="nearest",
                   cmap="magma", vmin=2, vmax=5.5, extent=(-0.5, len(dc.days) - 0.5, -0.5, ODD_OFFSET - 0.5))
    for j in np.where(~dc.good)[0]:
        ax.axvspan(j - 0.5, j + 0.5, color="0.6", alpha=0.9, lw=0)
    if dftags is not None and len(dftags):
        verd = {} if reconcile is None or reconcile.empty else dict(
            zip(reconcile["row_id"].astype(str), reconcile["verdict"]))
        for r in dftags[dftags["parity"] == parity].itertuples(index=False):
            j = dc.day_index(r.Date)
            ok = verd.get(str(r.row_id), "confirmed") in ("confirmed", "left_censored")
            ax.plot([j - 0.5, j - 0.5], [r.num_start, r.num_end], color="cyan" if ok else "red", lw=2)
    s = status[status["parity"] == parity].sort_values("number")
    dd = (pd.to_datetime(s["death_q50"]) - dc.days[0]).dt.days
    m = s["status"].isin(["free", "uncertain"]).to_numpy()
    ax.scatter(dd[m], s["number"][m], s=1, c="w", lw=0)
    ticks = np.arange(0, len(dc.days), 7)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dc.days[t].strftime("%m-%d") for t in ticks], rotation=90)
    first_intro = min([0] + ([dc.day_index(d) for d in dftags["Date"]] if dftags is not None and len(dftags) else []))
    ax.set_xlim(first_intro - 1.5, len(dc.days) - 0.5)
    ax.set_ylabel(f"{parity} tag number")
    ax.set_title(f"{parity} tags: log10 daily detections (grey = masked day; bars = log intro, "
                 f"red = not confirmed; dots = median death){title_extra}")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="log10(detections + 1)")
    code = s["status"].map({k: i for i, k in enumerate(STATUS_ORDER)}).to_numpy()[:, None]
    axs.imshow(code, aspect="auto", origin="lower", interpolation="nearest",
               cmap=ListedColormap([STATUS_COLORS[k] for k in STATUS_ORDER]), vmin=0,
               vmax=len(STATUS_ORDER) - 1, extent=(0, 1, -0.5, ODD_OFFSET - 0.5))
    axs.set_xticks([])
    axs.set_title("status", fontsize=8)
    if path is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_id(dc: DayCounts, bee_id: int, *, status: Optional[pd.DataFrame] = None,
            cut: Optional[float] = None, B: Optional[np.ndarray] = None, params: Optional[dict] = None,
            path=None):
    """Counts of one id and its twin, the cut, masked days, and the current death pmf."""
    import matplotlib.pyplot as plt

    x = np.arange(len(dc.days))
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.semilogy(x, np.nan_to_num(dc.norm[bee_id]) + 1, "k.-", lw=0.8, label=f"bee_id {bee_id}")
    ax.semilogy(x, np.nan_to_num(dc.norm[twin(bee_id)]) + 1, ".-", color="0.6", lw=0.6,
                label=f"twin {int(twin(bee_id))}")
    for j in np.where(~dc.good)[0]:
        ax.axvspan(j - 0.5, j + 0.5, color="0.85", lw=0)
    if cut:
        ax.axhline(cut, color="b", ls="--", lw=0.8, label=f"cut {cut:g}")
    if status is not None and B is not None:
        p = {**DEFAULT_PARAMS, **(params or {})}
        r = status.set_index("bee_id").loc[bee_id]
        T_idx = dc.day_index(dc.last_good_day() if p["last_day"] is None else p["last_day"])
        post, first = fit_segments(B, [bee_id], [r["start_idx"]], [T_idx], [r["sp"]],
                                   mu=p["mu_days_alive"], sigma=p["sigma_days_alive"],
                                   n_support=p["n_support"])
        ax2 = ax.twinx()
        xs = first[0] + np.arange(post.shape[1])
        keep = xs < len(dc.days) + 30
        ax2.fill_between(xs[keep], post[0][keep], color="r", alpha=0.3, step="mid")
        ax2.set_ylabel("P(last alive day)")
        ax.axvline(r["start_idx"], color="g", lw=1)
        ax.axvline(T_idx - p["margin_days"], color="r", lw=1, ls=":")
        ax.set_title(f"{r['parity']} {r['number']}  status={r['status']}  "
                     f"P(dead by cutoff)={r['p_dead_by_cutoff']:.3f}  {r['flags']}")
    ticks = np.arange(0, len(dc.days), 7)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dc.days[t].strftime("%m-%d") for t in ticks])
    ax.legend(loc="lower left", fontsize=7)
    if path is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_calibration(table: pd.DataFrame, chosen: dict, path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for a, g in table.groupby("twin_alpha"):
        ax.plot(g["false_worst"], g["miss_old"], "o-", label=f"twin_alpha={a:g}")
        for r in g.itertuples(index=False):
            ax.annotate(f"{r.cut / 1000:g}k", (r.false_worst, r.miss_old), fontsize=7)
    c = table[(table["cut"] == chosen["cut"]) & (table["twin_alpha"] == chosen["twin_alpha"])]
    ax.plot(c["false_worst"], c["miss_old"], "k*", ms=15, label="chosen")
    ax.set_xlabel("false-detection rate, worst twin stratum (negatives)")
    ax.set_ylabel("miss rate, alive bees > 21 days old")
    ax.legend(fontsize=8)
    if path is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
_README = """Tag status, hive {hive}, data through {last} (last complete recording day)

A tag number is FREE when the bee currently wearing it (its latest introduction in the tag
log) is estimated dead by {cutoff} with probability >= {p_free}, or it was never used.
UNCERTAIN: the model cannot decide yet (recently gone, weak or flickering signal, or signal
that could be a live bee seen rarely) -- do not use; re-check with more data.
OCCUPIED: a bee is on the tag.

A day counts as "detected" above {cut:g} detections (calibrated){alpha_note}. Days with missing
or partial recording are ignored. Dead tags never drop to zero detections: other tags are
sometimes misread as them, above all the same number on the other parity sheet (the "twin").

free_tags.txt        free numbers per parity and block of 128, plus two checks for the tagger:
                     logged-but-never-seen numbers, and numbers re-tagged while the previous
                     bee was probably still alive (two bees now share them)
tag_blocks_32.csv    one row per block of 32 numbers (tag_blocks_16.csv: blocks of 16): summary
                     label, free/uncertain/occupied counts, logged and first-seen introduction of
                     the current bees, when the dead ones died, the free numbers, whole free 16s;
                     for the bees still on tags: median birth, median age (as of {age_date}), and
                     live_bees = number:age d(median detections/day over the last good days, k),
                     '?' = uncertain, '*' = probably a misread of a one-bit neighbour (appeared
                     without a log entry right after a neighbour number was tagged, and reads
                     well below it). Real bees are typically at 50-300k/day.
tag_status.csv       one row per tag: status, flags, P(dead by cutoff), death estimate, ...
                     flags: shared_id = re-tagged while the previous bee was probably still
                     alive (two bees on one number); neighbor_leak = the 'bee' is probably a
                     misread of a one-bit neighbour; logged_never_seen = logged but never seen
                     after; unlogged_onset = a bee came on with no log row; recent_low_signal /
                     veto_recent_detection / threshold_sensitive = reasons a tag is 'uncertain'
free_runs.csv/.txt   runs of consecutive free numbers (free_blocks.csv: gaps <= 3 bridged)
log_discrepancies.csv  every tag-log row checked against the detections (verdict column)
unexplained_onsets.csv groups of tags that came on with no log row explaining them
reuse_conflicts.csv  every re-introduction: was the previous bee still alive?
calibration.csv, sensitivity.csv, day_quality.csv, params.json   how the call was made
figures/             evidence heatmaps (tag number x day) and the calibration plot
"""


def run_free_tags(daydata, *, tag_log=None, dftags: Optional[pd.DataFrame] = None,
                  out_dir=None, params: Optional[dict] = None, hive: Optional[str] = None,
                  figures: bool = True, verbose: bool = True) -> dict:
    """Full chain: counts -> onsets -> log reconciliation -> calibration -> status -> runs."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    say = print if verbose else (lambda *a, **k: None)

    dc = load_day_counts(daydata, hive=hive, **{k: p[k] for k in
                         ("top_k", "bad_q", "rescale_below", "manual_bad_days")},
                         quality_window=p["quality_window"])
    T = pd.Timestamp(p["last_day"]) if p["last_day"] else dc.last_good_day()
    bad = dc.quality[~dc.quality["good"]]
    say(f"hive {dc.hive}: {dc.days[0].date()}..{dc.days[-1].date()}, last good day {T.date()}; "
        f"masked: {', '.join(f'{d:%m-%d}({r})' for d, r in zip(bad['day'], bad['reason']))}")

    log = read_tag_log(tag_log) if tag_log is not None and not isinstance(tag_log, pd.DataFrame) else tag_log
    if dftags is None:
        dftags = log_to_dftags(log) if log is not None else pd.DataFrame(
            columns=["tag_start", "tag_end", "tag_start2", "tag_end2", "Hive", "Date", "parity", "row_id"])
    dftags = dftags.dropna(subset=["tag_start"]) if "tag_start" in dftags.columns else dftags
    if "parity" not in dftags.columns:
        dftags = dftags.assign(parity=np.where(dftags["tag_start"] >= ODD_OFFSET, "odd", "even"),
                               num_start=dftags["tag_start"] % ODD_OFFSET,
                               num_end=dftags["tag_end"] % ODD_OFFSET)
    if "row_id" not in dftags.columns:
        dftags = dftags.assign(row_id=[f"r{i}" for i in range(len(dftags))])
    intros = intro_table(dftags)
    intros = intros[intros["hive"].astype(str) == dc.hive]

    B_on = binarize(dc, cut=p["onset_cut"], twin_alpha=p["onset_twin_alpha"])
    onsets = explain_onsets(detect_onsets(B_on, dc), intros, tol=tuple(p["onset_tol"]))
    recon = reconcile_log(dc, log, B_on, onsets, tol=tuple(p["onset_tol"])) if log is not None else pd.DataFrame()
    clusters = unexplained_onset_clusters(onsets)
    if len(recon):
        say("log rows by verdict: " + recon["verdict"].value_counts().to_dict().__repr__())
    say(f"unexplained onset clusters: {len(clusters)}")

    since = T - pd.Timedelta(days=int(p["calibration_days"])) if p["calibration_days"] else None
    ctl = calibration_controls(dc, intros, onsets, B_strict=B_on, since=since, last_day=T,
                               mu=p["mu_days_alive"], sigma=p["sigma_days_alive"],
                               n_support=p["n_support"])
    cal_table, chosen = calibrate(ctl, dc)
    cut = p["cut"] if p["cut"] is not None else chosen["cut"]
    alpha = p["twin_alpha"] if p["twin_alpha"] is not None else chosen["twin_alpha"]
    say(f"detection rule: count > {cut:g}" + (f" and > {alpha:g} x twin" if alpha else "")
        + ("  (calibrated)" if p["cut"] is None else "  (fixed)"))

    B = binarize(dc, cut=cut, twin_alpha=alpha)
    segments = build_segments(dc, intros, onsets, last_day=T)
    conflicts = reuse_conflicts(dc, B, intros, onsets, mu=p["mu_days_alive"],
                                sigma=p["sigma_days_alive"], n_support=p["n_support"])
    st_kw = {k: p[k] for k in ("margin_days", "p_free", "p_occupied", "mu_days_alive",
                               "sigma_days_alive", "n_support", "strong_cut", "strong_twin_ratio",
                               "low_signal_cut", "low_signal_days", "sensitive_twin_alpha")}
    st_kw["veto_days"] = tuple(p["veto_days"])
    status = tag_status(dc, B, segments, last_day=T, conflicts=conflicts, **st_kw)
    B_loose = binarize(dc, cut=cut * p["sensitive_cut_factor"],
                       twin_alpha=max(alpha, p["sensitive_twin_alpha"]))
    status = downgrade_sensitive(status, tag_status(dc, B_loose, segments, last_day=T, **st_kw))
    lk = neighbor_leak(status)
    status.loc[lk, "flags"] = [";".join([f for f in (x, "neighbor_leak") if f]) for x in status.loc[lk, "flags"]]

    sens = []
    for c in cal_table["cut"].unique():
        for a in cal_table["twin_alpha"].unique():
            s = tag_status(dc, binarize(dc, cut=c, twin_alpha=a), segments, last_day=T, **st_kw)
            row = {"cut": c, "twin_alpha": a}
            for par in ("even", "odd"):
                v = s.loc[s["parity"] == par, "status"].value_counts()
                row[f"{par}_free"] = int(v.get("free", 0) + v.get("never_used", 0))
                row[f"{par}_occupied"] = int(v.get("occupied", 0))
                row[f"{par}_uncertain"] = int(v.get("uncertain", 0))
            row["flips_vs_chosen"] = int((s["status"].to_numpy() != status["status"].to_numpy()).sum())
            sens.append(row)
    sens = pd.DataFrame(sens)

    runs = free_runs(status)
    blocks = free_runs(status, max_gap=3)
    age_date = pd.Timestamp(p["age_date"]) if p.get("age_date") else pd.Timestamp.today().normalize()
    tables = {b: block_table(status, intros, block=b, sub_block=16 if b > 16 else None, age_date=age_date)
              for b in (32, 16)}
    counts = status.groupby(["parity", "status"]).size().unstack(fill_value=0)
    say(counts.to_string())

    used = {**p, "cut": cut, "twin_alpha": alpha, "last_day": str(T.date()),
            "calibrated": chosen, "hive": dc.hive,
            "masked_days": [str(d.date()) for d in bad["day"]]}
    res = {"daycounts": dc, "dftags": dftags, "intros": intros, "onsets": onsets,
           "reconcile": recon, "unexplained": clusters, "controls": ctl,
           "calibration": cal_table, "sensitivity": sens, "segments": segments,
           "conflicts": conflicts, "status": status, "runs": runs, "blocks": blocks,
           "block_tables": tables,
           "params": used, "B": B}

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        status = status.assign(birth_date=tag_birth_date(status).dt.date)
        status["age_today"] = np.where(status["status"].isin(["occupied", "uncertain"]),
                                       (age_date - pd.to_datetime(status["birth_date"])).dt.days, np.nan)
        res["status"] = status
        cols = ["parity", "number", "bee_id", "status", "flags", "birth_date", "age_today",
                "p_dead_by_cutoff", "p_alive_last",
                "death_q50", "death_q05", "death_q95", "last_detected", "n_days_detected",
                "n_days_observed", "recent_strong_days", "recent_median", "intro_date", "onset_day",
                "source", "generation",
                "uid", "row_id", "twin_status"]
        status.sort_values(["parity", "number"])[cols].to_csv(out / "tag_status.csv", index=False)
        runs.to_csv(out / "free_runs.csv", index=False)
        for b, tb in tables.items():
            tb.to_csv(out / f"tag_blocks_{b}.csv", index=False)
        blocks.to_csv(out / "free_blocks.csv", index=False)
        header = (f"Free tags, hive {dc.hive}, as of {T.date()} (last complete recording day).\n"
                  f"Free = the current bee on the tag is estimated dead by {(T - pd.Timedelta(days=p['margin_days'])).date()} "
                  f"(P >= {p['p_free']}), or the tag was never used.\n\n")
        fl = status["flags"].fillna("")
        extra = ["", "FREE BUT LOGGED AS TAGGED AND NEVER SEEN AFTERWARDS (check these tags were "
                 "not applied to bees that are simply not being detected):"]
        for par in ("odd", "even"):
            m = (status["parity"] == par) & (status["status"] == "free") & fl.str.contains("logged_never_seen")
            extra.append(f"  {par} ({int(m.sum())}): {compress_numbers(status.loc[m, 'number'])}")
        if len(conflicts):
            extra += ["", "RE-TAGGED WHILE THE PREVIOUS BEE WAS PROBABLY STILL ALIVE (two bees now "
                      "share the number; their data cannot be separated):"]
            cs = conflicts[conflicts["conflict"]]
            for (d, par), g in cs.groupby([cs["intro_date"].dt.date, "parity"]):
                extra.append(f"  tagged {d} {par} ({len(g)}): {compress_numbers(g['number'])}")
        (out / "free_tags.txt").write_text(header + format_blocks(status) + "\n".join(extra) + "\n")
        (out / "free_runs.txt").write_text(
            header + "STRICT RUNS\n" + format_runs(runs) + "\nPRACTICAL BLOCKS (gaps <= 3 skipped)\n"
            + format_runs(blocks, min_len=8))
        recon.to_csv(out / "log_discrepancies.csv", index=False)
        clusters.to_csv(out / "unexplained_onsets.csv", index=False)
        conflicts.to_csv(out / "reuse_conflicts.csv", index=False)
        cal_table.to_csv(out / "calibration.csv", index=False)
        sens.to_csv(out / "sensitivity.csv", index=False)
        dc.quality.to_csv(out / "day_quality.csv", index=False)
        (out / "params.json").write_text(json.dumps(used, indent=2, default=str))
        (out / "README.txt").write_text(_README.format(
            age_date=age_date.date(), hive=dc.hive, last=T.date(), cutoff=(T - pd.Timedelta(days=p["margin_days"])).date(),
            p_free=p["p_free"], cut=cut,
            alpha_note=f" and above {alpha:g} x the twin's count" if alpha else ""))
        if figures:
            fd = out / "figures"
            fd.mkdir(exist_ok=True)
            dftags_h = dftags[dftags["Hive"].astype(str) == dc.hive]
            for par in ("odd", "even"):
                plot_evidence_heatmap(dc, status, dftags_h, parity=par, reconcile=recon,
                                      path=fd / f"heatmap_{par}.png")
            plot_calibration(cal_table, {"cut": cut, "twin_alpha": alpha}, path=fd / "calibration.png")
        say(f"wrote {out}")
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("daydata", help="daydatamat.csv (hive, bee_id, day, num_detections)")
    ap.add_argument("--log", help="tag log CSV (raw + corrected columns)")
    ap.add_argument("--tags", help="dftags.csv in bee_id space (alternative to --log)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--hive", default=None)
    ap.add_argument("--params", help="JSON file overriding DEFAULT_PARAMS")
    ap.add_argument("--last-day", default=None)
    ap.add_argument("--margin", type=int, default=None)
    ap.add_argument("--cut", type=float, default=None, help="fixed detection cut (default: calibrated)")
    ap.add_argument("--twin-alpha", type=float, default=None)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    params = json.loads(Path(a.params).read_text()) if a.params else {}
    for k, v in (("last_day", a.last_day), ("margin_days", a.margin), ("cut", a.cut),
                 ("twin_alpha", a.twin_alpha)):
        if v is not None:
            params[k] = v
    dftags = pd.read_csv(a.tags) if a.tags else None
    run_free_tags(a.daydata, tag_log=a.log, dftags=dftags, out_dir=a.out, params=params,
                  hive=a.hive, figures=not a.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
