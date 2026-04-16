"""
UID (unique ID) support for reused tag IDs.

When physical tags are reused on new bees later in the season, the same
bee_id refers to different individuals at different times.  This module
assigns a unique ``uid`` to each (hive, bee_id, generation) combination:

    uid = bee_id + generation * 4096

Generation 0 is the first use, generation 1 is the first reuse, etc.
"""

import numpy as np
import pandas as pd


def build_reuse_intervals(dftags: pd.DataFrame) -> pd.DataFrame:
    """Build a lookup table mapping (hive, bee_id) to generation and uid.

    Parameters
    ----------
    dftags : pd.DataFrame
        Tag introduction table with columns
        ``[tag_start, tag_end, Hive, Date]`` and optionally
        ``[tag_start2, tag_end2]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``[hive, bee_id, generation, intro_date, uid]``.
        One row per (hive, bee_id, generation).
    """
    dftags = dftags.copy()
    dftags["Date"] = pd.to_datetime(dftags["Date"])
    # Strip timezone for consistent tz-naive comparison downstream
    if hasattr(dftags["Date"].dt, "tz") and dftags["Date"].dt.tz is not None:
        dftags["Date"] = dftags["Date"].dt.tz_localize(None)

    records = []
    for _, row in dftags.iterrows():
        hive = row["Hive"]
        intro_date = row["Date"]
        for a, b in (("tag_start", "tag_end"), ("tag_start2", "tag_end2")):
            if pd.notna(row.get(a)) and pd.notna(row.get(b)):
                start, end = int(row[a]), int(row[b])
                for bee_id in range(start, end + 1):
                    records.append(
                        {"hive": hive, "bee_id": bee_id, "intro_date": intro_date}
                    )

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["hive", "bee_id", "generation", "intro_date", "uid"])

    # Sort so that earlier introductions come first
    df = df.sort_values(["hive", "bee_id", "intro_date"]).reset_index(drop=True)

    # Assign generation number within each (hive, bee_id) group
    df["generation"] = df.groupby(["hive", "bee_id"]).cumcount()
    df["uid"] = df["bee_id"] + df["generation"] * 4096

    return df[["hive", "bee_id", "generation", "intro_date", "uid"]]


def assign_uid(
    df: pd.DataFrame,
    dftags: pd.DataFrame,
    *,
    date_col: str = "day",
    hive_col: str = "hive",
    bee_id_col: str = "bee_id",
) -> pd.Series:
    """Assign a uid to each row based on its date and bee_id.

    For each row, finds the latest introduction date <= the row's date
    for that (hive, bee_id) and returns the corresponding uid.

    Parameters
    ----------
    df : pd.DataFrame
        Data with date, hive, and bee_id columns.
    dftags : pd.DataFrame
        Tag introduction table (passed to :func:`build_reuse_intervals`).
    date_col, hive_col, bee_id_col : str
        Column names in *df*.

    Returns
    -------
    pd.Series
        uid values aligned with *df*'s index.
    """
    intervals = build_reuse_intervals(dftags)
    if intervals.empty:
        return df[bee_id_col].copy()

    # Normalise dates to tz-naive datetime64[ns] for merge_asof compatibility
    row_dates = pd.to_datetime(df[date_col])
    if hasattr(row_dates.dt, "tz") and row_dates.dt.tz is not None:
        row_dates = row_dates.dt.tz_localize(None)
    row_dates = row_dates.dt.normalize()

    intervals = intervals.copy()
    intro_dates = pd.to_datetime(intervals["intro_date"])
    if hasattr(intro_dates.dt, "tz") and intro_dates.dt.tz is not None:
        intro_dates = intro_dates.dt.tz_localize(None)
    intervals["intro_date_norm"] = intro_dates.dt.normalize()

    # For each row, merge_asof to find the latest intro_date <= row date
    lookup = df[[hive_col, bee_id_col]].copy()
    lookup["_date"] = row_dates.values
    lookup["_idx"] = np.arange(len(lookup))

    # Ensure bee_id dtype matches between lookup and intervals
    lookup[bee_id_col] = lookup[bee_id_col].astype(int)
    intervals["bee_id"] = intervals["bee_id"].astype(int)

    # Ensure matching datetime resolution for merge_asof
    common_res = "us"
    lookup["_date"] = lookup["_date"].astype(f"datetime64[{common_res}]")
    intervals["intro_date_norm"] = intervals["intro_date_norm"].astype(f"datetime64[{common_res}]")

    # Sort both sides for merge_asof
    lookup_sorted = lookup.sort_values("_date")
    intervals_sorted = intervals.sort_values("intro_date_norm")

    merged = pd.merge_asof(
        lookup_sorted,
        intervals_sorted.rename(columns={"intro_date_norm": "_date"}),
        on="_date",
        by=[hive_col, bee_id_col],
        direction="backward",
    )

    # Restore original order
    merged = merged.sort_values("_idx")
    uid_series = merged["uid"].values

    # Fall back to bee_id where no match was found (no reuse info)
    mask = np.isnan(uid_series) if merged["uid"].dtype == float else pd.isna(merged["uid"])
    result = pd.Series(uid_series, index=df.index, dtype="Int64")
    result[mask] = df.loc[result[mask].index, bee_id_col].astype("Int64")
    return result
