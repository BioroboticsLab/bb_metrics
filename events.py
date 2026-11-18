"""The generic experimenter-event table: manipulations, disturbances, pesticide pulses, and hand
annotations of observed behaviour tied to them.

One table per experiment, saved as ``saved_output/events/vN/events.parquet``. bb_metrics owns only
the format; parsing an experiment's own spreadsheet stays in the experiment repo.

Required columns (experiments may add more):

    hive       str                     hive / Mosaic group
    event_id   str                     unique within the table, stable within the season
    kind       str                     e.g. pierce, removal, uncapping, disturbance, pulse
    t_start    datetime64[ns, UTC]
    t_end      datetime64[ns, UTC]     >= t_start; equal for an instantaneous event
    cam        Int64 (nullable)        camera the event is seen on; null if not tied to one side
    cell_id    string (nullable)       comb cell id (from the comb annotation), if snapped
    x_px, y_px float (nullable)        location in the ANALYSIS frame (same frame as trajectory
                                       x_pixels/y_pixels and comb grids), before snapping
    notes      string

    from bb_metrics import events as ev
    df = ev.validate_events(df)            # coerces dtypes, raises on a broken table
    df = ev.load_events(cfg)               # newest saved_output/events/ version, validated
    df = ev.snap_to_cells(df, cells)       # fills cell_id from the nearest comb cell
"""
import numpy as np
import pandas as pd

COLUMNS = {
    "hive": "string",
    "event_id": "string",
    "kind": "string",
    "t_start": "datetime64[ns, UTC]",
    "t_end": "datetime64[ns, UTC]",
    "cam": "Int64",
    "cell_id": "string",
    "x_px": "float64",
    "y_px": "float64",
    "notes": "string",
}


def validate_events(df):
    """Return a copy with the required columns in front and coerced to their dtypes.
    Raises ValueError on missing columns, duplicate event_ids, missing hive/kind/times, or
    t_end < t_start."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"events table lacks columns {missing}")
    d = df.copy()
    for c, dt in COLUMNS.items():
        if dt.startswith("datetime"):
            d[c] = pd.to_datetime(d[c], utc=True).astype(dt)
        else:
            d[c] = d[c].astype(dt)
    for c in ("hive", "event_id", "kind", "t_start", "t_end"):
        if d[c].isna().any():
            raise ValueError(f"events: {int(d[c].isna().sum())} rows without {c}")
    dup = d["event_id"][d["event_id"].duplicated()]
    if len(dup):
        raise ValueError(f"events: duplicate event_id {sorted(dup.unique())[:10]}")
    bad = d["t_end"] < d["t_start"]
    if bad.any():
        raise ValueError(f"events: t_end < t_start for {d.loc[bad, 'event_id'].tolist()[:10]}")
    extra = [c for c in d.columns if c not in COLUMNS]
    return d[list(COLUMNS) + extra].reset_index(drop=True)


def load_events(cfg=None, version=None, candidates=None, verbose=True):
    """The newest (or pinned) saved_output/events/ table, validated."""
    from . import saved_output as so
    return validate_events(so.load("events", "events", cfg, version=version, verbose=verbose,
                                   candidates=candidates))


def snap_to_cells(events, cells, max_dist_px=None, cam_col="cam"):
    """Fill cell_id (and add snap_dist_px) from the nearest cell in `cells`.

    `cells` needs columns cell_id, x, y (analysis frame) and optionally cam; with a cam column,
    an event only snaps to cells of its own camera. Events without a location, or farther than
    max_dist_px from every cell, keep a null cell_id.
    """
    ev = events.copy()
    ev["snap_dist_px"] = np.nan
    ev["cell_id"] = ev["cell_id"].astype("string")
    by_cam = "cam" in cells.columns and cam_col in ev.columns
    groups = cells.groupby("cam") if by_cam else [(None, cells)]
    for cam, cg in groups:
        sel = ev["x_px"].notna() & ev["y_px"].notna()
        if by_cam:
            sel &= ev[cam_col] == cam
        if not sel.any() or not len(cg):
            continue
        P = ev.loc[sel, ["x_px", "y_px"]].to_numpy(float)
        C = cg[["x", "y"]].to_numpy(float)
        d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        d = np.sqrt(d2[np.arange(len(P)), j])
        ok = np.ones(len(P), bool) if max_dist_px is None else d <= max_dist_px
        idx = ev.index[sel]
        ev.loc[idx[ok], "cell_id"] = cg["cell_id"].astype(str).to_numpy()[j[ok]]
        ev.loc[idx, "snap_dist_px"] = d
    return ev
