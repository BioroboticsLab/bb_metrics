# Metrics reference

What each metric calculated by `bb_metrics` is: definition, units, the exact rule, and where it is computed.
This covers the per-segment metrics (`metrics-<td>-*.parquet`, for any time division such as `5min` or `60min`)
and the per-day matrix (`daydatamat.csv`). File locations and loading are documented per experiment, not here.

Code references are `file:line` at the time of writing. Thresholds are module constants in
`metricsfunctions.py:14-23`:

| constant | value | used by |
|---|---|---|
| `max_delay` | 1 s | speed (and therefore all speed-based events) |
| `moving_threshold` | 0.05 cm/s | `inplace_events` |
| `inplace_time_threshold` | 15 s | `inplace_events` |
| `burst_threshold` | 0.3 cm/s | `burst_events` |
| `burst_time_threshold` | 2 s | `burst_events` |
| `large_turn` | 2π rad | `large_turn_events` |
| `large_turn_time` | 2 s | `large_turn_events` |
| `time_outside_trip` | 120 s | `num_trips` |
| `exit_dist_trip_threshold` | 5 cm | `num_trips` |
| `all_detections_dt` | `60s` | `numbees*` (resolution of the all-detections density snapshots) |

---

## 1. Inputs and geometry

Metrics are computed per hive, per time segment, per `bee_id`, from the trajectory parquets of the
hive's two cameras (`datafile_to_metrics`, `metricsfunctions.py:242`).

The coordinates used are these:
- **`x_pixels`, `y_pixels`:** analysis-frame pixels (rotated, top-left origin), per camera.
- **`x_hive`, `y_hive`:** cm from the calibrated corner point (the exit). Axes follow the image, so
  the comb lies at negative `y_hive`.
- **`x_hive_flat`:** the second camera's x mirrored, `frame_width_cm − x_hive`
  (`metricsfunctions.py:325`). This lays the two sides of the comb onto one plane.
- **The two-camera bin grid** (`cfg.x_edges`, `cfg.y_edges`): the second camera is placed to the
  right of the first (x shifted by `cfg.xpixels`). Bins are about `cfg.pixels_per_bin` = 225 px
  (≈ 2 cm), giving `cfg.numxbins × cfg.numybins` bins (42 × 24 for the standard layout).
- **Frame regions:** each camera image is split at `y_hive = −cfg.offset_div_cm` (the divider
  between the two frames on one side of the comb).
  - The *lower* region `[−offset_div_cm, 10)` is the frame next to the corner point / exit.
  - The *upper* region `[−2·offset_div_cm − 10, −offset_div_cm)` is the other frame.

**Calibration dependence.** Metrics in cm (speeds, `dispersion`, the event counts) need
`pixels_per_cm`. `exit_distance_median`, `num_trips` and `frame_*` also need the corner point,
`offset_div_cm` and `frame_width_cm` (`dispersion` needs `frame_width_cm` too). If
`cfg.compute_frame_exit_metrics` is `False` (non-standard comb layouts), `num_trips`,
`exit_distance_median`, all `frame_*` columns and the gap-boundary columns (`exit_edge_dist_*`, `t_*_s`,
`cam_*`) are omitted (`metricsfunctions.py:598`).

## 2. Segments and the inclusion gate

- **Segments:** half-open `[timestamp_start, timestamp_end)` (UTC). Bins are laid on a
  `pd.date_range(..., freq=time_division)` grid starting at the hour the trajectory file starts.
  One output file covers one hive-hour.
- **Gate:** a (bee, segment) row is written only if the bee has at least `min_num_detections`
  detections in the segment.
  - Default: `ceil(min_track_seconds × fps)`, with fps inferred from the file's timestamps
    (`datafunctions.infer_fps`, `datafunctions.py:54`).
  - Default `min_track_seconds`: 15 s for `60min`, 5 s for `5min`, 2 s otherwise
    (`metrics_pipeline.py:85-89`).
  - An explicit `min_num_detections` overrides this and is used as a raw count.
- **A missing row is not zero activity.** It means the bee was tracked for less than the gate in
  that segment.

## 3. Segment metrics (`metrics-<td>-Hive_<H>_<start>--<end>.parquet`)

Computed in `get_metrics` (`metricsfunctions.py:480`) unless noted. Column order is fixed:
`hive, bee_id, timestamp_start, timestamp_end`, then the columns below, in this order.

| column | units | definition |
|---|---|---|
| `num_detections` | count | Trajectory rows for the bee in the segment, after trajectory filtering. **Scales with fps**; tracked seconds = `num_detections / fps`. |
| `dispersion` | cm | RMS distance of the bee's positions from their centroid, using (`x_hive_flat`, `y_hive`) (l.489). |
| `speed_median` | cm/s | Median step speed. A step is two consecutive detections on the **same camera** with `0 < dt ≤ max_delay` (1 s); other steps are NaN (l.497). NaN if 2 or fewer valid steps. |
| `speed_iqr` | cm/s | Interquartile range (q75 − q25) of the same step speeds. |
| `num_trips` | count | Approximate number of trips outside the hive (l.638); see the rule below. |
| `fraction_squares_visited` | 0–1 | Fraction of the two-camera bin grid with at least one detection (l.583). Grows with segment length. |
| `exit_distance_median` | cm | Median distance from the corner point (0, 0), computed in (`x_hive_flat`, y). For second-camera detections in the lower region, y is reflected to `−2·offset_div_cm − y_hive` (l.605), approximating the walking path across the two sides of the comb. |
| `frame_0_hist` … `frame_3_hist` | 0–1 | Fraction of the bee's detections in each frame region (`getframehist`, `datafunctions.py:622`): 0 = first camera upper, 1 = first camera lower, 2 = second camera upper, 3 = second camera lower. Detections outside the region bins are not counted, so the four can sum to less than 1. |
| `frame_0_centermedian` … `frame_3_centermedian` | — | **Do not use.** Intended as the median distance to each frame's centre, but it subtracts cm centres from pixel coordinates, and the upper/lower frame centres are swapped (l.624). |
| `inplace_events` | count | Runs of consecutive valid steps with speed < `moving_threshold` (0.05 cm/s) whose summed duration is ≥ `inplace_time_threshold` (15 s). Any invalid step (camera switch, gap > 1 s) ends a run (l.507). |
| `burst_events` | count | Runs of consecutive steps with speed > `burst_threshold` (0.3 cm/s) whose summed duration is ≥ `burst_time_threshold` (2 s) (l.530). |
| `large_turn_events` | count | Number of times the accumulated absolute orientation change exceeds `large_turn` (2π, one full turn) within a `large_turn_time` (2 s) window (l.552). After an event, or when the window is exceeded, accumulation restarts. |
| `exit_edge_dist_first`, `exit_edge_dist_last` | cm | Distance from the exit edge, `−y_hive`, at the bee's first / last detection in the segment. The exit edge is `y_hive = 0` by definition (the corner point is set there), and the y axis is shared by both cameras. Not clipped: detections in the bottom slot give small negative values. |
| `exit_edge_dist_min` | cm | Minimum of `−y_hive` over the segment (the bee's closest approach to the exit edge). |
| `t_first_s`, `t_last_s` | s | Time from `timestamp_start` to the bee's first / last detection in the segment, in `[0, segment length)`. Lets a gap builder time gaps between segments to the second. |
| `cam_first`, `cam_last` | 0/1 | Side of the comb at the first / last detection: 0 = the hive's first camera (`cam0`), 1 = the second (`cam0 + 1`). |
| `numbees0`, `numbees1`, `numbees2` | detections/frame | Local density (`compute_surrounding_bee_counts`, `metricsfunctions.py:189`); see the rule below. |
| `combhist_<label>` | 0–1 | Only when metrics are run with a `GridLookup`. Fraction of the bee's detections on each comb label (`_compute_combhist`, `metricsfunctions.py:448`); see the rule below. |

### `num_trips` rule
- **Gaps:** count every gap of ≥ `time_outside_trip` (120 s) between consecutive detections where
  both detections on either side of the gap are within `exit_dist_trip_threshold` (5 cm) of the
  exit.
- **End of segment:** add one if the bee's last detection is within 5 cm of the exit and the
  segment ends more than 120 s later.

This is an approximation, and it is **not additive across time divisions**. Short segments
rarely contain an internal gap of ≥ 120 s, and a gap that crosses a segment boundary is caught
only by the end-of-segment rule. Summing 5-min `num_trips` is therefore not equivalent to 60-min
`num_trips`. Use 60-min segments (or the day matrix) for trip counts.

### `numbees` rule
1. For each of the bee's detections, take the all-detections snapshot nearest in time. Snapshots
   come from `datafunctions.process_timestamp_chunk`: one per minute, only those inside the
   segment.
2. Find the bee's bin in the two-camera grid. Sum the density in the 1×1 (`numbees0`), 3×3
   (`numbees1`) and 5×5 (`numbees2`) block around it, clipped at the edges.
3. Average over the bee's detections.

The snapshot density is mean detections per frame per bin. It counts every detection type
(tagged, untagged, ...) and includes the focal bee itself. The values are NaN if no
all-detections data exists for the segment.

### `combhist` rule
- Each detection is labelled with `GridLookup.lookup_labels`: the grid nearest in time for that
  camera, `label_grid[y // ds, x // ds]`.
- `combhist_<label>` = count of detections with that label / `num_detections`. Detections without
  a label are not counted, so the columns can sum to less than 1.
- Labels follow the grid's `label_order`: the first four classes of `config/label_classes.json`
  (`empty_cell, open_brood, capped_honey, capped_brood`), then `other_cell` (every other
  annotated label), then `off_comb` (grid cells far from any annotated cell).

### Notes on interpretation
- **fps and resolution:** `num_detections` and anything compared against a fixed count scale
  with fps. `moving_threshold` is close to a one-pixel step per frame (at ~110 px/cm and 6 fps, a
  1 px step is ≈ 0.055 cm/s), so "stopped" effectively means no pixel-level movement between
  frames, and the share of steps under it depends on fps.
- **Segment length:** `fraction_squares_visited`, the event counts and `num_trips` grow with the
  tracked time in the segment. To compare across segments or bees, normalise by tracked time
  (`num_detections / fps`).
- **Gap-boundary columns** (`exit_edge_dist_*`, `t_*_s`, `cam_*`) describe where and when the bee entered and
  left view within the segment. They are meant for 1-min segments, as inputs to a gap/trip rule, and are not
  carried into the day matrix.
- **`large_turn_events` caveats:**
  - Orientation changes are not restricted to same-camera steps or to `dt ≤ max_delay`.
  - Each orientation change is paired with the time step before it, an off-by-one in the
    `zip(np.abs(angular_change), dtimes)` loop.
  - Treat it as an approximate count.

## 4. Day matrix (`daydatamat.csv`)

Built from segment metrics, normally 60-min (`metrics_pipeline.build_day_data_matrix`,
`metrics_pipeline.py:192`), then visit counts are merged in (`merge_daily_visit_metrics`,
`metrics_pipeline.py:301`). One row per `(hive, bee_id[, uid], day)` for which the bee has at least
one segment row that day.

| column | definition |
|---|---|
| `day` | Local date of `timestamp_start` in `tz` (default `Europe/Berlin`). |
| `daynum` | `day_to_number[day]`; usually `cfg.day_to_number`, where 0 is `cfg.startday`. |
| `uid` | Only when `dftags` is given: `bee_id + 4096 × generation` (`uid.assign_uid`). |
| `num_detections`, `num_trips`, `inplace_events`, `burst_events`, `large_turn_events` | **Sums** over the day's segments (`sumqs`). |
| every other numeric segment column (`dispersion`, `speed_*`, `fraction_squares_visited`, `exit_distance_median`, `frame_*`, `numbees*`, `combhist_*`) | **Mean weighted by `num_detections`**: `nansum(value × w) / nansum(w)` (l.219). The gap-boundary columns (`metricsfunctions.GAP_BOUNDARY_COLS`) are excluded. |
| `speed_circadian_coeff` | `Σ speed_median · sin(2π (h − (peak − 6)) / 24) · w / Σ speed_median · w`, where `h` is the local hour of `timestamp_start` (including minutes), `w = num_detections` and `peak = daysine_peaktime` (the sine peaks at `peak`). Range −1..1: positive means speed concentrated around `peak`, negative around `peak ± 12 h`. Without `daysine_peaktime`, the sine is `sin(2π h / 24)`. |
| `num_feeder_visits`, `num_exit_visits` | Number of visits that **started** on that local day (see §5). 0 if none. |
| `feeder_visit_duration_mean`, `exit_visit_duration_mean` | Mean visit `duration_seconds` that day. **0 (not NaN) when there were no visits**, so exclude zero-visit days before averaging durations. |

Notes:
- The weighted-mean denominator includes the weights of segments where the value is NaN (for
  example `speed_median` with too few valid steps). This biases those daily means low.
- Visits are left-joined. A bee seen at a feeder or exit on a day without any in-hive segment row
  does not appear in the matrix.
- Because the gate drops short segments, the day sums count only gated segments. For 60-min
  segments that means hours with at least 15 s tracked.
- `sumqs` / `weightedmeanqs` can be overridden; the lists above are the defaults.

## 5. Related outputs

**Visits** (`df_feedervisits.parquet`, `df_exitvisits.parquet`; `define_feeder_visits`,
`metrics_pipeline.py:114`). They are built per daily feeder- or exit-camera file:
- Only `TaggedBee` detections with `bee_id_confidence > confidence_threshold` are used.
- The file is sorted per `bee_id`, and a new visit starts after a gap of more than
  `visit_gap_seconds` (default 15 s).
- Columns:
  - `bee_id`
  - `visit_id`: per bee, restarting in each daily file
  - `start_time`, `end_time`: first and last detection
  - `detection_count`
  - `cam_id`: modal camera
  - `duration_seconds = end_time − start_time`

  There is no `hive` column; downstream code uses the last character of `cam_id`.

**Spatial histograms:**
- With `save_xy_hist=True` and segments of at least 60 min, each metrics file has a companion
  `xyhist-<td>-Hive_...h5`. It holds `/<bee_id>/<YYYYMMDD>/<HH>/xyhist`: per-segment detection
  counts on the two-camera bin grid, with `bee_id` / `timestamp_start` / `timestamp_end` as
  attributes.
- `build_day_xyhist` (`metrics_pipeline.py:336`) sums these per local day into
  `/<hive>/<bee_id>/day_<YYYYMMDD>`.

**All-detections density** (`alldetections/hive_<H>_ts_<start>_dt_60s.pklz`;
`process_timestamp_chunk`, `datafunctions.py:1094`):
- A gzip pickle: `{segment_start: {hive: {time_segment_end, num_detections, xyhist, framehist}}}`.
- `num_detections` is detections per frame (all types), and `xyhist` is the per-frame bin density
  used by `numbees*`.
- Segments without detections hold `num_detections = 0` and `xyhist = NaN`.
- `framehist` here is computed from pixel y against cm bins and is not meaningful.
