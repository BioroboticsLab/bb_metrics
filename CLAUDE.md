# bb_metrics

Turns BeesBook tracking output (tracked detections from the hive cameras, plus feeder/exit Pi-camera
detections) into calibrated trajectories, per-bee behavioural metrics, visit tables, day matrices
and birth/death estimates. Experiment notebooks call it; `notebook_templates/` shows the intended
order (0 calibration/tags → 1 trajectories, feeder cams, comb, all-detections → 2 metrics → 3 analysis).

Read first:
- **METRICS.md**: what every metric column means, its units, thresholds and code location.
- **README.md**: pipeline walkthrough and API overview.

## Modules
- `trajectories.py`: tracked dills → per-camera-hour trajectory parquets (rotation, calibration, filters).
- `metrics_pipeline.py`: pairing files, dispatching metrics, visit tables, day matrix, day xyhist.
- `metricsfunctions.py`: per-segment metric computation (`get_metrics`, `datafile_to_metrics`).
- `datafunctions.py`: loading, timestamps, `infer_fps`, histograms, comb annotations/grids, `GridLookup`, weather.
- `feedercams.py`: Pi feeder/exit detections → daily parquets, avgcounts, hourly averages.
- `calibration.py`: corner points and px/cm from CVAT annotations; constant fallback tables.
- `rotation.py`: recording frame ↔ analysis frame (cw90).
- `lifespan.py`: exact birth/death changepoint model, tag-log reconciliation, free-tag calls.
- `uid.py`: `uid = bee_id + 4096 × generation` from the tag-introduction table.
- `saved_output.py`: versioned shared tables in `saved_output/<folder>/` (`find`, `load`, `next_version`, `set_latest`) and `questions_dir` for layer 2 output. Used by all experiment repos; storage mounts are passed in as `candidates`, never stored here.
- `displayfunctions.py`: plotting helpers.

## Conventions
- **Config.** Load it by path, then set it:
  `cfg = bb_metrics.load_config(path); bb_metrics.set_config(cfg)`. `load_config` caches by
  path, so restart the kernel after editing a config.
- **This repo is public.** Never add experiment configs, real storage paths, server names,
  weather-station ids or treatment schedules. `config/example_config.py` is the sanitized
  template.
- **Coordinates.** Raw detections, comb annotations and backgrounds are in the recording frame.
  Trajectories, metrics and comb grids are in the analysis frame, rotated by
  `cfg.camera_rotation` (`rotation.py`). `x_hive`/`y_hive` are cm from the corner point, and
  `y_hive` is negative on the comb.
- **Time.** Hive-camera data is UTC. Pi-camera data is Europe/Berlin. Day boundaries use local
  time. Parquet files may hold `[us]` or `[ns]` timestamps; compare in one unit (see `_to_ns` in
  `datafunctions.py`).
- **fps is inferred** from timestamps (`infer_fps`), not configured. Detection-count thresholds
  are expressed in tracked seconds and converted per file.
- **Recompute rules.** Most outputs are skipped when newer than their inputs (`update`/`recalc`),
  and `reprocess=True` forces a rebuild. `daydatamat.csv` and `dayxyhist.h5` are always
  overwritten. Comb grid changes do not change trajectory mtimes, so metrics need
  `reprocess=True` after them.
- **Fixed column order.** Do not reorder or rename metric columns. Downstream notebooks rely on
  the order (see the comment above `compute_frame_exit_metrics` in `get_metrics`).

## Development
- Tests import the package as `bb_metrics`. Run them from the parent directory with
  `python -m pytest bb_metrics/tests`, or from anywhere after `pip install -e .`.
  If `feedercams` tests fail with `'NoneType' object has no attribute 'timezone'`, the env has an
  old pytz (pulled in by meteostat). Fix with `pip install "pytz>=2024.2"`.
- When the API changes, update `notebook_templates/` and the README in the same change.
- When a metric definition or threshold changes, update METRICS.md.
