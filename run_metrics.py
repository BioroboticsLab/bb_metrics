"""
Batch worker for cluster metrics runs: process a shard of hive-hour pairs serially.

    python -m bb_metrics.run_metrics --run-dir <metrics_dir>/_cluster/<td>-<stamp> <shard.jsonl>

All parameters (config, time division, gate, comb grids, ...) come from <run-dir>/RUN.json,
written by metrics_pipeline.submit_metrics_cluster. The shard is a subset of
<run-dir>/pairs.jsonl. One status line per pair is appended to <run-dir>/status/.
Exits non-zero if any pair errored.
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import bb_metrics
from . import datafunctions as dfunc
from . import metricsfunctions as mfunc
from . import metrics_pipeline as mp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Run directory holding RUN.json")
    p.add_argument("shard", help="JSONL file of pairs to process")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    run = json.loads((run_dir / "RUN.json").read_text())

    cfg = bb_metrics.load_config(run["config_path"])
    bb_metrics.set_config(cfg)
    dfunc.init(cfg)

    grid_lookup = None
    if run.get("grid_parquet"):
        ann_df = pd.read_parquet(run["grid_parquet"])
        ann_df["grid_path"] = ann_df["grid_path"].astype(str)
        grid_lookup = dfunc.GridLookup(ann_df)
        grid_lookup.check_frame("analysis")

    commit = mp.bb_metrics_commit()
    td = run["time_division"]
    pairs = mp.read_pairs_file(args.shard)
    status_dir = run_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / f"{socket.gethostname()}-{os.getpid()}-{Path(args.shard).stem}.jsonl"
    print(f"{len(pairs)} pairs, td={td}, bb_metrics {commit}, status -> {status_file}", flush=True)

    n_err = 0
    for hive, start, end, datafiles in pairs:
        t0 = time.time()
        status = mfunc.datafile_to_metrics(
            hive, start, end, datafiles,
            run["reprocess"], td,
            run["min_num_detections"], run["min_track_seconds"],
            run["save_xy_hist"], run["metrics_dir"],
            update=run["update"], grid_lookup=grid_lookup,
        )
        dt = time.time() - t0
        n_err += str(status).startswith("error")
        line = {
            "output": str(mfunc.metrics_file_for(hive, datafiles, td, run["metrics_dir"])),
            "hive": hive,
            "start": start.isoformat(),
            "status": status,
            "seconds": round(dt, 2),
            "finished": datetime.now(timezone.utc).isoformat(),
            "bb_metrics_commit": commit,
            "host": socket.gethostname(),
        }
        with open(status_file, "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"{status:8s} {dt:7.1f}s  Hive {hive} {start}", flush=True)

    print(f"done: {len(pairs)} pairs, {n_err} errors", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
