# EXAMPLE config template for bb_metrics.
#
# Real experiment configs are NOT stored in this repo. Copy this file into your
# per-year data/working folder, rename it (e.g. berlin2025.py), fill in the
# placeholders, and load it by path:
#
#     import bb_metrics
#     cfg = bb_metrics.load_config("berlin2025.py")   # path to your copy
#     bb_metrics.set_config(cfg)
#
# Every value below is a placeholder — replace the /path/to/... paths, dates,
# hive/camera topology, and imaging constants with your experiment's real values.
from pathlib import Path
import pandas as pd
import numpy as np
import os

from bb_metrics.config import default_label_config_path

year = 2000
startday = pd.Timestamp(year, 6, 1)   # REPLACE: experiment start date
endday = pd.Timestamp(year, 8, 31)    # REPLACE: experiment end date
weather_station_id = 'XXXXX'          # REPLACE: meteostat station id

# Hive/camera topology — REPLACE with your setup.
hives = ('A', 'B', 'C', 'D')
hive_cam_map = {
    "A": (0, 1),
    "B": (2, 3),
    "C": (4, 5),
    "D": (6, 7),
}
# reverse the mapping → cam_hive_map
cam_hive_map = {
    cam: hive
    for hive, cams in hive_cam_map.items()
    for cam in cams
}

# Filesystem paths — REPLACE with your data locations.
basedir = Path('/path/to/beesbook_year/')
trackdir = basedir / 'data_tracked'
alldetectiondir = basedir / 'data_alldetections'
traj_outdir = Path('/path/to/trajectory_data/')
metrics_dir = basedir / 'metrics'
# Feeder/exit cam data (raw detections and daily aggregates)
feedercam_input_dir = basedir / 'pi'
feedercam_daily_dir = basedir / 'pi_data_alldetections'
feedercam_avg_dir = feedercam_daily_dir / 'avgcounts'
# Local outputs
saved_output_dir = Path(os.path.join(basedir, 'saved_output'))

# Calibration assets — REPLACE with your comb-background locations.
comb_images_root = basedir / 'comb_background'
calib_xml_dir = comb_images_root / 'corner_and_frame_annotations'
annotations_dir = comb_images_root / 'annotations'
annotation_grids_dir = comb_images_root / 'annotation_grids'
# Label configuration (shared, non-secret; resolved from the installed package)
label_config_path = default_label_config_path()

# Optional: pesticide/treatment intervals per feeder (leave empty if none).
treatment_intervals = {}
rows = []
for feeder, intervals in treatment_intervals.items():
    for start_str, end_str in intervals:
        rows.append({
            'feeder': feeder,
            'start': pd.to_datetime(start_str),
            'end': pd.to_datetime(end_str),
        })
treat_df = pd.DataFrame(rows)

## '4-frame representation' and pixel distances (REPLACE / verify per experiment)
offset_div_cm = 23.2
frame_width_cm = 37.6

## pixel and image sizes (rotated / analysis coordinate frame)
ypixels, xpixels = (5312, 4608)

# Camera rotation: 'none', 'cw90', 'ccw90', or '180'
camera_rotation = 'cw90'
KNOWN_LENGTH_CM = 35.0  # for px/cm conversion
SCALE_FACTOR = 2        # if using down-scaled videos

# Spatial histogram bins
pixels_per_bin = 225  # ~ 2 cm
pixels_per_cm_approx = 111.761
numxbins = np.round(2 * xpixels / pixels_per_bin).astype(int)
numybins = np.round(ypixels / pixels_per_bin).astype(int)
numxbins = numxbins + (numxbins % 2)  # ensure xbins is even, so it divides the middle
x_edges = np.linspace(0, 2 * xpixels, numxbins + 1)
y_edges = np.linspace(0, ypixels, numybins + 1)

#########################################################################################
# helper date definitions, used in metrics code
alldaytimestamps = pd.date_range(start=startday, end=endday, freq='D')
day_numbers = alldaytimestamps.date
day_to_number = {day: idx for idx, day in enumerate(day_numbers)}
number_to_day = {idx: day for idx, day in enumerate(day_numbers)}
