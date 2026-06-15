# THIS IS FOR 2019 - Berlin
#
# Ported from the old MAIN/2019/definitions_2019Berlin.py to the new bb_metrics
# config style. 2019 is a SINGLE-HIVE, 2-CAMERA, 2-FRAME observation hive whose
# trajectory parquets already exist (built from a Postgres DB with the old
# calibration). The new metrics pipeline reads those parquets directly, so this
# config only needs the pieces that path touches: dates, hive/cam maps, the
# spatial-histogram bins, and frame_width_cm (for the x_hive_flat fold).
#
# The 2024+ 4-frame negative-y geometry does NOT apply here, so the frame/exit
# metrics are skipped via `compute_frame_exit_metrics = False` (see
# bb_metrics/metricsfunctions.py get_metrics). offset_div_cm, camera_rotation and
# the comb/feeder/calibration paths are intentionally omitted — unused on this path.
from pathlib import Path
import pandas as pd
import numpy as np


year = 2019
startday = pd.Timestamp(year, 7, 19)
endday = pd.Timestamp(year, 10, 19)
weather_station_id = '10381'  # weather station ID, for use in meteostat: Dahlem

# SINGLE HIVE SETUP (2 cameras)
hives = ('A',)
hive_cam_map = {
    "A": (0, 1),  # 2 cameras for hive A
}
# reverse the mapping → cam_hive_map
cam_hive_map = {0: "A", 1: "A"}

# Trajectory inputs and metrics outputs (remote /mnt/trove locations).
# Existing 2019 trajectory parquets are reused as-is.
traj_outdir = Path('/mnt/trove/beesbook_trajectory_data/berlin2019/')
# Write to a NEW dir during validation so the old metrics2019/ outputs are
# preserved for the parity check; switch to 'metrics2019/' for the production run.
metrics_dir = Path('/mnt/trove/beesbook2024/metrics2019_v2/')

# Skip the 4-frame frame/exit-distance geometry metrics for 2019 (2-frame hive).
# Read in get_metrics via getattr(cfg, 'compute_frame_exit_metrics', True), so
# other years (which do not define it) keep the default 4-frame behavior.
compute_frame_exit_metrics = False

## '2-frame representation' - only front and back (no top/bottom middle division)
ypixels, xpixels = (3000, 4000)

# frame_width_cm: used by datafile_to_metrics to fold cam1 into x_hive_flat
# (x_hive_flat = frame_width_cm - x_hive for cam1), which feeds `dispersion`.
# The old 2019 notebook folded across the per-camera x_hive max instead; this
# fixed width approximates that. ~ xpixels / pixels_per_cm_approx = 4000/95 ≈ 42.1.
# TODO: validate against the actual per-camera x_hive max on the remote and adjust
# if `dispersion` does not match the old daydatamat.csv.
pixels_per_cm_approx = 95  # from df_markers conversions; variation across cams is small
frame_width_cm = round(xpixels / pixels_per_cm_approx, 1)  # ≈ 42.1 cm

# Spatial histogram bins (used by getxyhist → fraction_squares_visited)
pixels_per_bin = 190  # approximately 2 cm
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
