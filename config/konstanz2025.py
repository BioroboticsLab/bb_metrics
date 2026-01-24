# THIS IS FOR 2025 - Konstanz
from pathlib import Path
import pandas as pd
import numpy as np
import os


year = 2025
startday = pd.Timestamp(year, 7, 21)  # TODO: UPDATE with actual start date
endday = pd.Timestamp(year, 10, 6)  # TODO: UPDATE with actual end date
weather_station_id = '10929'  # Konstanz weather station ID (verify this)

# SINGLE HIVE SETUP
hives = ('A',)

hive_cam_map = {
    "A": (0, 1),  # 2 cameras for hive A
}
# reverse the mapping → cam_hive_map
cam_hive_map = {0: "A", 1: "A"}



# Trajectory and detection processing paths
basedir = Path('/mnt/share/beesbook2025/')
trackdir = basedir / 'results/data_tracked'
alldetectiondir = basedir / 'results/data_alldetections'
traj_outdir = Path(os.path.join(basedir,'trajectory_data/'))
metrics_dir = basedir / 'metrics'
# Feeder/exit cam data (raw detections and daily aggregates)
feedercam_input_dir = Path(os.path.join(basedir,'pi/'))
feedercam_daily_dir = Path(os.path.join(basedir,'pi_data_alldetections/'))
feedercam_avg_dir = feedercam_daily_dir / 'avgcounts'

# Local outputs
saved_output_dir = Path(os.path.join(basedir,'saved_output'))

# Calibration assets (update per environment)
# TODO: UPDATE these paths to match the konstanz machine data location
comb_images_root = Path(os.path.join(basedir,"comb_background"))
calib_xml_dir = Path(os.path.join(comb_images_root, "corner_and_frame_annotations"))
# Comb annotation paths (derived from comb_images_root)
annotations_dir = comb_images_root / "annotations"
annotation_grids_dir = comb_images_root / "annotation_grids"
# Label configuration (shared across all experiments)
label_config_path = Path(__file__).parent / "label_classes.json"


## '4-frame representation' and pixel distances
# this is verified in "0 - Get corner points..".  Add this to the y-value of the corner point to have the separation between frames
offset_div_cm = 23.2
frame_width_cm = 37.6

## pixel and image sizes
# note:  using 'image rotation', which is always applied to the trajectories
ypixels, xpixels = (5312, 4608)  # Rotated (analysis coordinate frame) dimensions

# Camera rotation configuration
# Options: 'none', 'cw90' (clockwise 90°), 'ccw90' (counter-clockwise 90°), '180'
# For 2025, cameras are mounted requiring 90° clockwise rotation
camera_rotation = 'cw90'
KNOWN_LENGTH_CM = 35.0 # for settings px/cm conversion
SCALE_FACTOR = 2  # if using down-scaled videos (default in script, downscale by factor of 2

# Spatial histogram bins
pixels_per_bin = 225  # this is approximately 2 cm
pixels_per_cm_approx = 111.761 # average of df_px_per_cm
numxbins = np.round(2*xpixels/pixels_per_bin).astype(int)
numybins = np.round(ypixels/pixels_per_bin).astype(int)
numxbins = numxbins + (numxbins % 2)  # ensure that xbins is even, so it divides the middle
x_edges = np.linspace(0,2*xpixels,numxbins+1)
y_edges = np.linspace(0,ypixels,numybins+1)

#########################################################################################
# helper date definitions, used in metrics code
alldaytimestamps = pd.date_range(start=startday,end=endday,freq='D')
# Create day to number and number to day mappings
day_numbers = alldaytimestamps.date
day_to_number = {day: idx for idx, day in enumerate(day_numbers)}
number_to_day = {idx: day for idx, day in enumerate(day_numbers)}
