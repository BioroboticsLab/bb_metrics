# THIS IS FOR 2026 - Berlin
from pathlib import Path
import pandas as pd
import numpy as np
import os

year = 2026
startday = pd.Timestamp(year,6,1)
endday = pd.Timestamp(year,8,19)
weather_station_id = '10381'  # weather station ID, for use in meteostat:  Dahlem

hives = ('A','B','C','D')

hive_cam_map = {
    "A": (0,1),
    "B": (2,3),
    "C": (4,5),
    "D": (6,7),
}
# reverse the mapping → cam_hive_map
cam_hive_map = {
    cam: hive
    for hive, cams in hive_cam_map.items()
    for cam in cams
}

# Calibration assets (update per environment)
comb_images_root = Path("/mnt/trove/beesbook2026/comb_background/")
calib_xml_dir = Path(os.path.join(comb_images_root, "corner_and_frame_annotations"))
# Comb annotation paths (derived from comb_images_root)
annotations_dir = comb_images_root / "annotations"
annotation_grids_dir = comb_images_root / "annotation_grids"
# Label configuration (shared across all experiments)
label_config_path = Path(__file__).parent / "label_classes.json"

# Trajectory processing paths
basedir = Path('/mnt/trove/beesbook2026/')
trackdir = basedir / 'data_tracked'
alldetectiondir = basedir / 'data_alldetections'
traj_outdir = Path('/mnt/trove/beesbook_trajectory_data/berlin2026/')
metrics_dir = basedir / 'metrics'
# Feeder/exit cam data (raw detections and daily aggregates)
feedercam_input_dir = Path('/mnt/trove/beesbook2026/pi/')
feedercam_daily_dir = Path('/mnt/trove/beesbook2026/pi_data_alldetections/')
feedercam_avg_dir = feedercam_daily_dir / 'avgcounts'
# Local outputs
saved_output_dir = Path(os.path.join(basedir,'saved_output'))

# Pestcide treatment days.  These are noted in "Bee Experiments 2026 - Tasks and Protocol - Hive and Feeder Interactions.csv"
# The start/end times here are edited for data use and to fill in missing times, using other time notes where applicable
# manually extracted treatment intervals for Feeder B and Feeder D
treatment_intervals = {
    "Feeder A": [
    ],
    "Feeder D": [
    ]
}
rows = []
for feeder, intervals in treatment_intervals.items():
    for start_str, end_str in intervals:
        rows.append({
            'feeder': feeder,
            'start': pd.to_datetime(start_str),
            'end':   pd.to_datetime(end_str)
        })
treat_df = pd.DataFrame(rows)


## '4-frame representation' and pixel distances
# this is verified in "0 - Get corner points..".  Add this to the y-value of the corner point to have the separation between frames
## JUNE 2026:  THESE ARE NOT VERIFIED FOR 2026 YET
offset_div_cm = 23.2
frame_width_cm = 37.6

## pixel and image sizes
# note:  using 'image rotation', which is always applied to the trajectories in post-processing - see rotation settings
ypixels, xpixels = (5312, 4608)  # Rotated (analysis coordinate frame) dimensions 

# Camera rotation configuration
# Options: 'none', 'cw90' (clockwise 90°), 'ccw90' (counter-clockwise 90°), '180'
# For 2026, cameras are mounted requiring 90° clockwise rotation
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
