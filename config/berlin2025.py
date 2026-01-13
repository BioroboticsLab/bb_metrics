# THIS IS FOR 2025 - Berlin
from pathlib import Path
import pandas as pd
import numpy as np
import os


year = 2025
startday = pd.Timestamp(year,6,19)
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
comb_images_root = Path("/mnt/trove/beesbook2025/comb_background/")
calib_xml_dir = Path(os.path.join(comb_images_root, "corner_and_frame_annotations"))

# Trajectory processing paths
basedir = Path('/mnt/trove/beesbook2025/')
trackdir = basedir / 'data_tracked'
traj_outdir = Path('/mnt/trove/beesbook_trajectory_data/berlin2025/')
metrics_dir = basedir / 'metrics'
# Feeder/exit cam data (raw detections and daily aggregates)
feedercam_input_dir = Path('/mnt/trove/beesbook2025/pi/')
feedercam_daily_dir = Path('/mnt/trove/beesbook2025/pi_data_alldetections/')
feedercam_avg_dir = feedercam_daily_dir / 'avgcounts'
# Local outputs
saved_output_dir = Path('saved_output')

# Pestcide treatment days.  These are noted in "Bee Experiments 2025 - Tasks and Protocol - Hive and Feeder Interactions.csv"
# The start/end times here are edited for data use and to fill in missing times, using other time notes where applicable
# manually extracted treatment intervals for Feeder B and Feeder D
treatment_intervals = {
    "Feeder B": [
        # 06-26: TRUE but no time → 12:00 
        ("2025-06-26 12:00", "2025-06-28 16:17"),
        # 07-03: TRUE at 15:00.  07-04 has no time - so use the same one
        ("2025-07-03 15:00", "2025-07-04 15:00"),
        # 07-11: explicit “11:50”, then bottle remaineed the next day
        ("2025-07-11 11:50", "2025-07-13 14:57"),
        # 07-18: “bis 10:00” → start at 13:44 that day, end two days later at 10:00
        ("2025-07-18 13:44", "2025-07-20 10:00"),
        ("2025-07-24 12:35", "2025-07-26 15:15"),
        ("2025-07-31 17:25", "2025-08-02 15:03"),
        ("2025-08-07 12:00", "2025-08-09 16:10"),
        ("2025-08-14 11:00", "2025-08-16 13:51") ## IN-HIVE TREATMENT
    ],
    "Feeder D": [
        # 06-26: TRUE but no time → 12:00
        ("2025-06-26 12:00", "2025-06-28 15:50"),
        # 07-03: TRUE at 15:00.  07-04 has no time - so use the same one        
        ("2025-07-03 15:00", "2025-07-04 15:00"),
        ("2025-07-11 11:40", "2025-07-13 14:00"),
        ("2025-07-18 13:38", "2025-07-20 10:00"),
        ("2025-07-24 12:15", "2025-07-26 14:55"),
        ("2025-07-31 17:07", "2025-08-02 14:39"),        
        ("2025-08-07 12:00", "2025-08-09 16:08"),
        ("2025-08-14 11:00", "2025-08-16 13:51")  ## IN-HIVE TREATMENT
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
offset_div_cm = 23.2
frame_width_cm = 37.6

## pixel and image sizes
# note:  using 'image rotation', which is always applied to the trajectories in post-processing - see rotation settings
ypixels, xpixels = (5312, 4608)  # Original (pre-rotation) dimensions

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
