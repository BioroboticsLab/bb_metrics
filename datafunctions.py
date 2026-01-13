import numpy as np
import pandas as pd
import numbers
import re
from datetime import datetime, timezone
from meteostat import Hourly, Daily
import json, os
from pathlib import Path
import dill
import uuid
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from . import get_config

# Config holder; prefer bb_metrics.set_config(cfg). init remains for compatibility.
bd = None
def init(bd_input=None):
    """Set active config; prefer bb_metrics.set_config(cfg) instead of calling init."""
    global bd
    bd = bd_input or get_config()

def _get_bd():
    global bd
    if bd is None:
        try:
            bd = get_config()
        except Exception as e:
            raise RuntimeError("Config not set; call bb_metrics.set_config(cfg) or dfunc.init(cfg)") from e
    return bd

################### Misc useful functions
# takes timestampts, and converts them to integers
def assign_integer_framenums(times):
    sectimes = np.array([pd.Timestamp(t).hour*3600 + pd.Timestamp(t).minute*60 + pd.Timestamp(t).second + pd.Timestamp(t).microsecond/10**6 for t in times])
    return np.floor(sectimes*3).astype(int)

def assign_integer_framenums_hourminsec(hour,minute,second):
    # second can be a float
    return np.floor( (hour*3600 + minute*60 + second)*3 ).astype(int)

def flat_to_hist(flatrow):
    # assume that the hist is at the end of the row
    bd_cfg = _get_bd()
    numhistbins = bd_cfg.numxbins*bd_cfg.numybins
    return np.reshape(np.array(flatrow)[-numhistbins:],(bd_cfg.numxbins,bd_cfg.numybins))

def get_weather_data(station_id, start_date, end_date, data_type='hourly'):
    """
    Fetches weather data from the given station ID within the provided date range.
    
    Args:
        station_id (str): The ID of the weather station.
        start_date (datetime)
        end_date (datetime)
    
    Returns:
        pandas.DataFrame: A dataframe containing the weather data.
    """
    # Fetch the weather data between the start_date and end_date
    if data_type == 'hourly':
        weatherdata = Hourly(station_id, start_date, end_date)
        weatherdata = weatherdata.fetch()
    elif data_type == 'daily':
        weatherdata = Daily(station_id, start_date, end_date)
        weatherdata = weatherdata.fetch()
    else:
        print('data_type can be hourly or daily')
        weatherdata = np.nan
    return weatherdata


## the bb_monitory function has input 'numdays', and this should simply get all
# this gets temperature data as stored in csv files from bb_temperature monitor
def get_temperature_data(data_folder, startday=None):
    json_file_path = data_folder+'hexcodes_locations.json'
    with open(json_file_path, 'r') as f:
        hex_codes_dict = json.load(f)

    # Expected headers derived from the hex codes in the JSON file
    expected_headers = ['Time'] + sorted(['Temp'+key for hive in hex_codes_dict.values() for key in hive.keys()])
    
    # Function to check if the first line is a header
    def is_header(file_path):
        with open(file_path, 'r') as file:
            first_line = file.readline().strip().split(',')
            return first_line[0] == 'Time'
    
    # List all CSV files in the folder that match the pattern "temperature_data_YYYY-MM-DD.csv"
    csv_files = []
    for file_name in os.listdir(data_folder):
        if file_name.startswith('temperature_data_') and file_name.endswith('.csv'):
            # Extract the date part from the filename
            file_date_str = file_name[len('temperature_data_'):-len('.csv')]
            file_date = datetime.strptime(file_date_str, '%Y-%m-%d').date()
    
            # Skip the file if it's before the startday
            if (startday is not None) and (file_date < startday):
                continue
    
            file_path = os.path.join(data_folder, file_name)
            csv_files.append(file_path)
    
    # Read in all data files and combine them into a single DataFrame
    data_frames = []
    for file_path in csv_files:
        if is_header(file_path):
            df = pd.read_csv(file_path, parse_dates=['Time'])
            if not(np.all(df.columns)==expected_headers):
                print('ERROR:  header columns do not match:',file_path)
        else:
            df = pd.read_csv(file_path, header=None, parse_dates=[0])
            df.columns = expected_headers
        data_frames.append(df)
    
    # Combine all data frames into one
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    # Filter out rows where 'Time' column contains the string 'Time'
    combined_data = combined_data[~combined_data['Time'].astype(str).str.contains('Time')]
    
    # Ensure 'Time' column is datetime
    combined_data['Time'] = pd.to_datetime(combined_data['Time'])
    combined_data = combined_data.sort_values(by='Time')
    
    
    # 1) Remove values with temperature changes greater than max_temp_diff
    max_temp_diff = 5
    for col in combined_data.columns[1:]:  # Skip 'Time' column
        combined_data[col] = combined_data[col].astype(float)
        combined_data = combined_data[(combined_data[col].diff().abs() <= max_temp_diff) | (combined_data[col].diff().isnull())]
    
    # 2) Apply a moving average filter with a default window of 5 minutes
    combined_data.set_index('Time', inplace=True)
    avgwindow_minutes = 30
    combined_data = combined_data.rolling(str(avgwindow_minutes)+'min').mean().reset_index()

    ## label dictionary for legend labels
    label_dict = {'Temp' + key: f"{hive[-1]}: {hex_codes_dict[hive][key]}" for hive in hex_codes_dict for key in hex_codes_dict[hive]}
    label_dict = dict(sorted(label_dict.items(), key=lambda item: ('brood' not in item[1], 'honey' not in item[1], 'room' not in item[1], item[1])))
    # sort for plotting
    label_dict = dict(sorted(label_dict.items(), key=lambda item: ('brood' not in item[1], 'honey' not in item[1], 'room' not in item[1], item[1])))    

    return combined_data, hex_codes_dict, label_dict

def parse_data_file_timestamps(filename):
    """
    Parses the start and end timestamps from data filenames

    Args:
        filename (str): The filename from which to extract the timestamps.

    Returns:
        tuple: (start_timestamp, end_timestamp) as timezone-aware datetime objects in UTC.
    """
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}Z)--(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}Z)'
    match = re.search(timestamp_pattern, filename)
    if match:
        start_str = match.group(1).replace('_', ':')
        end_str = match.group(2).replace('_', ':')
        start_timestamp = datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        end_timestamp = datetime.strptime(end_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        return start_timestamp, end_timestamp
    else:
        raise ValueError(f"Timestamps not found in filename: {filename}")

def parse_hive_name(filename):
    """
    Parses the hive name from the given filename.

    Args:
        filename (str): The filename from which to extract the hive name.

    Returns:
        str: The hive name extracted from the filename.
    """
    # Use regular expression to find 'Hive_' followed by the hive identifier
    match = re.search(r'Hive_([A-Za-z0-9]+)', filename)
    if match:
        hive_name = match.group(1)
        return hive_name
    else:
        raise ValueError(f"Hive name not found in filename: {filename}")

# Get unique timestamps sorted in ascending order
def get_timestamp_int(df):
    # Create a range of timestamps starting from the first unique timestamp with the given frequency
    num_periods = int(  (df['timestamp_start'].max() - df['timestamp_start'].min()) / pd.Timedelta(hours=6) ) 
    timestamp_start_num = {timestamp: i for i, timestamp in enumerate(pd.date_range(start=df['timestamp_start'].min(), 
                                                                                    periods=num_periods+1, 
                                                                                    freq='6h'))}
    # Map the timestamp to the corresponding integer
    return df['timestamp_start'].map(timestamp_start_num)

def filter_df_by_numdetections(df,min_time_detection_minutes=1):
    min_num_detections = min_time_detection_minutes*60*6
    filtered_df = df[df['num_detections'] >= min_num_detections].copy()
    return filtered_df

def get_birthdeath_df_by_detections(df):
    # Group by bee_id and get the first and last timestamp_num for each bee
    df_birth_death = df.groupby('bee_id').agg(
        birthdate=('timestamp_start', 'min'),
        deathdate=('timestamp_start', 'max')
    ).reset_index()
    # Convert birthdate and deathdate to just the date (ignore hours)
    df_birth_death['birthdate'] = df_birth_death['birthdate'].dt.date
    df_birth_death['deathdate'] = df_birth_death['deathdate'].dt.date    
    return df_birth_death

def calculate_average_temperature(df, minutes=5, startday=None):
    # Convert 'Time' column to datetime if not already
    df['Time'] = pd.to_datetime(df['Time'])
    # If startday is provided, filter the data starting from this day
    if startday is not None:
        df = df[df['Time'].dt.date >= startday]
    # Set 'Time' as the index
    df.set_index('Time', inplace=True)
    # Resample the data to N-minute intervals and calculate the mean for each column
    df_resampled = df.resample(f'{minutes}min').mean()
    # Reset the index to make 'Time' a column again
    df_resampled.reset_index(inplace=True)
    return df_resampled

###########################################################################################################################
## Markers and converting to cm
# multiple calculations that are useful for data reduction
###########################################################################################################################

def get_corner_point_for_date(cam: int, ts, merged: pd.DataFrame):
    """
    Return (corner_x, corner_y, chosen_row) for the given camera and target timestamp.
    
    Selection rule:
      - Use merged['midday_utc'].
      - Pick the row with midday_utc <= ts that is *closest* in time.
      - If none are before (or equal), pick the absolute closest (which will be after).
    
    Args:
      cam: camera id/string (e.g., 'cam-0')
      ts:  pandas-timestamp-like (string/Datetime); will be converted to UTC
      merged: DataFrame with columns ['hive','cam','midday_utc','corner_x','corner_y',...]
      hive: optional hive filter (e.g., 'A')

    Returns:
      (corner_x, corner_y, row)  # row is the full pandas Series of the chosen entry
    """

    df = merged
    df = df[df['cam'] == cam].copy()
    df['midday_utc'] = pd.to_datetime(df["midday_utc"], utc=True)

    if df.empty:
        raise ValueError(f"No rows found for cam={cam}" + (f", hive={hive}" if hive else ""))

    # Prefer rows at/before target
    before = df[df['midday_utc'] <= ts]
    if not before.empty:
        idx = (ts - before['midday_utc']).idxmin()
        row = df.loc[idx]
    else:
        # Fallback: absolute closest (will be after)
        idx = (df['midday_utc'] - ts).abs().idxmin()
        row = df.loc[idx]

    return float(row['corner_x']), float(row['corner_y']), row

def pixels_to_cm(
    x: np.ndarray,
    y: np.ndarray,
    date,                 # array-like of timestamps or a single timestamp
    cam_id,               # scalar (e.g., 'cam-0') or array-like same length as x/y
    df_cornerpoints: pd.DataFrame,
    df_px_per_cm: pd.DataFrame,
):
    """
    Convert pixel coords -> cm in the hive frame using new calibration tables.

    Equivalent to the old API but uses:
      - df_cornerpoints with columns ['cam','midday_utc','corner_x','corner_y']
      - df_px_per_cm with columns ['cam','pixels_per_cm']

    Selection rule for corner points:
      - For each timestamp, use the row with `midday_utc <= ts` that is closest in time.
      - If none exist before, use the absolute closest row (the next one).

    Supports vector inputs (recommended). Returns (x_hive, y_hive) as numpy arrays.
    """
    # Normalize inputs to arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Broadcast/align date to an array
    if np.isscalar(date) or not hasattr(date, "__len__"):
        ts = pd.to_datetime([date] * len(x), utc=True)
    else:
        ts = pd.to_datetime(date, utc=True)

    # Broadcast/align cam_id to an array of strings
    if np.isscalar(cam_id) or (hasattr(cam_id, "__len__") and len(np.atleast_1d(cam_id)) == 1):
        cams = np.array([cam_id] * len(x), dtype=object)
    else:
        cams = np.asarray(cam_id)

    # Output arrays
    x_hive = np.full_like(x, np.nan, dtype=float)
    y_hive = np.full_like(y, np.nan, dtype=float)

    # Process per camera (vectorized per-cam)
    for cam in pd.unique(cams):
        idx = np.where(cams == cam)[0]
        if idx.size == 0:
            continue

        # Get cm_per_pixel for this cam
        row_ppc = df_px_per_cm[df_px_per_cm["cam"] == cam]
        if row_ppc.empty:
            # no calibration for this cam
            continue
        cm_per_pixel = 1.0 / float(row_ppc["pixels_per_cm"].iloc[0])

        # Corner table for this cam
        dfc = df_cornerpoints[df_cornerpoints["cam"] == cam].copy()
        if dfc.empty:
            # no corner points => cannot convert
            continue

        # Ensure proper types and sort
        dfc["midday_utc"] = pd.to_datetime(dfc["midday_utc"], utc=True)
        dfc = dfc.sort_values("midday_utc")

        # Build a “query” DF with timestamps for this cam
        q = pd.DataFrame({"ts": ts[idx]}).sort_values("ts")
        # Backward merge_asof: pick latest <= ts
        m1 = pd.merge_asof(
            q, dfc.rename(columns={"midday_utc": "ts"}), on="ts", direction="backward"
        )
        # For rows that didn’t match (NaNs), fall back to nearest
        need_nearest = m1["corner_x"].isna()
        if need_nearest.any():
            m2 = pd.merge_asof(
                q[need_nearest], dfc.rename(columns={"midday_utc": "ts"}), on="ts", direction="nearest"
            )
            m1.loc[need_nearest, ["corner_x", "corner_y"]] = m2[["corner_x", "corner_y"]].values

        # Re-order to original idx order
        m1 = m1.set_index(pd.Index(idx)).loc[idx]

        # Compute cm using the chosen bottom-left corner
        x_hive[idx] = (x[idx] - m1["corner_x"].to_numpy()) * cm_per_pixel
        y_hive[idx] = (y[idx] - m1["corner_y"].to_numpy()) * cm_per_pixel

    return x_hive, y_hive


###########################################################################################################################
## tagged bees: .dill output to tracked parquet dataframe
###########################################################################################################################
# this is used by bb_monitor_2025 to process dill files, and also in 1-Process Trajectories to save final version
# note!  I removed caching, because this makes it messier to keep up with the files
# for updating bb_monitor, should use the code in 'Process trajectories' and simply remove the filtering by tag intro dates if that info is not available
def get_tracked_dataframe(
    filename: str,
    df_cornerpoints: pd.DataFrame | None = None,
    df_px_per_cm: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Load a *.dill* tracking file and return as a DataFrame.

    Optional:
      If both `df_cornerpoints` and `df_px_per_cm` are provided, add:
        ['x_hive', 'y_hive']  — cm-space coordinates computed via `pixels_to_cm`.
    """
    filename = Path(filename)



    # Parse dill
    columns = [
        'timestamp', 'frame_id', 'track_id',
        'x_pixels', 'y_pixels', 'orientation_pixels',
        'detection_index', 'detection_type',
        'cam_id', 'bee_id', 'bee_id_confidence'
    ]

    tracks = []
    if filename.exists():
        with filename.open('rb') as fh:
            while True:
                try:
                    batch = dill.load(fh)
                    tracks.extend(batch)
                except EOFError:
                    break
                except dill.UnpicklingError as e:
                    print(f"⚠️  Warning: could not unpickle {filename.name}: {e}")
                    break
                except Exception as e:
                    print(f"⚠️  Warning: unexpected error reading {filename.name}: {e}")
                    break

    df = pd.DataFrame(tracks, columns=columns)

    if not df.empty:
        # Transform coordinates using rotation configuration
        # Detection pipeline outputs (x_rot, y_rot) in rotated image coords (origin top-left)
        # We need coordinates with origin at bottom-left to match calibration coordinate system
        from . import get_config
        from .rotation import get_rotation_config

        cfg = get_config()
        rot_cfg = get_rotation_config(cfg)

        # Transform pixel coordinates
        x_calib, y_calib = rot_cfg.transform_detections(
            df['x_pixels'].to_numpy(),
            df['y_pixels'].to_numpy()
        )
        df['x_pixels'] = x_calib
        df['y_pixels'] = y_calib

        # Transform orientation
        df['orientation_pixels'] = rot_cfg.transform_orientation(
            df['orientation_pixels'].to_numpy()
        )
        # numeric coercion
        for c in columns[1:]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        # rename this column to simply 'orientation'
        df = df.rename(columns={'orientation_pixels': 'orientation'})

        # timestamps to UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

        # Optional pixels->cm conversion (vectorized per-cam under the hood)
        if df_cornerpoints is not None and df_px_per_cm is not None:
            try:
                x_hive, y_hive = pixels_to_cm(
                    x=df['x_pixels'].to_numpy(),
                    y=df['y_pixels'].to_numpy(),
                    date=df['timestamp'].to_numpy(),
                    cam_id=df['cam_id'].to_numpy(),
                    df_cornerpoints=df_cornerpoints,
                    df_px_per_cm=df_px_per_cm,
                )
                df['x_hive'] = x_hive
                df['y_hive'] = y_hive
            except Exception as e:
                print(f"⚠️  pixel→cm conversion failed: {e}")

    return df

    
###########################################################################################################################
## CALCULATIONS
# multiple calculations that are useful for data reduction
###########################################################################################################################
    
def fixanglerange(angles):
    return np.arctan2(np.sin(angles),np.cos(angles))

## helper function:  get the min camera for the hive
def get_hive_cam0(camera):
    """
    Returns the 'cam0' (min cam id) for the hive that the given camera belongs to.
    Accepts:
      - int (e.g., 3)
      - float (e.g., 3.0)
      - numpy scalar
      - list/tuple/ndarray (takes first non-NA)
      - pandas Series/Index (takes first non-NA by position)
    """
    import numpy as np
    import pandas as pd

    def _first_scalar(x):
        # Series/Index -> first non-NA by position
        if isinstance(x, (pd.Series, pd.Index)):
            if x.size == 0:
                raise ValueError("camera input is empty (Series/Index)")
            # get first non-NA
            if x.notna().any():
                return x.iloc[int(np.flatnonzero(x.notna())[0])]
            else:
                raise ValueError("camera input has no non-NA values (Series/Index)")
        # list/tuple/ndarray -> first element
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                raise ValueError("camera input is empty (array-like)")
            return x[0]
        return x  # scalar-ish

    cam = _first_scalar(camera)

    # numpy scalar -> python scalar
    if isinstance(cam, np.generic):
        cam = cam.item()

    # float -> int
    if isinstance(cam, float):
        cam = int(cam)

    if not isinstance(cam, int):
        raise TypeError(f"camera must resolve to an int, got {type(cam)}")

    bd_cfg = _get_bd()
    # map cam -> hive -> cam0
    hive = bd_cfg.cam_hive_map[cam]           # e.g., 'A'
    cam0 = bd_cfg.hive_cam_map[hive][0]       # e.g., 0 for ('A': (0,1))
    return cam0

# returns counts of which 'frame' of the observation hive a bees was on
## NEXT:  CONVERT THIS TO USING Y_HIVE INSTEAD - AND THEN SET THE DIVS TO (REASONABLE) ADDED VALUES THAT DEFINITELY GET THE RANGE, E.G. +10CM
def getframehist(y,camera):
    # input: y as hive coordinates in cm
    # note:  in 2024 and other analysis this used pixel coordinates, but now using hive coordinates because the middle div is defined in cm
    bd_cfg = _get_bd()
    cam0 = get_hive_cam0(camera)
    bins = [-10, bd_cfg.offset_div_cm, bd_cfg.offset_div_cm*2+10 ] # set limits to cover the frame.  bins are the same for each camera now
    vals_l = np.histogram(y[camera==cam0],bins=bins)[0] 
    vals_r = np.histogram(y[camera==cam0+1],bins=bins)[0]        
    return np.array([vals_l,vals_r]) 

# return x-y histogram, using the bins and edges that are set in definitions
def getxyhist(x,y,camera):
    bd_cfg = _get_bd()
    x_adjusted = x + (camera-get_hive_cam0(camera))*bd_cfg.xpixels  # camera 0 left, camera 1 on the right
    hist = np.histogram2d(x_adjusted,y,bins=[bd_cfg.x_edges,bd_cfg.y_edges])[0]
    return hist



####################################################################################################
# Functions
###### SUBSTRATE AND INSIDE/OUTSIDE FUNCTIONS ##################################
############################################################    

def get_inout_estimates(dfday, obs_threshold=5, exitdistthreshold=1000,numtimedivs=288):  # dfday = dataframe containing data for one day
    day_uids = np.unique(dfday['Bee unique ID']).astype(int) 
    bee_obs = np.tile(np.nan,(len(day_uids),numtimedivs))
    bee_exitdist = np.tile(np.nan,(len(day_uids),numtimedivs))
    dfids = np.array(dfday['Bee unique ID']).astype(int)
    day_ages = np.tile(-1,len(day_uids))

    for j,beeid in enumerate(day_uids):
        sel = (dfids==beeid)

        dfsel = dfday[sel].copy()
        day_ages[j] = dfsel['Age'].astype(int).values[0]
        td = dfsel['timedivision'].astype(int)
        bee_obs[j,td] = dfsel['Num. observations']
        bee_exitdist[j,td] = dfsel['Exit distance (median)']
        bee_exitdist[j,np.isnan(bee_obs[j])] = np.nan

    all_inhive = np.tile(np.nan,(len(day_uids),numtimedivs))

    for beenum in range(len(day_uids)):
        obs = bee_obs[beenum]>=obs_threshold
        closetoexit = bee_exitdist[beenum]<exitdistthreshold

        bins = np.append(np.insert(np.where(np.abs(np.diff(obs).astype(int)))[0]+1,0,0),numtimedivs)
        sections = np.array([bins[0:-1],bins[1:]]).T
        # each section has the same values, all True or all False.
        # set to 'in hive', where all are above the threshold
        if len(sections)==1: # special case of all are the same
            if obs[0]:
                all_inhive[beenum,:] = 1
            # if not, dont say anything, because don't know, this bee could be dead.
        else:
            for j,s in enumerate(sections):
                if obs[s[0]]:
                    all_inhive[beenum,s[0]:s[1]] = 1
                else:
                    if j==0: # treat the first section different
                        if closetoexit[np.min((s[1]+1,numtimedivs-1))]:
                            all_inhive[beenum,s[0]:s[1]] = 0
                        else:
                            all_inhive[beenum,s[0]:s[1]] = 1
                    else:
                        if closetoexit[s[0]-1]:    
                            all_inhive[beenum,s[0]:s[1]] = 0
                        else:

                            all_inhive[beenum,s[0]:s[1]] = 1

    return  day_uids, day_ages, all_inhive, bee_obs, bee_exitdist



def get_onsubstrate(dfday, obs_threshold=5, topfraction_threshold=0.5, substratename='Festoon',numtimedivs=288 ):  # dfday = dataframe containing data for one day
    day_uids = np.unique(dfday['Bee unique ID']).astype(int) 
    bee_obs = np.tile(np.nan,(len(day_uids),numtimedivs))
    bee_topframe = np.tile(np.nan,(len(day_uids),numtimedivs))
    bee_data = np.tile(np.nan,(len(day_uids),numtimedivs,dfday.shape[-1]))

    dfids = np.array(dfday['Bee unique ID']).astype(int)

    day_ages = np.tile(-1,len(day_uids))
    
    for j,beeid in enumerate(day_uids):
        sel = (dfids==beeid)

        dfsel = dfday[sel].copy()
        day_ages[j] = dfsel['Age'].astype(int).values[0]
        td = dfsel['timedivision'].astype(int)
        bee_obs[j,td] = dfsel['Num. observations']
        if substratename=='topframe':
            bee_topframe[j,td] = dfsel['Frame 0'] + dfsel['Frame 3']
        else:    
            bee_topframe[j,td] = dfsel[substratename]
        
        bee_data[j,td] = dfsel
        bee_topframe[j,np.isnan(bee_obs[j])] = np.nan
        bee_data[j,np.isnan(bee_obs[j])] = np.nan
        
    all_ontop = np.tile(np.nan,(len(day_uids),numtimedivs))

    for beenum in range(len(day_uids)):
        obs = bee_obs[beenum]>=obs_threshold
        mostlyontop = bee_topframe[beenum]>topfraction_threshold

        bins = np.append(np.insert(np.where(np.abs(np.diff(obs).astype(int)))[0]+1,0,0),numtimedivs)
        sections = np.array([bins[0:-1],bins[1:]]).T
        # each section has the same values, all True or all False.
        # set to 'in hive', where all are above the threshold
        if len(sections)==1: # special case of all are the same
            if obs[0]&mostlyontop[0]:
                all_ontop[beenum,:] = True
            # if not, dont say anything, because don't know, this bee could be dead.
        else:
            for j,s in enumerate(sections):
                if obs[s[0]]:  # if observed in this section
                    all_ontop[beenum,s[0]:s[1]] = mostlyontop[s[0]]  # mark as ontop if above threshold
                else:  # if not observed
                    if j==0: # treat the first section different
                        all_ontop[beenum,s[0]:s[1]] = mostlyontop[s[1]]  # if the next segment has them on top, mark as on top
                    else:
                        all_ontop[beenum,s[0]:s[1]] = mostlyontop[s[0]-1] # if prev segment has on top, mark as on top

    return  day_uids, day_ages, all_ontop, bee_obs, bee_data

###########################################################################################################################
# Annotation + grid utilities (comb background)
###########################################################################################################################

def load_annotation_json(json_path: str | Path):
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        cells = json.load(f)

    rows = []
    for cell in cells:
        rows.append(
            {
                "id": cell.get("id", str(uuid.uuid4())),
                "center_x": cell["center_x"],
                "center_y": cell["center_y"],
                "radius": cell["radius"],
                "label": cell["label"],
            }
        )

    df = pd.DataFrame(rows)
    points = df[["center_y", "center_x"]].to_numpy()
    diameters = (df["radius"] * 2).to_numpy(dtype=float)
    labels = df["label"].to_list()
    return df, points, diameters, labels


def load_label_config(label_config_path: Path):
    with open(label_config_path, "r") as f:
        label_config = json.load(f)
    label_color_hex = {entry["name"]: entry["color"] for entry in label_config}
    label_order = [entry["name"] for entry in label_config]
    keep_labels = set(label_order[:4])
    other_label = label_order[4] if len(label_order) >= 5 else "other_cell"
    return label_color_hex, label_order, keep_labels, other_label


def normalize_label(label: str, keep_labels: set, other_label: str) -> str:
    return label if label in keep_labels else other_label


def hex_to_rgba(hex_color: str):
    h = hex_color.lstrip("#")
    if len(h) == 8:  # RRGGBBAA
        r, g, b, a = h[0:2], h[2:4], h[4:6], h[6:8]
        return tuple(int(v, 16) / 255 for v in (r, g, b, a))
    if len(h) == 6:  # RRGGBB
        r, g, b = h[0:2], h[2:4], h[4:6]
        return tuple(int(v, 16) / 255 for v in (r, g, b)) + (1.0,)
    return (1.0, 0.0, 1.0, 1.0)  # fallback magenta


def with_alpha(rgba, a):
    return (rgba[0], rgba[1], rgba[2], a)


def plot_annotated_row(
    row,
    label_color_hex,
    label_order,
    keep_labels,
    other_label,
    alpha_fill=0.25,
    alpha_edge=0.1,
    fill_empty_cell=False,
    empty_label="empty_cell",
    rotate_clockwise=True,
    cmap="gray",
    figsize=(10, 7),
    ax=None,
    show=True,
):
    img_path = Path(row["path"])
    ann_path = Path(row["annotation_path"])

    with open(ann_path, "r") as f:
        cells = json.load(f)

    # Get rotation config if available
    try:
        from .rotation import get_rotation_config
        cfg = get_config()
        rot_cfg = get_rotation_config(cfg)
        use_rotation_config = True
    except Exception:
        rot_cfg = None
        use_rotation_config = False

    img = plt.imread(img_path)
    h, _ = img.shape[:2]

    # Apply rotation for display
    if use_rotation_config and rot_cfg is not None:
        k = rot_cfg.numpy_rot90_k()
        if k != 0:
            img = np.rot90(img, k=k)
    elif rotate_clockwise:
        img = np.rot90(img, k=-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(img, cmap=cmap)

    for cell in cells:
        label = normalize_label(cell["label"], keep_labels, other_label)
        base_rgba = hex_to_rgba(label_color_hex.get(label, "#ff00ff"))

        if label == empty_label and not fill_empty_cell:
            face_rgba = with_alpha(base_rgba, 0.0)
        else:
            face_rgba = with_alpha(base_rgba, alpha_fill)

        x = cell["center_x"]
        y = cell["center_y"]

        # Transform annotation coordinates for display on rotated image
        if use_rotation_config and rot_cfg is not None:
            x_plot, y_plot = rot_cfg.transform_annotation_coords(x, y)
        elif rotate_clockwise:
            x_plot = h - 1 - y
            y_plot = x
        else:
            x_plot = x
            y_plot = y

        circle = Circle(
            (x_plot, y_plot),
            cell["radius"],
            edgecolor=with_alpha(base_rgba, alpha_edge),
            facecolor=face_rgba,
            linewidth=1.0,
        )
        ax.add_patch(circle)

    mapped_labels = {normalize_label(c["label"], keep_labels, other_label) for c in cells}
    legend_labels = [lbl for lbl in label_order[:5] if lbl in mapped_labels]
    for lbl in sorted(mapped_labels):
        if lbl not in legend_labels:
            legend_labels.append(lbl)

    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=hex_to_rgba(label_color_hex.get(lbl, "#ff00ff")),
            markeredgecolor="k",
            label=lbl,
            markersize=8,
        )
        for lbl in legend_labels
    ]
    ax.legend(handles=legend_elems, loc="upper right", frameon=True)

    day_str = pd.to_datetime(row["day"]).strftime("%Y-%m-%d")
    ax.set_title(f"cam {row['cam']} - {day_str}")
    ax.axis("off")

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


def parse_background_image_fname(fname: str | Path, prefix: str = "background_"):
    """Parse comb image filenames by stripping a prefix and using bb_binary.parse_image_fname."""
    basename = Path(fname).name
    if prefix and prefix in basename:
        basename = basename.split(prefix, 1)[1]

    suffix = Path(basename).suffix.lower()
    if suffix and suffix not in (".png", ".jpg", ".jpeg"):
        basename = Path(basename).stem

    from bb_binary.parsing import parse_image_fname

    return parse_image_fname(basename)


def build_background_files_df(
    comb_background_dir: Path, cam_hive_map: dict, prefix: str = "background_"
):
    comb_background_dir = Path(comb_background_dir)
    file_glob = f"{prefix}*" if prefix else "*"

    rows = []
    for cam_dir in sorted(comb_background_dir.glob("cam-*")):
        if not cam_dir.is_dir():
            continue
        for path in sorted(cam_dir.glob(file_glob)):
            try:
                cam, timestamp = parse_background_image_fname(path.name, prefix=prefix)
            except Exception:
                continue

            rows.append(
                {
                    "path": str(path),
                    "cam": cam,
                    "hive": cam_hive_map.get(cam),
                    "timestamp": timestamp,
                }
            )

    if not rows:
        background_files_df = pd.DataFrame(
            columns=["path", "cam", "hive", "timestamp", "day", "image_key"]
        )
        return background_files_df

    background_files_df = (
        pd.DataFrame(rows).sort_values(["cam", "timestamp"]).reset_index(drop=True)
    )
    background_files_df["day"] = background_files_df["timestamp"].dt.round("D")
    background_files_df["image_key"] = background_files_df["path"].map(lambda p: Path(p).stem)
    return background_files_df


def build_annotation_files_df(
    annotations_dir: Path, cam_hive_map: dict, prefix: str = "background_"
):
    annotations_dir = Path(annotations_dir)

    ann_rows = []
    for path in sorted(annotations_dir.glob("*.json")):
        try:
            cam, timestamp = parse_background_image_fname(path.name, prefix=prefix)
        except Exception:
            continue

        ann_id = None
        if prefix and prefix in path.name:
            ann_prefix = path.name.split(prefix, 1)[0].rstrip("_")
            if ann_prefix.isdigit():
                ann_id = int(ann_prefix)

        stem = Path(path).stem
        if prefix and prefix in stem:
            image_key = prefix + stem.split(prefix, 1)[1]
        elif "_" in stem:
            image_key = stem.split("_", 1)[1]
        else:
            image_key = stem

        ann_rows.append(
            {
                "annotation_id": ann_id,
                "path": str(path),
                "cam": cam,
                "hive": cam_hive_map.get(cam),
                "timestamp": timestamp,
                "image_key": image_key,
            }
        )

    if not ann_rows:
        annotation_files_df = pd.DataFrame(
            columns=["annotation_id", "path", "cam", "hive", "timestamp", "image_key", "day"]
        )
        return annotation_files_df

    annotation_files_df = (
        pd.DataFrame(ann_rows).sort_values(["cam", "timestamp"]).reset_index(drop=True)
    )
    annotation_files_df["day"] = annotation_files_df["timestamp"].dt.round("D")
    return annotation_files_df


def build_combined_annotation_df(
    comb_background_dir: Path, cam_hive_map: dict, prefix: str = "background_"
):
    comb_background_dir = Path(comb_background_dir)
    background_files_df = build_background_files_df(
        comb_background_dir, cam_hive_map, prefix=prefix
    )
    annotation_files_df = build_annotation_files_df(
        comb_background_dir / "annotations", cam_hive_map, prefix=prefix
    )

    combined_df = background_files_df.merge(
        annotation_files_df[["image_key", "path"]].rename(
            columns={"path": "annotation_path"}
        ),
        on="image_key",
        how="left",
    )
    combined_df["is_annotated"] = combined_df["annotation_path"].notna()
    return background_files_df, annotation_files_df, combined_df


def get_valid_timestamp_starts(hive, df_datafiles: pd.DataFrame):
    """Return timestamp_start values that have both cams present for a hive."""
    timestamp_starts, counts = np.unique(
        df_datafiles.loc[df_datafiles["hive"] == hive, "timestamp_start"],
        return_counts=True,
    )
    valid_mask = counts == 2
    return timestamp_starts[valid_mask]


def process_timestamp_chunk(
    timestamp_start,
    timestamp_end,
    hive,
    df_datafiles: pd.DataFrame,
    savedir,
    time_division,
    update: bool = True,
):
    """
    Reads Parquet data for the given chunk, splits it into sub-intervals,
    computes histograms, and writes the result to disk in `savedir`.
    """
    import gzip
    import pickle
    import time
    import gc

    savedir = Path(savedir)
    outpath = savedir / f"hive_{hive}_ts_{timestamp_start}_dt_{time_division}.pklz"
    outpath = Path(str(outpath).replace(":", "_"))

    datafiles = df_datafiles.loc[
        (df_datafiles["timestamp_start"] == timestamp_start)
        & (df_datafiles["hive"] == hive),
        "datafile",
    ].tolist()

    if outpath.exists():
        if not update:
            return None
        if datafiles:
            out_ts = outpath.stat().st_mtime
            newest_src_ts = max(Path(f).stat().st_mtime for f in datafiles)
            if newest_src_ts <= out_ts:
                return None
        else:
            return None

    starttime = time.time()

    if not datafiles:
        print(f"[SKIP] No datafiles for {timestamp_start}, hive {hive}")
        return None

    df = pd.concat(pd.read_parquet(f) for f in datafiles)

    # Transform coordinates using rotation configuration
    # Detection pipeline outputs (x_rot, y_rot) in rotated image coords (origin top-left)
    # We need coordinates with origin at bottom-left to match calibration coordinate system
    if not df.empty:
        from . import get_config
        from .rotation import get_rotation_config

        cfg = get_config()
        rot_cfg = get_rotation_config(cfg)

        # Transform pixel coordinates
        x_calib, y_calib = rot_cfg.transform_detections(
            df['x_pixels'].to_numpy(),
            df['y_pixels'].to_numpy()
        )
        df['x_pixels'] = x_calib
        df['y_pixels'] = y_calib

    chunk_dict = {}
    time_segments = pd.date_range(
        start=timestamp_start, end=timestamp_end, freq=time_division
    )

    for i in range(len(time_segments) - 1):
        seg_start = time_segments[i]
        seg_end = time_segments[i + 1]

        df_segment = df[
            (df["timestamp"] >= seg_start) & (df["timestamp"] < seg_end)
        ]
        num_timestamps = len(df_segment["timestamp"].unique())

        if num_timestamps > 0:
            num_detections = len(df_segment) / num_timestamps
            xyhist = (
                getxyhist(
                    df_segment["x_pixels"],
                    df_segment["y_pixels"],
                    df_segment["cam_id"],
                )
                / num_timestamps
            )
            framehist = (
                getframehist(df_segment["y_pixels"], df_segment["cam_id"])
                / num_timestamps
            )
        else:
            num_detections = 0
            xyhist = framehist = np.nan

        if seg_start not in chunk_dict:
            chunk_dict[seg_start] = {}

        chunk_dict[seg_start][hive] = {
            "time_segment_end": seg_end,
            "num_detections": num_detections,
            "framehist": framehist,
            "xyhist": xyhist,
        }

    del df
    gc.collect()

    savedir.mkdir(parents=True, exist_ok=True)
    with gzip.open(outpath, "wb") as f:
        pickle.dump(chunk_dict, f)

    elapsed = round(time.time() - starttime, 2)
    print(f"[DONE] {timestamp_start}: processed in {elapsed} sec => {outpath}")
    return chunk_dict


def build_label_index(label_order, keep_labels, other_label, off_comb_label=None):
    if other_label not in label_order:
        label_order = label_order + [other_label]
    if off_comb_label is not None and off_comb_label not in label_order:
        label_order = label_order + [off_comb_label]
    label_to_idx = {lbl: i for i, lbl in enumerate(label_order)}
    return label_order, label_to_idx


def compute_label_grid_idx(
    annotation_path,
    raw_w,
    raw_h,
    ds,
    keep_labels,
    other_label,
    label_to_idx,
    chunk_rows=256,
    off_comb_label=None,
    off_comb_threshold=None,
):
    """
    Compute label grid indices for downsampled annotation grid.

    Parameters
    ----------
    annotation_path : str or Path
        Path to annotation JSON file
    raw_w : int
        Raw image width
    raw_h : int
        Raw image height
    ds : int
        Downsample factor
    keep_labels : set
        Labels to keep as-is
    other_label : str
        Label for unmapped categories
    label_to_idx : dict
        Mapping from label to index
    chunk_rows : int
        Number of rows to process per chunk
    off_comb_label : str, optional
        Label for grid cells too far from any annotation (e.g., "off_comb" or "frame")
    off_comb_threshold : float, optional
        Distance threshold in pixels. Grid cells farther than this from any annotation
        will be labeled as off_comb_label. Defaults to ds (one grid cell spacing).

    Returns
    -------
    label_grid_idx : np.ndarray
        Grid of label indices, shape (grid_h, grid_w)
    """
    with open(annotation_path, "r") as f:
        cells = json.load(f)

    centers = np.array([[c["center_x"], c["center_y"]] for c in cells], dtype=np.float32)
    labels_idx = np.array(
        [label_to_idx[normalize_label(c["label"], keep_labels, other_label)] for c in cells],
        dtype=np.int16,
    )

    grid_w = int(np.ceil(raw_w / ds))
    grid_h = int(np.ceil(raw_h / ds))

    xs = (np.arange(grid_w, dtype=np.float32) + 0.5) * ds
    ys = (np.arange(grid_h, dtype=np.float32) + 0.5) * ds
    xs = np.clip(xs, 0, raw_w - 1)
    ys = np.clip(ys, 0, raw_h - 1)

    label_grid_idx = np.full((grid_h, grid_w), label_to_idx[other_label], dtype=np.int16)

    if centers.size:
        # Set default threshold to 50px, which is a reasonable value - slightly larger than the size of the comb cell
        if off_comb_threshold is None:
            off_comb_threshold = 50

        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(centers)
            for r0 in range(0, grid_h, chunk_rows):
                r1 = min(r0 + chunk_rows, grid_h)
                Xg, Yg = np.meshgrid(xs, ys[r0:r1])
                pts = np.column_stack([Xg.ravel(), Yg.ravel()])
                dists, nn_idx = tree.query(pts, k=1, workers=-1)
                labels = labels_idx[nn_idx].reshape(r1 - r0, grid_w)
                label_grid_idx[r0:r1] = labels

                # Mark grid cells that are too far from any annotation
                if off_comb_label is not None and off_comb_label in label_to_idx:
                    dists_grid = dists.reshape(r1 - r0, grid_w)
                    off_comb_mask = dists_grid > off_comb_threshold
                    label_grid_idx[r0:r1][off_comb_mask] = label_to_idx[off_comb_label]

        except Exception:
            # Chunked brute-force fallback
            for r0 in range(0, grid_h, chunk_rows):
                r1 = min(r0 + chunk_rows, grid_h)
                Xg, Yg = np.meshgrid(xs, ys[r0:r1])
                pts = np.column_stack([Xg.ravel(), Yg.ravel()]).astype(np.float32)
                diff = pts[:, None, :] - centers[None, :, :]
                dist2 = np.sum(diff * diff, axis=2)
                nn_idx = np.argmin(dist2, axis=1)
                labels = labels_idx[nn_idx].reshape(r1 - r0, grid_w)
                label_grid_idx[r0:r1] = labels

                # Mark grid cells that are too far from any annotation
                if off_comb_label is not None and off_comb_label in label_to_idx:
                    dists = np.sqrt(np.min(dist2, axis=1)).reshape(r1 - r0, grid_w)
                    off_comb_mask = dists > off_comb_threshold
                    label_grid_idx[r0:r1][off_comb_mask] = label_to_idx[off_comb_label]

    return label_grid_idx


def save_grid_npz(out_path, label_grid_idx, ds, raw_w, raw_h, label_order):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        label_grid=label_grid_idx,
        ds=np.array(ds, dtype=np.int16),
        raw_w=np.array(raw_w, dtype=np.int32),
        raw_h=np.array(raw_h, dtype=np.int32),
        label_order=np.array(label_order, dtype=object),
    )


def process_one_annotation(
    annotation_path,
    output_dir,
    raw_w,
    raw_h,
    ds,
    label_order,
    keep_labels,
    other_label,
    recalc=False,
    chunk_rows=256,
    off_comb_label=None,
    off_comb_threshold=None,
):
    """
    Process one annotation file and save as downsampled grid.

    Parameters
    ----------
    annotation_path : str or Path
        Path to annotation JSON file
    output_dir : str or Path
        Directory to save output npz file
    raw_w : int
        Raw image width
    raw_h : int
        Raw image height
    ds : int
        Downsample factor
    label_order : list
        Preferred ordering of labels
    keep_labels : set
        Labels to keep as-is
    other_label : str
        Label for unmapped categories
    recalc : bool
        If True, recalculate even if output exists
    chunk_rows : int
        Number of rows to process per chunk
    off_comb_label : str, optional
        Label for grid cells too far from any annotation (e.g., "off_comb" or "frame").
        If None, edge areas will be assigned to nearest cell label.
    off_comb_threshold : float, optional
        Distance threshold in pixels. Grid cells farther than this from any annotation
        will be labeled as off_comb_label. Defaults to ds (one grid cell spacing).

    Returns
    -------
    tuple : (str, str)
        (output_path, status) where status is "skip" or "saved"
    """
    annotation_path = Path(annotation_path)
    out_path = Path(output_dir) / f"{annotation_path.stem}_ds_{ds}.npz"
    if out_path.exists() and not recalc:
        return str(out_path), "skip"

    label_order, label_to_idx = build_label_index(label_order, keep_labels, other_label, off_comb_label)
    label_grid_idx = compute_label_grid_idx(
        annotation_path,
        raw_w,
        raw_h,
        ds,
        keep_labels,
        other_label,
        label_to_idx,
        chunk_rows=chunk_rows,
        off_comb_label=off_comb_label,
        off_comb_threshold=off_comb_threshold,
    )
    save_grid_npz(out_path, label_grid_idx, ds, raw_w, raw_h, label_order)
    return str(out_path), "saved"


def build_grid_metadata_df(combined_df: pd.DataFrame, grid_dir: Path, ds: int):
    grid_dir = Path(grid_dir)
    ann_df = combined_df[combined_df["is_annotated"]].copy()
    ann_df["grid_path"] = ann_df["annotation_path"].apply(
        lambda p: grid_dir / f"{Path(p).stem}_ds_{ds}.npz"
    )
    ann_df = ann_df[ann_df["grid_path"].map(lambda p: Path(p).exists())].copy()
    return ann_df


class GridLookup:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["ts_ns"] = pd.to_datetime(self.df["timestamp"]).astype("int64")

        # per-cam sorted index of timestamps and grid paths
        self.index = {}
        for cam, g in self.df.groupby("cam"):
            g = g.sort_values("ts_ns")
            self.index[int(cam)] = {
                "ts_ns": g["ts_ns"].to_numpy(dtype=np.int64),
                "grid_path": g["grid_path"].astype(str).to_numpy(),
            }

    @staticmethod
    @lru_cache(maxsize=64)
    def _load_grid(path_str: str):
        data = np.load(path_str, allow_pickle=True)
        label_grid = data["label_grid"]
        ds = int(data["ds"])
        raw_w = int(data["raw_w"])
        raw_h = int(data["raw_h"])
        label_order = list(data["label_order"])
        return label_grid, ds, raw_w, raw_h, label_order

    def _nearest_grid_paths(self, cam: int, timestamps):
        cam = int(cam)
        if cam not in self.index:
            raise KeyError(f"cam {cam} not in grid index")

        ts_ns = pd.to_datetime(timestamps).astype("int64").to_numpy()
        ts_arr = self.index[cam]["ts_ns"]
        paths = self.index[cam]["grid_path"]

        pos = np.searchsorted(ts_arr, ts_ns)
        idx = np.empty_like(pos)

        left_mask = pos == 0
        right_mask = pos >= len(ts_arr)
        mid_mask = ~(left_mask | right_mask)

        idx[left_mask] = 0
        idx[right_mask] = len(ts_arr) - 1

        if np.any(mid_mask):
            pos_mid = pos[mid_mask]
            left = ts_arr[pos_mid - 1]
            right = ts_arr[pos_mid]
            ts_mid = ts_ns[mid_mask]
            choose_left = (ts_mid - left) <= (right - ts_mid)
            idx_mid = np.where(choose_left, pos_mid - 1, pos_mid)
            idx[mid_mask] = idx_mid

        return paths[idx]

    def _nearest_grid_path(self, cam: int, timestamp) -> str:
        cam = int(cam)
        if cam not in self.index:
            raise KeyError(f"cam {cam} not in grid index")

        ts_ns = pd.to_datetime(timestamp).value
        ts_arr = self.index[cam]["ts_ns"]
        paths = self.index[cam]["grid_path"]

        pos = np.searchsorted(ts_arr, ts_ns)
        if pos == 0:
            return paths[0]
        if pos >= len(ts_arr):
            return paths[-1]

        before = ts_arr[pos - 1]
        after = ts_arr[pos]
        if abs(ts_ns - before) <= abs(after - ts_ns):
            return paths[pos - 1]
        return paths[pos]

    def lookup_labels(self, cam, timestamp, x, y, return_index=False):
        cams = np.asarray(cam)
        if cams.ndim == 0:
            return self.lookup_label(cam, timestamp, x, y, return_index=return_index)

        timestamps = np.asarray(timestamp)
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        if not (len(cams) == len(timestamps) == len(xs) == len(ys)):
            raise ValueError("cam, timestamp, x, y must have the same length")

        if return_index:
            out = np.full(len(cams), -1, dtype=int)
        else:
            out = np.empty(len(cams), dtype=object)
            out[:] = None

        for cam_id in np.unique(cams):
            cam_mask = cams == cam_id
            paths = self._nearest_grid_paths(cam_id, timestamps[cam_mask])
            cam_idx = np.flatnonzero(cam_mask)

            for path in np.unique(paths):
                path_mask = paths == path
                idxs = cam_idx[path_mask]
                if idxs.size == 0:
                    continue

                label_grid, ds, raw_w, raw_h, label_order = self._load_grid(path)
                x_sel = xs[idxs]
                y_sel = ys[idxs]
                valid = (x_sel >= 0) & (x_sel < raw_w) & (y_sel >= 0) & (y_sel < raw_h)
                if not np.any(valid):
                    continue

                gx = (x_sel[valid] // ds).astype(int)
                gy = (y_sel[valid] // ds).astype(int)
                gx = np.clip(gx, 0, label_grid.shape[1] - 1)
                gy = np.clip(gy, 0, label_grid.shape[0] - 1)
                label_idx = label_grid[gy, gx].astype(int)

                valid_idxs = idxs[valid]
                if return_index:
                    out[valid_idxs] = label_idx
                else:
                    labels = np.asarray(label_order, dtype=object)
                    out[valid_idxs] = labels[label_idx]

        return out

    def lookup_label(self, cam: int, timestamp, x: float, y: float, return_index=False):
        result = self.lookup_labels(
            np.array([cam]),
            np.array([timestamp]),
            np.array([x]),
            np.array([y]),
            return_index=return_index,
        )
        return result[0]
