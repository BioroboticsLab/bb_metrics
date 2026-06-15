import numpy as np
import pickle
import os
from pathlib import Path
import h5py
import gzip
from datetime import datetime
import pandas as pd

from . import get_config
from . import datafunctions as dfunc


max_delay = 1 # in seconds - don't calculate speed or turning velocity if speed if delay is greater than this
time_outside_trip = 60*2  # seconds
exit_dist_trip_threshold = 5  # in cm
moving_threshold = 0.05 # cm/s
inplace_time_threshold = 15 # sec
burst_threshold = 0.3 # cm/sec
burst_time_threshold = 2 # sec
large_turn = 2*np.pi
large_turn_time = 2 # sec
all_detections_dt = '60s'

# for calculations with all detections


def get_bin_indices_vectorized(x_array, y_array, cam):
    """
    Vectorized version of get_bin_indices.
    x_array, y_array, cam_array are 1D NumPy arrays.
    Returns (ix, iy) also as 1D arrays.
    """
    cfg = get_config()
    # SHIFT X by (camera * cfg.xpixels)
    # NOTE: dfunc.get_hive_cam0 might be a function that returns 0 or 1.
    #       If it depends on the camera array, either vectorize it or just do a known mapping.
    # Example (if hive A uses cams 0,1 => get_hive_cam0(cam) == 0 for both):
    # For a general approach, we might do:
    offset = dfunc.get_hive_cam0(cam)  # or a vector approach
    x_adj = x_array + (cam - offset) * cfg.xpixels

    # searchsorted for X, Y
    ix = np.searchsorted(cfg.x_edges, x_adj, side='right') - 1
    iy = np.searchsorted(cfg.y_edges, y_array, side='right') - 1

    # clamp
    max_ix = len(cfg.x_edges) - 2
    max_iy = len(cfg.y_edges) - 2
    ix = np.clip(ix, 0, max_ix)
    iy = np.clip(iy, 0, max_iy)

    return ix, iy

def find_closest_indices(ts_values, sorted_key_values):
    """
    Vectorized approach:
    For each timestamp in ts_values, find the index in sorted_key_values
    that is closest (absolute difference).
    Assumes sorted_key_values is a sorted 1D array of int64.
    Returns an array of integer indices.
    """
    # Preliminary index from searchsorted (gives the insertion position)
    idxs = np.searchsorted(sorted_key_values, ts_values)
    
    # We will compare to idxs-1 (left) and idxs (right), then pick whichever is closer
    # First clamp idxs to valid range [0, len-1]
    # Because searchsorted could return 0 or len(sorted_key_values)
    # We'll shift so that idxs is in [1, len-1], then we can safely do idxs-1
    idxs = np.clip(idxs, 1, len(sorted_key_values) - 1)

    left  = idxs - 1
    right = idxs

    left_diff  = ts_values - sorted_key_values[left]
    right_diff = sorted_key_values[right] - ts_values

    use_left = left_diff <= right_diff
    closest = np.where(use_left, left, right)
    return closest

def sum_counts_in_layer(xyhist, ix, iy, d):
    """
    Sums xyhist counts in the neighborhood of (ix, iy)
    with Chebyshev distance <= d, using direct NumPy slicing.

    We avoid building a list of neighbors, which was slow in Python,
    and let NumPy handle the sub-block summation in compiled code.
    """
    cfg = get_config()
    # xyhist.shape: (num_x_bins, num_y_bins)
    x_max = cfg.numxbins - 1
    y_max = cfg.numybins - 1
    
    # Compute bounding box
    ix_min = max(0, ix - d)
    ix_max_ = min(x_max, ix + d)
    iy_min = max(0, iy - d)
    iy_max_ = min(y_max, iy + d)
    
    # Slice the sub-array in one go, then sum
    sub_arr = xyhist[ix_min:ix_max_+1, iy_min:iy_max_+1]
    total = np.sum(sub_arr)
    return total


def get_surrounding_bins(ix, iy, d, x_max, y_max):
    """
    Return a list of (ix2, iy2) that are within distance d in index space:
      max(|ix2-ix|, |iy2-iy|) <= d

    x_max, y_max = maximum valid bin index for x and y.
    """
    ix_min = max(0, ix - d)
    ix_max_ = min(x_max, ix + d)
    iy_min = max(0, iy - d)
    iy_max_ = min(y_max, iy + d)
    
    neighbors = []
    for i2 in range(ix_min, ix_max_ + 1):
        for j2 in range(iy_min, iy_max_ + 1):
            neighbors.append((i2, j2))
    return neighbors

def sum_counts_in_layer_loop(xyhist, ix, iy, d):
    """
    Sums xyhist counts in the neighborhood of (ix, iy)
    with Chebyshev distance <= d.
    """
    cfg = get_config()
    # xyhist.shape: (num_x_bins, num_y_bins)
    x_max = cfg.numxbins - 1
    y_max = cfg.numybins - 1
    neighbor_bins = get_surrounding_bins(ix, iy, d, x_max, y_max)
    total = 0.0
    for (nx, ny) in neighbor_bins:
        total += xyhist[nx, ny]
    return total

def sum_counts_3distances(xyhist, ix, iy):
    """
    Return a tuple (count0, count1, count2) where:
      - count0 sums the bin (ix, iy) alone (d=0).
      - count1 sums the 3x3 block for d=1 around (ix, iy).
      - count2 sums the 5x5 block for d=2 around (ix, iy).

    If (ix, iy) is near an edge, it's clipped accordingly.
    """
    cfg = get_config()

    if type(xyhist)==float: # this happens if the xyhist is nan.  This can only happen if the all detections data was not processed, but the tagged bees were, for a certain time period
        return np.nan, np.nan, np.nan
    # xyhist.shape: (num_x_bins, num_y_bins)
    x_max = cfg.numxbins - 1
    y_max = cfg.numybins - 1

    # -------------------
    # d=0 => single cell
    # -------------------
    if ix < 0 or ix > x_max or iy < 0 or iy > y_max:
        count0 = 0
    else:
        count0 = xyhist[ix, iy]

    # -------------------
    # d=1 => up to 3x3
    # -------------------
    x0_1 = max(0, ix - 1)
    x1_1 = min(x_max, ix + 1)
    y0_1 = max(0, iy - 1)
    y1_1 = min(y_max, iy + 1)

    sub1 = xyhist[x0_1:x1_1 + 1, y0_1:y1_1 + 1]
    count1 = sub1.sum()

    # -------------------
    # d=2 => up to 5x5
    # -------------------
    x0_2 = max(0, ix - 2)
    x1_2 = min(x_max, ix + 2)
    y0_2 = max(0, iy - 2)
    y1_2 = min(y_max, iy + 2)

    sub2 = xyhist[x0_2:x1_2 + 1, y0_2:y1_2 + 1]
    count2 = sub2.sum()

    return count0, count1, count2

def compute_surrounding_bee_counts(dfbee, dict_detections_hive, detection_keys, detection_key_array):
    """
    For each row in dfbee, find the 'closest' timestamp's xyhist
    from `dict_detections_hive`, then sum counts in layers
    around (x, y, cam).
    
    Returns a dict of lists with keys like "numbees0", "numbees1", ...
    You can convert to DataFrame (pd.DataFrame(results)) or average them as needed.

    Assumes:
      - dict_detections_hive[ts]['xyhist'] is the 2D histogram array for that ts.
      - Timestamps in dict_detections_hive are spaced by 1 min or your chosen interval.
      - We want to find the nearest minute/interval key for each row's 'timestamp'.
      - We already filtered `dict_detections_hive` to the correct hive and time segment.
    """
    
    # Prepare the result container
    results = {"numbees0": [], "numbees1": [], "numbees2": []}

    if len(dict_detections_hive) == 0:
        # If there's no data in dict_detections_hive, everything is NaN
        for _ in range(len(dfbee)):
            for d in distances:
                results[f"numbees{d}"].append(np.nan)
        return results

    # compute closest timestamp for dfbee
    dfbee_ts_array = dfbee['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None).to_numpy(dtype='datetime64[ns]').astype('int64')
    closest_idx = find_closest_indices(dfbee_ts_array, detection_key_array)
    dfbee = dfbee.copy()
    dfbee['closest_key'] = [detection_keys[i] for i in closest_idx]

    ix, iy = get_bin_indices_vectorized(dfbee['x_pixels'].to_numpy(), dfbee['y_pixels'].to_numpy(), dfbee['cam_id'].to_numpy())
    dfbee['ix'] = ix
    dfbee['iy'] = iy

    for row in dfbee.itertuples(index=False):
        # row.closest_key, row.ix, row.iy, etc.
        xyhist = dict_detections_hive[row.closest_key]['xyhist']
        c0, c1, c2 = sum_counts_3distances(xyhist, row.ix, row.iy)
        results["numbees0"].append(c0)
        results["numbees1"].append(c1)
        results["numbees2"].append(c2)           

    average_results = {}
    for d in (0,1,2):
        colname = f"numbees{d}"
        # If no data or all NaN, np.nanmean returns NaN
        avg_val = np.nanmean(results[colname]) if len(results[colname]) > 0 else np.nan
        average_results[colname] = avg_val
    return average_results


def datafile_to_metrics(
    hive,
    timestamp_start,
    timestamp_end,
    datafiles,
    reprocess,
    time_division,
    min_num_detections,
    min_track_seconds,
    save_xy_hist,
    metrics_dir,
    update=True,
    grid_lookup=None,
    comb_label_order=None,
):
    cfg = get_config()
    try:
        metrics_dir = Path(metrics_dir)
        # Disable xyhist writes for fine-grained bins to avoid huge files/HDF clashes
        if pd.Timedelta(time_division) < pd.Timedelta('60min'):
            save_xy_hist = False

        # set filenames for saving
        # these should be e.g. metrics-5min-Hive_B_2024-08-27T03_00_00Z--2024-08-27T09_00_00Z.parquet
        # Per-camera files (2024+) start with 'Cam_<id>_'; strip that so the save name
        # is 'Hive_<hive>_<ISO>--<ISO>'. Combined files (2016/2019) have both cameras in
        # one file and no 'Cam_' prefix, so keep the full name (prefixed with '_'). Both
        # cases yield a name that parse_hive_name / parse_data_file_timestamps can read
        # downstream (e.g. build_day_xyhist).
        _src_name = Path(datafiles[0]).name  # data files already time-matched; use the first
        _name_suffix = _src_name[5:] if _src_name.startswith('Cam_') else ('_' + _src_name)
        hive_save_name = 'Hive_' + hive + _name_suffix
        metricsfile = metrics_dir / f"metrics-{time_division}-{hive_save_name}"
        xyhistfile = metrics_dir / f"xyhist-{time_division}-{hive_save_name.replace('.parquet', '.h5')}"

        if metricsfile.exists() and not reprocess:
            if update:
                newest_src_ts = max(os.path.getmtime(d) for d in datafiles)
                metrics_ts = metricsfile.stat().st_mtime
                if newest_src_ts <= metrics_ts:
                    return
            else:
                return

        df = pd.concat([pd.read_parquet(d) for d in datafiles])

        # Resolve the minimum-detections gate from tracked time using the fps
        # inferred from this file's timestamps, so the threshold is consistent
        # across sampling rates (3/6 fps and 14-fps bursts). An explicit
        # min_num_detections (not None) overrides and is used as-is.
        if min_num_detections is None:
            mts = min_track_seconds if min_track_seconds is not None else 2.0
            fps = dfunc.infer_fps(df['timestamp'])
            min_num_detections = max(1, int(np.ceil(mts * fps)))

        # Generate time segments based on the specified time division
        time_segments = pd.date_range(start=timestamp_start, end=timestamp_end, freq=time_division)
        if time_segments[-1] < timestamp_end: # just to make sure all are included
            time_segments = time_segments.append(pd.DatetimeIndex([timestamp_end]))

        # import all detections results for these time segements and concatenate
        dict_detections = {}
        time_segments_hr = pd.date_range(start=timestamp_start, end=timestamp_end, freq='h')
        if time_segments_hr[-1] < timestamp_end:  # just to make sure all are included
            time_segments_hr = time_segments_hr.append(pd.DatetimeIndex([timestamp_end]))
        for ts_start in time_segments_hr:
            outpath = metrics_dir / 'alldetections' / f"hive_{hive}_ts_{ts_start}_dt_{all_detections_dt}.pklz"
            outpath = Path(str(outpath).replace(':','_'))
            
            if outpath.exists():
                with gzip.open(outpath, 'rb') as f:
                    partial_dict = pickle.load(f)   
                # Merge the loaded dictionary into dict_detections
                for seg_start, hive_dict in partial_dict.items():
                    if seg_start not in dict_detections:
                        dict_detections[seg_start] = {}
                    for h, val in hive_dict.items():
                        dict_detections[seg_start][h] = val

        # Calculate 'flat' hive coordinates
        # This is done by simply reversing x of the back side camera to "fold it" onto the front one.  Using frame_width_cm to do this
        cam0 = dfunc.get_hive_cam0(df['cam_id'])
        cam1 = cam0+1
        df.loc[df['cam_id'] == cam1, 'x_hive_flat'] = cfg.frame_width_cm - df.loc[df['cam_id'] == cam1, 'x_hive']
        df.loc[df['cam_id'] == cam0, 'x_hive_flat'] = df.loc[df['cam_id'] == cam0, 'x_hive']

        # Initialize an empty list to store metrics for all time segments
        metrics_list = []

        # Open HDF5 file for writing xyhist data if save_xy_hist is True
        if save_xy_hist:
            hdf_file = h5py.File(xyhistfile, 'w')

        # Loop over each time segment
        for i in range(len(time_segments) - 1):
            time_segment_start = time_segments[i]
            time_segment_end = time_segments[i + 1]

            # Select data within the time segment
            df_segment = df[(df['timestamp'] >= time_segment_start) & (df['timestamp'] < time_segment_end)]
            if df_segment.empty:
                continue  # Skip if no data in this segment
                
            # Filter dict_detections for timestamps within [time_segment_start, time_segment_end)
            # plus the correct hive. We'll call the result dict_detections_hive.
            dict_detections_hive = {}
            for ts_key, val_dict in dict_detections.items():
                # Check if ts_key is in the time window:
                if (ts_key >= time_segment_start) and (ts_key < time_segment_end):
                    # Also ensure the dictionary has data for this hive
                    if hive in val_dict:
                        # We'll store only the hive-specific data at that timestamp
                        dict_detections_hive[ts_key] = val_dict[hive]
     
            # Sort keys (timestamps) for nearest-search
            detection_keys = sorted(dict_detections_hive.keys())
            if len(detection_keys)>0:
                detection_key_array = pd.Series(detection_keys).dt.tz_convert('UTC').dt.tz_localize(None).to_numpy(dtype='datetime64[ns]').astype('int64')

            # Get unique bee IDs in this segment
            bee_ids = df_segment['bee_id'].unique()

            # Loop through each bee_id to calculate metrics
            for bee_id in bee_ids:
                dfbee = df_segment[df_segment['bee_id'] == bee_id]
                if len(dfbee) >= min_num_detections:  # Only process if sufficient detections
                    # Calculate metrics and xyhist
                    metrics_dict, xyhist = get_metrics(
                        dfbee,
                        time_segment_end,
                        grid_lookup=grid_lookup,
                        comb_label_order=comb_label_order,
                    )
                    # Calculate density of surrounding bees:
                    if len(detection_keys)>0:
                        alld_density = compute_surrounding_bee_counts(dfbee, dict_detections_hive, detection_keys, detection_key_array)
                    else:
                        alld_density = {"numbees0": np.nan, "numbees1": np.nan, "numbees2": np.nan}
                    metrics_dict.update(alld_density)
                    # Add the time segment start and end to the metrics dictionary
                    metrics_dict['timestamp_start'] = time_segment_start
                    metrics_dict['timestamp_end'] = time_segment_end
                    # Append the metrics dictionary to the list
                    metrics_list.append(metrics_dict)
                    # Save xyhist to HDF5 file if save_xy_hist is True
                    if save_xy_hist:
                        save_xyhist_to_hdf(hdf_file, bee_id, time_segment_start, time_segment_end, xyhist)

        # Close the HDF5 file if it was opened
        if save_xy_hist:
            hdf_file.close()
            # print('Wrote xyhist to', os.path.basename(xyhistfile))

        # After processing all segments, save the metrics DataFrame
        df_metrics = pd.DataFrame(metrics_list)
        df_metrics.insert(0, 'hive', hive)

        # Move the timestamp columns for nicer organization and display
        if len(df_metrics)>0:
            cols = list(df_metrics.columns)  # Get the current column order
            cols.remove('timestamp_start')
            cols.remove('timestamp_end')
            cols.insert(2, 'timestamp_start')
            cols.insert(3, 'timestamp_end')
            df_metrics = df_metrics.reindex(columns=cols)

            df_metrics.to_parquet(metricsfile)
            print('Wrote metrics to', os.path.basename(metricsfile))
        else:
            print('Skipping, no content')

    except Exception as e:
        print('Error with', datafiles, ':', e)


def save_xyhist_to_hdf(hdf_file, bee_id, time_segment_start, time_segment_end, xyhist):
    """
    Saves the xyhist data to the HDF5 file for the given bee_id and time segment.

    Args:
        hdf_file: Open HDF5 file object.
        bee_id: The bee ID.
        time_segment_start: Start time of the segment.
        time_segment_end: End time of the segment.
        xyhist: The xyhist data array to save.
    """
    # Create group hierarchy: bee_id/day/hour
    bee_group = hdf_file.require_group(str(bee_id))
    day_str = time_segment_start.strftime('%Y%m%d')
    day_group = bee_group.require_group(day_str)
    hour_str = time_segment_start.strftime('%H')
    hour_group = day_group.require_group(hour_str)

    # Store xyhist array with compression
    hour_group.create_dataset(
        'xyhist', data=xyhist, compression='gzip', compression_opts=9
    )

    # Store metadata as attributes
    hour_group.attrs['bee_id'] = bee_id
    hour_group.attrs['timestamp_start'] = time_segment_start.isoformat()
    hour_group.attrs['timestamp_end'] = time_segment_end.isoformat()

## CALCULATING METRICS


def _compute_combhist(dfbee, grid_lookup, comb_label_order=None):
    if grid_lookup is None or dfbee.empty:
        return None, comb_label_order

    if comb_label_order is None:
        try:
            sample = dfbee.iloc[0]
            grid_path = grid_lookup._nearest_grid_path(sample["cam_id"], sample["timestamp"])
            _, _, _, _, comb_label_order = grid_lookup._load_grid(grid_path)
        except Exception:
            comb_label_order = None

    if not comb_label_order:
        return None, comb_label_order

    label_idxs = grid_lookup.lookup_labels(
        dfbee["cam_id"].to_numpy(),
        dfbee["timestamp"].to_numpy(),
        dfbee["x_pixels"].to_numpy(),
        dfbee["y_pixels"].to_numpy(),
        return_index=True,
    )
    label_idxs = label_idxs[label_idxs >= 0]

    if len(label_idxs) == 0:
        return None, comb_label_order

    counts = np.bincount(label_idxs, minlength=len(comb_label_order))
    combhist = counts / len(dfbee)
    return combhist, comb_label_order


def get_metrics(dfbee, last_df_time, grid_lookup=None, comb_label_order=None):
    """Input is dfbee, already selected for a certain amount of time."""
    cfg = get_config()
    dfbee = dfbee.sort_values(by='timestamp')
    dfbee = dfbee.reset_index(drop=True)
    cam0 = dfunc.get_hive_cam0(dfbee['cam_id'])
    
    # Dispersion:  root mean square dist from center point, consider 2D nest as 'flat' (only applies to x-coordinates)
    # the flattened coordinates need to already be calculated (done in the loop)
    dist2_from_mean = (dfbee['x_hive_flat']-dfbee['x_hive_flat'].mean())**2 + (dfbee['y_hive']-dfbee['y_hive'].mean())**2
    dispersion =  np.sqrt(np.mean(dist2_from_mean))
    
    # speed
    dist = np.sqrt(dfbee['x_hive'].diff()**2 + dfbee['y_hive'].diff()**2)
    dtimes = dfbee['timestamp'].diff().dt.total_seconds() # differences between frames in seconds
    samecamera = dfbee['cam_id'].diff()==0
    speed = dist / dtimes
    speed[np.logical_not(samecamera)|(dtimes>max_delay)] = np.nan  # don't count the switching camera ones, or when the delay is large
    speed[dtimes==0] = np.nan # just in case there are some zero differences (I think I filtered these out though)
    if np.sum(~np.isnan(speed))>2:
        speed_median = np.nanmedian(speed)
        speed_iqr = np.nanquantile(speed,0.75) - np.nanquantile(speed,0.25)        
    else:
        speed_median = np.nan
        speed_iqr = np.nan

    # fraction time moving, and number of stopped events
    is_stopped = speed < moving_threshold
    fraction_time_stopped = np.nanmean(is_stopped)
    # Count number of 'inplace' events, where the bee is stopped for a while
    inplace_events = 0
    stopped_duration = 0  # initialize before the loop 
    # Iterate over the is_stopped array and corresponding time differences
    for stopped, dtime in zip(is_stopped, dtimes):
        if stopped:
            # If the bee is stopped, accumulate the duration
            stopped_duration += dtime
        else:
            # If the bee starts moving, check if the previous stopped duration qualifies as an inplace event
            if stopped_duration >= inplace_time_threshold:
                inplace_events += 1
            # Reset the stopped duration
            stopped_duration = 0
    # Handle the case where the bee remains stopped at the end
    if stopped_duration >= inplace_time_threshold:
        inplace_events += 1
    # inplace_events now contains the count of "inplace" events    

    # number of 'burst' speed events, where moving fast continuously
    # Define whether the bee is in a "burst" (speed above the threshold)
    is_bursting = speed > burst_threshold
    # Initialize a counter for burst events
    burst_events = 0
    burst_duration = 0 # initialize before the loop
    # Iterate over the is_bursting array and corresponding time differences
    for bursting, dtime in zip(is_bursting, dtimes):
        if bursting:
            # If the bee is bursting, accumulate the duration
            burst_duration += dtime
        else:
            # If the bee stops bursting, check if the previous bursting duration qualifies as a burst event
            if burst_duration >= burst_time_threshold:
                burst_events += 1
            # Reset the burst duration
            burst_duration = 0
    # Handle the case where the bee remains bursting at the end
    if burst_duration >= burst_time_threshold:
        burst_events += 1
    # burst_events now contains the count of "burst" events    
    
    # orientation-related metrics
    orientation_col = 'orientation' if ('orientation' in dfbee.columns) else 'orientation_pixels'
    angular_change = np.diff(np.unwrap(dfbee[orientation_col]))
    angvel = angular_change / dtimes[1:]
    rms_angvel = np.sqrt(np.nanmean(angvel**2))
    # number of 'large turn' events
    large_turn_events, cumulative_turn, cumulative_time = 0, 0, 0 # Initialize variables
    # Loop through angular changes and time intervals
    for delta_angle, delta_time in zip(np.abs(angular_change), dtimes):
        if cumulative_time < large_turn_time:
            # Accumulate turn and time if within the time window
            cumulative_turn += delta_angle
            cumulative_time += delta_time
            # Check if a large turn event is detected
            if cumulative_turn > large_turn:
                large_turn_events += 1
                # Reset accumulators after detecting a large turn
                cumulative_turn = 0
                cumulative_time = 0
        else:
            # Reset accumulators if time window is exceeded without a large turn
            cumulative_turn = delta_angle
            cumulative_time = delta_time
    # large_turn_events now contains the count of large turn events

    # histograms
    num_detections = len(dfbee)
    xyhist = dfunc.getxyhist(dfbee['x_pixels'],dfbee['y_pixels'],dfbee['cam_id'])

    ## Comb usage
    combhist, comb_label_order = _compute_combhist(dfbee, grid_lookup, comb_label_order)

    # Fraction squares visited in this time period
    fraction_squares_visited = np.nansum(xyhist>0)/(cfg.numxbins*cfg.numybins)

    # Add bee_id to the dictionary
    bee_id = dfbee['bee_id'].iloc[0]

    # Frame and exit-distance metrics assume the 4-frame, negative-y comb geometry
    # (cfg.offset_div_cm; exit at the (0,0) corner point). Sites with a different
    # comb layout (e.g. the 2019 2-frame hive) set cfg.compute_frame_exit_metrics =
    # False to skip them. The getattr default keeps the historical behavior (and the
    # exact metrics_dict column order) for every year that does not set the flag.
    if getattr(cfg, 'compute_frame_exit_metrics', True):
        # which 'frame' of the observation hive the bee was on
        framehist = dfunc.getframehist(dfbee['y_hive'],dfbee['cam_id']) / num_detections

        # Exit distance (median)
        # 2025: the exit is located at (0cm, 0cm), in hive coordinates, by definition - i.e. the 'corner point'
        # distance from exit
        x_exit, y_exit = 0, 0
        # get shift coordinates
        #  shift the y pixels for ones on frame 2 (back side, bottom), to define the 'shortest path' to the exit
        y_exitdist = dfbee['y_hive'].copy()
        sel = (dfbee['cam_id']==cam0+1)&(dfbee['y_hive']>-cfg.offset_div_cm)
        y_exitdist[sel] = -2*cfg.offset_div_cm - y_exitdist[sel]
        exitdist = np.sqrt( (dfbee['x_hive_flat']-x_exit)**2 + (y_exitdist-y_exit)**2 )
        exitdist_median = np.median(exitdist)

        # Distance from center of current frame
        framebins = [-cfg.offset_div_cm*2-10, -cfg.offset_div_cm, 10] # set limits to cover the frame (negative y_hive coords)
        # get 'framenum' as digitized label
        currentframe = np.tile(np.nan,len(dfbee))
        sel0, sel1 = dfbee['cam_id']==cam0, dfbee['cam_id']==cam0+1
        # Digitize y_pixels for cam0 and assign frame numbers 0 and 1
        if sel0.any():
            currentframe[sel0] = np.digitize(dfbee.loc[sel0, 'y_hive'], framebins) - 1
        # Digitize y_pixels for cam1 and assign frame numbers 2 and 3
        if sel1.any():
            currentframe[sel1] = np.digitize(dfbee.loc[sel1, 'y_hive'], framebins) - 1 + 2
        # Calculate frame centers
        x_center = cfg.frame_width_cm/2
        bottomcenter, topcenter = -0.5*cfg.offset_div_cm, -1.5*cfg.offset_div_cm

        y_centers = {0: bottomcenter, 1: topcenter, 2: bottomcenter, 3: topcenter}
        # Loop through each frame number and calculate the median distance
        median_distances = np.full(4, np.nan)
        for frame_num in range(4):
            bee_in_frame = currentframe == frame_num
            if np.any(bee_in_frame):
                # Calculate distances from the frame center
                x_diff = dfbee.loc[bee_in_frame, 'x_pixels'] - x_center
                y_diff = dfbee.loc[bee_in_frame, 'y_pixels'] - y_centers[frame_num]
                distances = np.sqrt(x_diff**2 + y_diff**2)
                median_distances[frame_num] = np.median(distances)

        # Num Outside Trips (approx measure)
        numtrips = 0
        longbreaks = np.where(dtimes>=time_outside_trip)[0]
        for breakind in longbreaks:
            # if was near the exit before last being seen, and next seen near the exit again
            if np.all(exitdist.iloc[breakind-1:breakind+1]<exit_dist_trip_threshold):
                numtrips = numtrips+1
        # check for a trip at the end
        if exitdist.iloc[len(dfbee)-1]<exit_dist_trip_threshold:
            # time until the end is more than the outside trip time, count it
            if (last_df_time - dfbee.iloc[len(dfbee)-1]['timestamp']).total_seconds() > time_outside_trip:
                numtrips = numtrips+1

        # Create a dictionary of metrics
        metrics_dict = {
            'bee_id': bee_id,
            'num_detections': num_detections,
            'dispersion': dispersion,
            'speed_median': speed_median,
            'speed_iqr': speed_iqr,
            'num_trips': numtrips,
            'fraction_squares_visited': fraction_squares_visited,
            'exit_distance_median': exitdist_median,
            'frame_0_hist': framehist[0, 0],
            'frame_1_hist': framehist[0, 1],
            'frame_2_hist': framehist[1, 0],
            'frame_3_hist': framehist[1, 1],
            'frame_0_centermedian': median_distances[0],
            'frame_1_centermedian': median_distances[1],
            'frame_2_centermedian': median_distances[2],
            'frame_3_centermedian': median_distances[3],
            'inplace_events': inplace_events,
            'burst_events': burst_events,
            'large_turn_events': large_turn_events
        }
    else:
        # 2-frame / non-standard comb layout: geometry-independent metrics only
        # (frame_*_hist, frame_*_centermedian, exit_distance_median and num_trips
        # are omitted because they depend on the 4-frame exit geometry).
        metrics_dict = {
            'bee_id': bee_id,
            'num_detections': num_detections,
            'dispersion': dispersion,
            'speed_median': speed_median,
            'speed_iqr': speed_iqr,
            'fraction_squares_visited': fraction_squares_visited,
            'inplace_events': inplace_events,
            'burst_events': burst_events,
            'large_turn_events': large_turn_events
        }

    if combhist is not None and comb_label_order is not None:
        for idx, label in enumerate(comb_label_order):
            metrics_dict[f"combhist_{label}"] = combhist[idx]

    return metrics_dict, xyhist
