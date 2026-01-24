# bb_metrics

A Python package for processing and analyzing honey bee behavioral data from BeesBook tracking experiments. This package handles trajectory processing, behavioral metrics calculation, and feeder/exit camera data analysis.

## Overview

`bb_metrics` provides a complete pipeline for transforming raw tracking data into analyzable behavioral metrics for honey bee colonies. It processes:

- **Trajectory data** from overhead observation hive cameras (tracked bee positions over time)
- **Feeder/exit camera detections** (bee visits to feeding stations and hive exits)
- **Behavioral metrics** computed at various time scales (1min, 5min, hourly, daily)

The package supports multi-hive experiments with configurable experimental setups and treatment schedules.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bb_metrics.git
cd bb_metrics

# Install dependencies (recommended: use a virtual environment)
pip install -r requirements.txt
```

## Configuration

Before using the package, set up a configuration file for your experiment. Example configs are provided in `config/`:

- `berlin2025.py` - Multi-hive setup (4 hives, A-D)
- `konstanz2025.py` - Single-hive setup


## Processing Pipeline

The data processing workflow consists of three main steps:

### Step 0: Calibration and Setup

**Notebook**: `0 - Get corner points and px-cm.ipynb`

- Load comb background images
- Annotate frame corners and calibration distances
- Calculate pixel-to-cm conversion factors
- Verify camera rotation settings

**Notebook**: `0 - Save tag info and feeder use data.ipynb`

- Load bee tag assignment data
- Prepare treatment schedules
- Set up metadata for the experiment

### Step 1: Comb Annotations (Optional)

**Notebook**: `1 - Comb images and annotations.ipynb`

Creates spatially-resolved comb substrate maps from external annotation tool output:

```python
from bb_metrics import datafunctions as dfunc

# Load annotation images from specialized comb annotation tool
# Each annotation image has color-coded regions for substrate types
annot_img = load_annotation_image(annot_path)

# Convert to label grids mapping pixel coordinates to substrate labels
label_grid = dfunc.annotation_image_to_grid(annot_img, label_colors)

# Save grids as .npz files for later lookup
np.savez(grid_path, label_grid=label_grid, ds=downsample_factor,
         raw_w=width, raw_h=height, label_order=label_order)
```

**What it does:**
- Loads annotation images from a specialized comb substrate annotation tool (separate from CVAT)
- Creates downsampled grids mapping pixel coordinates to substrate labels
- Supports labels: `empty_cell`, `open_brood`, `capped_brood`, `capped_honey`, `other`
- Enables per-detection substrate classification in metrics calculation

**Outputs**: `grid_<cam>_<timestamp>.npz` files per camera per annotation timepoint

### Step 1: Process Trajectories

**Notebook**: `1 - Process trajectories.ipynb`

Converts raw tracking data (`.dill` files) into cleaned trajectory parquet files:

```python
from bb_metrics import trajectories as traj

# Process trajectory files
pairs, unmatched = traj.process_directory(
    cfg.trackdir,
    cfg.traj_outdir,
    cam_hive_map=cfg.cam_hive_map,
    reprocess=False,
    num_processes=6
)
```

**What it does:**
- Loads raw track files from tracking pipeline
- Filters speed jumps and implausible movements
- Applies camera rotation corrections
- Converts coordinates to hive-centric system
- Saves cleaned trajectories as parquet files (one per camera per video)

**Outputs**: `trajectory_data/{hive}_{camL}_{camR}_{start}_{end}.parquet`

### Step 1b: Process Feeder/Exit Camera Data

**Notebook**: `1 - Feeder cams data.ipynb`

Processes detection data from feeder and exit cameras:

```python
from bb_metrics import feedercams as fc

# Process daily detection files
fc.process_datedir(
    input_dir=cfg.feedercam_input_dir,
    output_dir=cfg.feedercam_daily_dir,
    date_start=cfg.startday,
    date_end=cfg.endday
)

# Calculate average counts per video
fc.process_daily_files(
    daily_dir=cfg.feedercam_daily_dir,
    output_dir=cfg.feedercam_avg_dir,
    date_start=cfg.startday,
    date_end=cfg.endday
)
```

**What it does:**
- Aggregates per-video detections into daily files
- Computes average counts (total, tagged, untagged) per 30-second video
- Handles both CLAHE and non-CLAHE processed detections
- Converts bee IDs to standard format

**Outputs**:
- Daily files: `YYYYMMDD_feedercam-c.parquet`, `YYYYMMDD_feedercam-nc.parquet`
- Average counts: `avgcounts/YYYYMMDD_feedercam-c.parquet`

### Step 2: Calculate Metrics

**Notebook**: `2 - Calculate metrics.ipynb`

Computes behavioral metrics from trajectory data:

```python
from bb_metrics import metrics_pipeline as mp

# Build camera pairs from trajectory files
pairs, unmatched = mp.build_pairs_from_traj(
    traj_files,
    cfg.cam_hive_map
)

# Calculate metrics at 1-minute resolution
mp.run_metrics_from_pairs(
    pairs,
    time_division="1min",
    min_num_detections=12,
    save_xy_hist=True,
    metrics_dir=cfg.metrics_dir,
    num_processes=6,
    grid_lookup=grid_lookup,
    comb_label_order=comb_label_order
)
```

**Metrics computed** (per bee, per time segment):
- **Movement**: dispersion, speed (median, IQR), number of trips
- **Spatial**: fraction of squares visited, distance to exit, distance to top feeder
- **Comb usage**: histogram of positions across 4 frames, frame center positions
- **Activity**: in-place events, burst events, large turn events
- **Social**: number of nearby bees (0-2 bee-distances)
- **Comb state**: empty cells, open brood, capped brood, capped honey, other cells

**Outputs**:
- `metrics-1min-{start}-{end}.parquet` (one row per bee per 1-min segment)
- `metrics-5min-{start}-{end}.parquet` (5-min aggregation)
- Can also generate hourly (`60min`) aggregations

### Step 2b: Calculate Visit Metrics

**Notebook**: `2 - Calculate metrics.ipynb` (continuation)

Computes feeder and exit visit statistics:

```python
# Calculate feeder visits
df_feedervisits = mp.calculate_feeder_visits(
    df_feedercam,
    min_interval_seconds=1,
    max_interval_seconds=300
)

# Calculate exit visits
df_exitvisits = mp.calculate_exit_visits(
    df_exitcam,
    min_interval_seconds=1,
    max_interval_seconds=300
)
```

**What it does:**
- Groups consecutive detections into visit events
- Computes visit duration, detection count per visit
- Links visits to individual bees (for tagged detections)

**Outputs**:
- `df_feedervisits.parquet` (bee_id, cam_id, start_time, end_time, duration_seconds, detection_count)
- `df_exitvisits.parquet`

### Step 3: Analysis and Visualization

**Notebooks**:
- `3 - Metrics analysis - v1.ipynb` - Behavioral metrics analysis
- `3 - Feederplots.ipynb` - Feeder/exit camera visualizations

Load and analyze processed metrics:

```python
# Load metrics
df_metrics = pd.read_parquet(cfg.metrics_dir / 'metrics-5min-*.parquet')
df_feedervisits = pd.read_parquet(cfg.metrics_dir / 'df_feedervisits.parquet')

# Add treatment information
# ... (see notebooks for treatment processing)

# Plot metrics by hour of day, treatment status, etc.
# Use displayfunctions (bp) for standardized plots
```

**Common analyses:**
- Hour-of-day activity patterns
- Treatment vs. control comparisons
- Metrics aligned to feeder visits
- Daily and hourly aggregations
- Weather correlations

## Module Overview

### Core Modules

- **`trajectories.py`** - Trajectory filtering, processing, and parquet conversion
- **`metrics_pipeline.py`** - High-level pipeline functions for pairing files and dispatching metric calculations
- **`metricsfunctions.py`** - Core metric computation functions (movement, spatial, comb histograms)
- **`feedercams.py`** - Feeder/exit camera detection processing and visit calculation
- **`calibration.py`** - Camera calibration, corner detection, pixel-to-cm conversion
- **`rotation.py`** - Coordinate system rotation utilities

### Utility Modules

- **`datafunctions.py`** - Data loading, weather data, date parsing utilities, `GridLookup` class for comb substrate queries
- **`displayfunctions.py`** - Standardized plotting functions for metrics and time series

### Configuration

- **`config/`** - Experiment-specific configuration files
- **`__init__.py`** - Config management (`set_config()`, `get_config()`)

## Key Functions Reference

### Trajectory Processing

```python
# Process a single trajectory file
traj.process_file(
    input_path,
    output_path,
    cam_hive_map,
    dftags=None,
    min_num_obs=100,
    reprocess=False
)

# Filter speed jumps
df_filtered = traj.filter_speed_jumps(df, min_num_obs=100, max_speed=3.0)
```

### Metrics Calculation

```python
# Calculate metrics for a bee trajectory
metrics_dict = mfunc.compute_metrics_for_bee(
    df_bee,
    grid_lookup,
    comb_label_order,
    pixels_per_cm,
    save_xy_hist=True
)
```

### Feeder Camera Processing

```python
# Clean and standardize feeder camera detections
df_clean = fc.get_df_feedercam(df_raw, timezone="Europe/Berlin")

# Calculate average counts per video
df_avg = fc.get_average_counts_daily(df_detections, video_duration_seconds=30)
```

### Display Functions

```python
# Plot time series with treatment shading
bp.plot_feedercam_segments(ax, df, color='blue', whichcol='taggedcounts', gap_seconds=3660)
bp.shade_treatments(ax, treat_df, color='red', alpha=0.2)

# Format axes with date ranges
bp.common_plot_formatting(axes, startday, endday)
```

## Output Data Formats

### Trajectory Files

Parquet files with columns:
- `bee_id` - Unique bee identifier (ferwar format)
- `timestamp` - Detection timestamp
- `cam_id` - Camera ID
- `x_hive`, `y_hive` - Hive-centric coordinates (rotated, cm)
- `x_pixels`, `y_pixels` - Original pixel coordinates
- `orientation` - Bee orientation angle

### Metrics Files

Parquet files with one row per bee per time segment:
- Bee info: `hive`, `bee_id`, `timestamp_start`, `timestamp_end`, `num_detections`
- Movement: `dispersion`, `speed_median`, `speed_iqr`, `num_trips`
- Spatial: `fraction_squares_visited`, `exit_distance_median`
- Comb usage: `frame_0_hist` through `frame_3_hist`, centermedian values
- Activity: `inplace_events`, `burst_events`, `large_turn_events`
- Social: `numbees0`, `numbees1`, `numbees2`
- Comb state: `combhist_empty_cell`, `combhist_open_brood`, etc.

### Visit Files

Parquet files with one row per visit:
- `bee_id`, `cam_id`, `hive`
- `start_time`, `end_time`
- `duration_seconds`, `detection_count`

## Tips and Best Practices

1. **Always set config first**: Call `bb_metrics.set_config(cfg)` at the start of each notebook
2. **Use reprocess flags carefully**: Set `reprocess=False` to skip existing files and save time
3. **Parallel processing**: Adjust `num_processes` based on available CPU cores
4. **Memory management**: Process data in chunks for large datasets; use time segment filtering
5. **Treatment analysis**: Mark treatment days carefully, accounting for timezone differences
6. **Data quality**: Check for missing data periods and handle NaN values appropriately

## Troubleshooting

**Timezone issues**: Ensure all timestamps use consistent timezones (UTC or Europe/Berlin). Use `.tz_localize()` or `.tz_convert()` as needed.

**Missing files**: Check that trajectory files are paired correctly (two cameras per hive per time segment).

**Memory errors**: Reduce `num_processes`, process fewer files at once, or use coarser time divisions (5min instead of 1min).

**Speed jumps**: If too much data is filtered out, increase `max_speed` threshold in `filter_speed_jumps()`.
