#!/bin/bash
# MakeVideosPerHive.sh
# Combine cam pairs per hive into side-by-side rotated videos.
# Default: scale to 50%; use --full-res to keep full resolution.

# --- Default Configuration ---
comb_images_dir="./"
OUTDIR="videos_per_hive"
FULL_RES=false

# --- Parse command-line arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --comb-images-dir)
      comb_images_dir="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --full-res)
      FULL_RES=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--comb-images-dir DIR] [--outdir DIR] [--full-res]"
      exit 1
      ;;
  esac
done

# Ensure output directory exists
mkdir -p "$OUTDIR"

echo "Using comb_images_dir: $comb_images_dir"
echo "Using output directory: $OUTDIR"
echo "Full resolution: $FULL_RES"

# --- Define hives as: hive_name cam_left cam_right ---
HIVES=(
  "A cam-0 cam-1"
  "B cam-2 cam-3"
  "C cam-4 cam-5"
  "D cam-6 cam-7"
)

# --- Main loop ---
for hive_info in "${HIVES[@]}"; do
    set -- $hive_info
    hive=$1
    cam_left_dir="$comb_images_dir/$2"
    cam_right_dir="$comb_images_dir/$3"

    if [ ! -d "$cam_left_dir" ] || [ ! -d "$cam_right_dir" ]; then
        echo "Skipping hive $hive ($cam_left_dir,$cam_right_dir) - missing directories"
        continue
    fi

    echo "Processing hive $hive from $cam_left_dir + $cam_right_dir ..."

    # Sorted lists of PNGs (use sort -V for natural/version number sorting)
    left_imgs=($(ls "$cam_left_dir"/*.png 2>/dev/null | sort -V))
    right_imgs=($(ls "$cam_right_dir"/*.png 2>/dev/null | sort -V))

    num_left=${#left_imgs[@]}
    num_right=${#right_imgs[@]}
    num=$(( num_left>num_right ? num_left : num_right ))

    echo "  Found $num_left images in $cam_left_dir"
    echo "  Found $num_right images in $cam_right_dir"

    if [ "$num" -eq 0 ]; then
        echo "  No matching images for hive $hive, skipping"
        continue
    fi

    echo "  Combining $num frames (left: $num_left, right: $num_right)..."
    echo "  Checking image dimensions..."

    # Check dimensions of all images to ensure consistency
    # Get expected dimensions from first image of left camera
    expected_dims=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "${left_imgs[0]}" 2>/dev/null)
    expected_w=$(echo $expected_dims | cut -d'x' -f1)
    expected_h=$(echo $expected_dims | cut -d'x' -f2)

    echo "  Expected dimensions (from first left image): ${expected_w}x${expected_h}"

    # Check all left camera images
    mismatched_images=()
    for img in "${left_imgs[@]}"; do
        dims=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$img" 2>/dev/null)
        if [ "$dims" != "$expected_dims" ]; then
            mismatched_images+=("$(basename "$img"): $dims (expected ${expected_w}x${expected_h})")
        fi
    done

    # Check all right camera images
    for img in "${right_imgs[@]}"; do
        dims=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$img" 2>/dev/null)
        if [ "$dims" != "$expected_dims" ]; then
            mismatched_images+=("$(basename "$img"): $dims (expected ${expected_w}x${expected_h})")
        fi
    done

    # If there are mismatched images, report and exit
    if [ ${#mismatched_images[@]} -gt 0 ]; then
        echo ""
        echo "  ERROR: Some images have different resolutions!"
        echo "  Expected: ${expected_w}x${expected_h}"
        echo "  Mismatched images:"
        for mismatch in "${mismatched_images[@]}"; do
            echo "    - $mismatch"
        done
        echo ""
        echo "  Please ensure all images are in the original camera reference frame (not rotated)."
        echo "  Skipping hive $hive"
        continue
    fi

    echo "  All images have consistent dimensions: ${expected_w}x${expected_h}"

    TMPDIR=$(mktemp -d)

    # Decide scaling and suffix
    if $FULL_RES; then
        suffix="_full-res"
        scale_filter="[0:v]transpose=1[left];[1:v]transpose=1[right];[left][right]hstack=inputs=2[out]"
    else
        suffix=""
        scale_filter="[0:v]transpose=1[left];[1:v]transpose=1[right];[left][right]hstack=inputs=2[stack];[stack]scale=iw/2:ih/2[out]"
    fi

    # Generate combined frames
    failed_frames=0
    for ((i=0; i<num; i++)); do
        # Use last frame if we've run out of images for this camera
        if [ $i -lt $num_left ]; then
            left=${left_imgs[$i]}
        else
            left=${left_imgs[$((num_left-1))]}
        fi

        if [ $i -lt $num_right ]; then
            right=${right_imgs[$i]}
        else
            right=${right_imgs[$((num_right-1))]}
        fi

        out="$TMPDIR/frame_$(printf "%05d" $i).png"

        # Run ffmpeg and check if output file was created (more reliable than exit code)
        ffmpeg -y -loglevel error -i "$left" -i "$right" \
          -filter_complex "$scale_filter" \
          -map "[out]" "$out" 2>/dev/null

        # Check if the output file was actually created
        if [ ! -f "$out" ]; then
            echo "  WARNING: Failed to process frame $i (left: $(basename "$left"), right: $(basename "$right"))"
            ((failed_frames++))
        fi
    done

    if [ $failed_frames -gt 0 ]; then
        echo "  WARNING: $failed_frames frames failed to process"
    fi

    # Count frames actually created
    actual_frames=$(ls "$TMPDIR"/frame_*.png 2>/dev/null | wc -l)
    echo "  Successfully created $actual_frames combined frames"

    # Create video
    ffmpeg -y -framerate 2 -pattern_type glob -i "$TMPDIR/frame_*.png" \
        -c:v libx264 -pix_fmt yuv420p "$OUTDIR/hive_${hive}${suffix}.mp4"

    rm -rf "$TMPDIR"
    echo "  -> Created $OUTDIR/hive_${hive}${suffix}.mp4"
done