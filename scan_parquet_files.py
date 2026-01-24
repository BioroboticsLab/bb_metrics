#!/usr/bin/env python3
"""
Scan parquet files to identify corrupted, empty, or suspicious files.

This script checks parquet files for:
- Files that cannot be read (corrupted)
- Files with 0 rows (empty data)
- Files with suspiciously small sizes (likely incomplete writes)
- Files missing required columns

Examples:
  # Scan and report issues
  python scan_parquet_files.py /path/to/data --dry-run

  # Scan and remove problematic files
  python scan_parquet_files.py /path/to/data

  # Only check specific pattern
  python scan_parquet_files.py /path/to/data --pattern "Cam_*.parquet"

  # Check for required columns
  python scan_parquet_files.py /path/to/data --required-columns timestamp,x_pixels,y_pixels
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def check_parquet_file(
    file_path: Path,
    required_columns: list[str] | None = None,
    suspicious_size_threshold: int = 2000
) -> tuple[str, str, int]:
    """
    Check a single parquet file for issues.

    Args:
        file_path: Path to parquet file
        required_columns: List of required column names (optional)
        suspicious_size_threshold: Files smaller than this (bytes) are flagged as suspicious

    Returns:
        Tuple of (status, message, file_size) where status is one of:
        - "ok": File is valid
        - "unreadable": File cannot be read
        - "empty": File has 0 rows (readable but no data)
        - "suspicious_size": File size is suspiciously small AND unreadable/corrupted
        - "missing_columns": File missing required columns
    """
    try:
        # Check file size
        file_size = file_path.stat().st_size

        # Try to read the file
        try:
            df = pd.read_parquet(file_path)
        except Exception as read_err:
            # File is unreadable - if also small, mark as suspicious_size, else unreadable
            if file_size < suspicious_size_threshold:
                return ("suspicious_size", f"Size={file_size} bytes, unreadable: {type(read_err).__name__}", file_size)
            else:
                return ("unreadable", f"Size={file_size} bytes, {type(read_err).__name__}: {read_err}", file_size)

        # File is readable - check if empty
        if len(df) == 0:
            return ("empty", f"Size={file_size} bytes, 0 rows, columns={list(df.columns)}", file_size)

        # Check for required columns
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                return ("missing_columns", f"Missing: {missing}, has: {list(df.columns)}", file_size)

        return ("ok", f"{len(df)} rows, {len(df.columns)} columns", file_size)

    except Exception as e:
        file_size = 0
        try:
            file_size = file_path.stat().st_size
        except:
            pass
        return ("unreadable", f"Size={file_size} bytes, {type(e).__name__}: {e}", file_size)


def scan_directory(
    directory: Path,
    pattern: str,
    required_columns: list[str] | None,
    suspicious_size_threshold: int,
    dry_run: bool,
    verbose: bool
) -> dict[str, list[Path]]:
    """
    Scan directory for parquet files and categorize by status.

    Returns:
        Dictionary mapping status to list of file paths
    """
    if not directory.exists():
        print(f"[ERROR] Directory does not exist: {directory}", file=sys.stderr)
        return {}

    print(f"[scan] {directory}")
    print(f"[scan] Pattern: {pattern}")
    if required_columns:
        print(f"[scan] Required columns: {required_columns}")

    # Find all matching files
    files = sorted(directory.rglob(pattern))
    print(f"[scan] Found {len(files)} files to check")

    results = {
        "ok": [],
        "unreadable": [],
        "empty": [],
        "suspicious_size": [],
        "missing_columns": []
    }

    for i, file_path in enumerate(files, 1):
        if verbose and i % 100 == 0:
            print(f"[progress] Checked {i}/{len(files)} files...")

        status, message, file_size = check_parquet_file(
            file_path,
            required_columns,
            suspicious_size_threshold
        )

        results[status].append(file_path)

        if status != "ok":
            rel_path = file_path.relative_to(directory) if file_path.is_relative_to(directory) else file_path
            print(f"[{status}] {rel_path}")
            if verbose:
                print(f"  {message}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Scan parquet files for corruption, empty data, or missing columns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan for issues (dry-run)
  %(prog)s /mnt/share/beesbook2025/results/data_alldetections --dry-run

  # Scan and remove problematic files
  %(prog)s /mnt/share/beesbook2025/results/data_alldetections

  # Check for required columns
  %(prog)s /path/to/data --required-columns timestamp,x_pixels,y_pixels,cam_id

  # Custom suspicious size threshold
  %(prog)s /path/to/data --suspicious-size 5000
        """
    )

    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more directories to scan recursively"
    )

    parser.add_argument(
        "--pattern",
        default="*.parquet",
        help='Glob pattern for files to check (default: "*.parquet")'
    )

    parser.add_argument(
        "--required-columns",
        help='Comma-separated list of required column names (e.g., "timestamp,x_pixels,y_pixels")'
    )

    parser.add_argument(
        "--suspicious-size",
        type=int,
        default=2000,
        help="Flag files smaller than this many bytes as suspicious (default: 2000)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan only; do not delete files"
    )

    parser.add_argument(
        "--remove-problematic",
        action="store_true",
        help="Remove files that are unreadable, empty, or suspicious size (requires confirmation)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each problematic file"
    )

    args = parser.parse_args()

    required_columns = None
    if args.required_columns:
        required_columns = [col.strip() for col in args.required_columns.split(",")]

    all_results = {
        "ok": [],
        "unreadable": [],
        "empty": [],
        "suspicious_size": [],
        "missing_columns": []
    }

    for directory in args.directories:
        dir_path = Path(directory)
        results = scan_directory(
            dir_path,
            args.pattern,
            required_columns,
            args.suspicious_size,
            args.dry_run,
            args.verbose
        )

        # Aggregate results
        for status, files in results.items():
            all_results[status].extend(files)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"OK:               {len(all_results['ok'])} files")
    print(f"Unreadable:       {len(all_results['unreadable'])} files")
    print(f"Empty:            {len(all_results['empty'])} files")
    print(f"Suspicious size:  {len(all_results['suspicious_size'])} files")
    print(f"Missing columns:  {len(all_results['missing_columns'])} files")

    problematic = (
        all_results['unreadable'] +
        all_results['empty'] +
        all_results['suspicious_size']
    )

    if problematic and args.remove_problematic and not args.dry_run:
        print(f"\n[WARNING] About to remove {len(problematic)} problematic files!")
        response = input("Type 'yes' to confirm deletion: ")
        if response.lower() == 'yes':
            removed = 0
            for file_path in problematic:
                try:
                    file_path.unlink()
                    removed += 1
                    if args.verbose:
                        print(f"[removed] {file_path}")
                except Exception as e:
                    print(f"[error] Failed to remove {file_path}: {e}", file=sys.stderr)
            print(f"[done] Removed {removed}/{len(problematic)} files")
        else:
            print("[cancelled] No files removed")
    elif problematic and not args.remove_problematic:
        print(f"\n[info] Use --remove-problematic to delete {len(problematic)} problematic files")
        print("[info] (Will require 'yes' confirmation)")


if __name__ == "__main__":
    main()
