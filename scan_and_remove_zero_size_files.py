#!/usr/bin/env python3
"""
Scan directories for zero-size files and optionally remove them.

This script recursively searches specified directories for files with zero bytes
and can remove them. Useful for cleaning up corrupted or incomplete data files
(dill, parquet, bbb, etc.) that were created but never written to.

Examples:
  # Dry run - scan all files in directory
  python scan_and_remove_zero_size_files.py /path/to/data --dry-run

  # Remove zero-size parquet files
  python scan_and_remove_zero_size_files.py /path/to/data --file-type "*.parquet"

  # Scan multiple date directories for zero-size dill files
  python scan_and_remove_zero_size_files.py 20251001 20251002 20251003 --file-type "*.dill"

  # Remove all zero-size files (any type)
  python scan_and_remove_zero_size_files.py /path/to/data

  # Verbose output showing each file found
  python scan_and_remove_zero_size_files.py /path/to/data --verbose --dry-run
"""

import argparse
import os
import sys
from pathlib import Path


def find_zero_size_files(directory: Path, file_pattern: str = "*") -> list[Path]:
    """
    Recursively find all zero-size files matching the pattern.

    Args:
        directory: Root directory to search
        file_pattern: Glob pattern for files (e.g., "*.parquet", "*.dill", "*")

    Returns:
        List of Path objects for zero-size files
    """
    zero_size_files = []

    try:
        # Use rglob for recursive search with pattern
        for file_path in directory.rglob(file_pattern):
            if file_path.is_file():
                try:
                    if file_path.stat().st_size == 0:
                        zero_size_files.append(file_path)
                except (OSError, PermissionError) as e:
                    print(f"[warn] Cannot stat {file_path}: {e}", file=sys.stderr)
    except (OSError, PermissionError) as e:
        print(f"[error] Cannot access directory {directory}: {e}", file=sys.stderr)

    return zero_size_files


def scan_directory(
    directory: Path,
    file_pattern: str,
    dry_run: bool,
    verbose: bool
) -> tuple[int, int]:
    """
    Scan a directory for zero-size files and optionally remove them.

    Args:
        directory: Directory to scan
        file_pattern: Glob pattern for file matching
        dry_run: If True, only report files without removing
        verbose: If True, print each file found

    Returns:
        Tuple of (files_found, files_removed)
    """
    if not directory.exists():
        print(f"[skip] Directory does not exist: {directory}")
        return 0, 0

    if not directory.is_dir():
        print(f"[skip] Not a directory: {directory}")
        return 0, 0

    print(f"[scan] {directory}")

    zero_files = find_zero_size_files(directory, file_pattern)

    if verbose:
        for f in zero_files:
            print(f"  [zero-size] {f.relative_to(directory) if f.is_relative_to(directory) else f}")

    removed = 0
    if not dry_run and zero_files:
        for file_path in zero_files:
            try:
                file_path.unlink()
                removed += 1
                if verbose:
                    print(f"  [removed] {file_path.relative_to(directory) if file_path.is_relative_to(directory) else file_path}")
            except Exception as e:
                print(f"[warn] Failed to remove {file_path}: {e}", file=sys.stderr)

    return len(zero_files), removed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan directories for zero-size files and optionally remove them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on single directory
  %(prog)s /mnt/share/beesbook2025/pi/ --dry-run

  # Remove zero-size parquet files from multiple directories
  %(prog)s /mnt/share/beesbook2025/pi/20251001 /mnt/share/beesbook2025/pi/20251002 --file-type "*.parquet"

  # Remove all zero-size dill files with verbose output
  %(prog)s /mnt/share/beesbook2025/results/data_tracked --file-type "*.dill" --verbose

  # Scan multiple date directories (e.g., split across machines)
  %(prog)s 20251001 20251002 20251003 20251004 20251005 --file-type "*.parquet"
        """
    )

    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more directories to scan recursively"
    )

    parser.add_argument(
        "--file-type",
        default="*",
        help='Glob pattern for files to check (default: "*" for all files). Examples: "*.parquet", "*.dill", "*.bbb"'
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan only; do not delete files"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each zero-size file found"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    directories = [Path(d) for d in args.directories]

    total_found = 0
    total_removed = 0

    for directory in directories:
        found, removed = scan_directory(
            directory,
            args.file_type,
            args.dry_run,
            args.verbose
        )

        total_found += found
        total_removed += removed

        if args.dry_run:
            print(f"[dir] found={found} (dry-run)")
        else:
            print(f"[dir] found={found} removed={removed}")

    print()
    if args.dry_run:
        print(f"[total] found={total_found} zero-size files (dry-run)")
        if total_found > 0:
            print(f"[info] Run without --dry-run to remove these files")
    else:
        print(f"[total] found={total_found} removed={total_removed}")


if __name__ == "__main__":
    main()
