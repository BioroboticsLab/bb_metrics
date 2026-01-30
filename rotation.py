"""
Coordinate transformation utilities for handling camera rotations.

This module provides a unified system for handling different camera rotation
configurations across datasets (2024 vs 2025, etc.).

## Coordinate Frames

- **Original coords**: Raw camera image coordinates (e.g., 5312×4608 for 2025)
- **Analysis coords**: Rotated coordinates used for all analysis and visualization

The config parameter `camera_rotation` specifies the rotation TO BE APPLIED
to transform from original coords to analysis coords.

Rotation types:
- 'none': No rotation
- 'cw90': 90° clockwise rotation
- 'ccw90': 90° counter-clockwise rotation
- '180': 180° rotation

Example for cw90:
    Original (5312×4608) → apply cw90 → Analysis (4608×5312)

All coordinates use top-left origin (standard image convention):
- (0, 0) at top-left corner
- x increases rightward
- y increases downward
"""

import numpy as np
from typing import Tuple, Literal

RotationType = Literal['none', 'cw90', 'ccw90', '180']


class RotationConfig:
    """
    Configuration for camera rotation and coordinate transformations.

    Parameters
    ----------
    rotation : RotationType
        Type of rotation to apply (original → analysis)
    original_width : int
        Width of original (pre-rotation) image
    original_height : int
        Height of original (pre-rotation) image
    """

    def __init__(
        self,
        rotation: RotationType,
        original_width: int,
        original_height: int,
    ):
        self.rotation = rotation
        self.original_width = original_width
        self.original_height = original_height

        # Dimensions after rotation (analysis frame)
        if rotation in ('cw90', 'ccw90'):
            self.analysis_width = original_height
            self.analysis_height = original_width
        else:
            self.analysis_width = original_width
            self.analysis_height = original_height

    def apply_rotation_pixels(
        self,
        x_orig: np.ndarray,
        y_orig: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rotation to transform original image coords to analysis coords.
        Both input and output use top-left origin (standard image coordinates).

        Parameters
        ----------
        x_orig : np.ndarray
            X coordinates in original image (origin top-left)
        y_orig : np.ndarray
            Y coordinates in original image (origin top-left)

        Returns
        -------
        x_analysis : np.ndarray
            X coordinates in analysis image (origin top-left)
        y_analysis : np.ndarray
            Y coordinates in analysis image (origin top-left)
        """
        if self.rotation == 'none':
            return x_orig, y_orig

        elif self.rotation == 'cw90':
            # 90° clockwise: (x, y) → (h - 1 - y, x)
            # where h = original_height
            x_analysis = self.original_height - 1 - y_orig
            y_analysis = x_orig
            return x_analysis, y_analysis

        elif self.rotation == 'ccw90':
            # 90° counter-clockwise: (x, y) → (y, w - 1 - x)
            # where w = original_width
            x_analysis = y_orig
            y_analysis = self.original_width - 1 - x_orig
            return x_analysis, y_analysis

        elif self.rotation == '180':
            # 180°: (x, y) → (w - 1 - x, h - 1 - y)
            x_analysis = self.original_width - 1 - x_orig
            y_analysis = self.original_height - 1 - y_orig
            return x_analysis, y_analysis

        else:
            raise ValueError(f"Unknown rotation type: {self.rotation}")

    def transform_detections(
        self,
        x_orig: np.ndarray,
        y_orig: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full transformation pipeline for detection coordinates.

        Input: coordinates from detection files (original image space)
        Output: coordinates in analysis image space (for visualization/metrics)

        This is the main function to use when processing trajectory data.
        Coordinates remain in top-left origin (standard image convention).

        Parameters
        ----------
        x_orig : np.ndarray
            X coordinates from detection files (original image space)
        y_orig : np.ndarray
            Y coordinates from detection files (original image space)

        Returns
        -------
        x_pixels : np.ndarray
            X coordinates in analysis image (top-left origin)
        y_pixels : np.ndarray
            Y coordinates in analysis image (top-left origin)
        """
        return self.apply_rotation_pixels(x_orig, y_orig)

    def transform_orientation(self, orientation_orig: np.ndarray) -> np.ndarray:
        """
        Transform orientation angles to match the rotation.

        Parameters
        ----------
        orientation_orig : np.ndarray
            Orientation angles in original image (radians)

        Returns
        -------
        orientation_analysis : np.ndarray
            Orientation angles in analysis image (radians)
        """
        if self.rotation == 'none':
            return orientation_orig
        elif self.rotation == 'cw90':
            # Apply 90° clockwise rotation to orientation
            return (orientation_orig - np.pi / 2) % (2 * np.pi)
        elif self.rotation == 'ccw90':
            # Apply 90° counter-clockwise rotation to orientation
            return (orientation_orig + np.pi / 2) % (2 * np.pi)
        elif self.rotation == '180':
            # Apply 180° rotation to orientation
            return (orientation_orig + np.pi) % (2 * np.pi)
        else:
            raise ValueError(f"Unknown rotation type: {self.rotation}")

    def numpy_rot90_k(self) -> int:
        """
        Get the k parameter for np.rot90 to apply the rotation for display.

        Returns
        -------
        k : int
            Parameter for np.rot90(img, k=k)
            Positive k rotates counter-clockwise
        """
        if self.rotation == 'none':
            return 0
        elif self.rotation == 'cw90':
            return -1  # or 3, equivalent for np.rot90
        elif self.rotation == 'ccw90':
            return 1
        elif self.rotation == '180':
            return 2
        else:
            raise ValueError(f"Unknown rotation type: {self.rotation}")

    def transform_annotation_coords(
        self,
        x_annot: float,
        y_annot: float,
    ) -> Tuple[float, float]:
        """
        Transform annotation coordinates from original to analysis frame.

        NOTE: This function may not be needed if annotations are already
        created in the analysis frame (e.g., from CVAT on rotated images).

        Parameters
        ----------
        x_annot : float
            X coordinate in original image
        y_annot : float
            Y coordinate in original image

        Returns
        -------
        x_display : float
            X coordinate in analysis image
        y_display : float
            Y coordinate in analysis image
        """
        # Same as apply_rotation_pixels but for single values
        x_arr, y_arr = self.apply_rotation_pixels(
            np.array([x_annot]),
            np.array([y_annot])
        )
        return float(x_arr[0]), float(y_arr[0])


def get_rotation_config(cfg) -> RotationConfig:
    """
    Extract rotation configuration from a config object.

    Parameters
    ----------
    cfg : module or object
        Configuration object/module with rotation settings.
        Expected attributes:
        - camera_rotation: rotation type ('none', 'cw90', 'ccw90', '180')
        - xpixels: width in analysis (rotated) frame
        - ypixels: height in analysis (rotated) frame

    Returns
    -------
    rotation_config : RotationConfig
        Rotation configuration object
    """
    # Get rotation type (default to 'cw90' for backward compatibility with 2025 data)
    rotation = getattr(cfg, 'camera_rotation', 'cw90')

    # Get image dimensions
    # Note: cfg.xpixels and cfg.ypixels are in ANALYSIS (rotated) frame
    # We need to compute the ORIGINAL dimensions
    if rotation in ('cw90', 'ccw90'):
        # For 90° rotations, analysis dimensions are swapped from original
        # cfg.xpixels = analysis_width = original_height
        # cfg.ypixels = analysis_height = original_width
        original_width = cfg.ypixels
        original_height = cfg.xpixels
    else:
        # For 'none' or '180', dimensions don't swap
        original_width = cfg.xpixels
        original_height = cfg.ypixels

    return RotationConfig(
        rotation=rotation,
        original_width=original_width,
        original_height=original_height,
    )
