"""
Coordinate transformation utilities for handling camera rotations.

This module provides a unified system for handling different camera rotation
configurations across datasets (2024 vs 2025, etc.).

Rotation types:
- 'none': No rotation
- 'cw90': 90° clockwise rotation
- 'ccw90': 90° counter-clockwise rotation
- '180': 180° rotation (flip)

The detection pipeline outputs coordinates in the rotated image space.
This module handles:
1. Reversing the rotation to get original image coordinates
2. Converting to bottom-left origin (y-up) for calibration consistency
"""

import numpy as np
import pandas as pd
from typing import Tuple, Literal

RotationType = Literal['none', 'cw90', 'ccw90', '180']


class RotationConfig:
    """
    Configuration for camera rotation and coordinate transformations.

    Parameters
    ----------
    rotation : RotationType
        Type of rotation applied by the camera/pipeline
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

        # Dimensions after rotation
        if rotation in ('cw90', 'ccw90'):
            self.rotated_width = original_height
            self.rotated_height = original_width
        else:
            self.rotated_width = original_width
            self.rotated_height = original_height

    def reverse_rotation_pixels(
        self,
        x_rot: np.ndarray,
        y_rot: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert from rotated image coordinates to original image coordinates.
        Both input and output use top-left origin (standard image coordinates).

        Parameters
        ----------
        x_rot : np.ndarray
            X coordinates in rotated image (origin top-left)
        y_rot : np.ndarray
            Y coordinates in rotated image (origin top-left)

        Returns
        -------
        x_orig : np.ndarray
            X coordinates in original image (origin top-left)
        y_orig : np.ndarray
            Y coordinates in original image (origin top-left)
        """
        if self.rotation == 'none':
            return x_rot, y_rot

        elif self.rotation == 'cw90':
            # For 90° clockwise: point (x,y) -> (h-1-y, x)
            # Reverse: (x_rot, y_rot) -> (y_rot, w_rot-1-x_rot)
            x_orig = y_rot
            y_orig = self.rotated_width - 1 - x_rot
            return x_orig, y_orig

        elif self.rotation == 'ccw90':
            # For 90° counter-clockwise: point (x,y) -> (y, w-1-x)
            # Reverse: (x_rot, y_rot) -> (h_rot-1-y_rot, x_rot)
            x_orig = self.rotated_height - 1 - y_rot
            y_orig = x_rot
            return x_orig, y_orig

        elif self.rotation == '180':
            # For 180°: point (x,y) -> (w-1-x, h-1-y)
            # Reverse is the same operation
            x_orig = self.rotated_width - 1 - x_rot
            y_orig = self.rotated_height - 1 - y_rot
            return x_orig, y_orig

        else:
            raise ValueError(f"Unknown rotation type: {self.rotation}")

    def to_bottom_left_origin(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert from top-left origin to bottom-left origin.
        Input: (x, y) with origin at top-left, y increases downward
        Output: (x, y) with origin at bottom-left, y increases upward

        Parameters
        ----------
        x : np.ndarray
            X coordinates (origin top-left)
        y : np.ndarray
            Y coordinates (origin top-left, y increases downward)

        Returns
        -------
        x_bl : np.ndarray
            X coordinates (unchanged)
        y_bl : np.ndarray
            Y coordinates (origin bottom-left, y increases upward)
        """
        y_bl = self.original_height - 1 - y
        return x, y_bl

    def transform_detections(
        self,
        x_rot: np.ndarray,
        y_rot: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full transformation pipeline for detection coordinates.

        Input: coordinates from detection pipeline (rotated image, top-left origin)
        Output: coordinates for calibration (original image, bottom-left origin)

        This is the main function to use when processing trajectory data.

        Parameters
        ----------
        x_rot : np.ndarray
            X coordinates from detection pipeline
        y_rot : np.ndarray
            Y coordinates from detection pipeline

        Returns
        -------
        x_calib : np.ndarray
            X coordinates in calibration system
        y_calib : np.ndarray
            Y coordinates in calibration system
        """
        # Step 1: Reverse the rotation
        x_orig, y_orig = self.reverse_rotation_pixels(x_rot, y_rot)

        # Step 2: Convert to bottom-left origin
        x_calib, y_calib = self.to_bottom_left_origin(x_orig, y_orig)

        return x_calib, y_calib

    def transform_orientation(self, orientation_rot: np.ndarray) -> np.ndarray:
        """
        Transform orientation angles to match the rotation reversal.

        Parameters
        ----------
        orientation_rot : np.ndarray
            Orientation angles in rotated image (radians)

        Returns
        -------
        orientation_orig : np.ndarray
            Orientation angles in original image (radians)
        """
        if self.rotation == 'none':
            return orientation_rot
        elif self.rotation == 'cw90':
            # Reverse 90° clockwise rotation
            return (orientation_rot + np.pi / 2) % (2 * np.pi)
        elif self.rotation == 'ccw90':
            # Reverse 90° counter-clockwise rotation
            return (orientation_rot - np.pi / 2) % (2 * np.pi)
        elif self.rotation == '180':
            # Reverse 180° rotation
            return (orientation_rot + np.pi) % (2 * np.pi)
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
        Transform annotation coordinates for display on rotated image.

        Annotations are in original image coordinates (top-left origin).
        This transforms them to rotated image coordinates for plotting.

        Parameters
        ----------
        x_annot : float
            X coordinate in original image
        y_annot : float
            Y coordinate in original image

        Returns
        -------
        x_display : float
            X coordinate for display on rotated image
        y_display : float
            Y coordinate for display on rotated image
        """
        if self.rotation == 'none':
            return x_annot, y_annot

        elif self.rotation == 'cw90':
            # 90° clockwise: (x,y) -> (h-1-y, x)
            x_display = self.original_height - 1 - y_annot
            y_display = x_annot
            return x_display, y_display

        elif self.rotation == 'ccw90':
            # 90° counter-clockwise: (x,y) -> (y, w-1-x)
            x_display = y_annot
            y_display = self.original_width - 1 - x_annot
            return x_display, y_display

        elif self.rotation == '180':
            # 180°: (x,y) -> (w-1-x, h-1-y)
            x_display = self.original_width - 1 - x_annot
            y_display = self.original_height - 1 - y_annot
            return x_display, y_display

        else:
            raise ValueError(f"Unknown rotation type: {self.rotation}")

    def get_calibration_transform_matrix(self) -> np.ndarray:
        """
        Get the transformation needed for calibration corner points.

        Returns
        -------
        needs_y_flip : bool
            Whether y coordinates need to be flipped for calibration
        height_for_flip : int
            The height value to use for y-flipping (original_height)
        """
        # Calibration always needs bottom-left origin
        return True, self.original_height


def get_rotation_config(cfg) -> RotationConfig:
    """
    Extract rotation configuration from a config object.

    Parameters
    ----------
    cfg : module or object
        Configuration object/module with rotation settings

    Returns
    -------
    rotation_config : RotationConfig
        Rotation configuration object
    """
    # Get rotation type (default to 'cw90' for backward compatibility with 2025 data)
    rotation = getattr(cfg, 'camera_rotation', 'cw90')

    # Get image dimensions
    # Note: ypixels and xpixels in config refer to ORIGINAL (pre-rotation) dimensions
    original_height = cfg.ypixels
    original_width = cfg.xpixels

    return RotationConfig(
        rotation=rotation,
        original_width=original_width,
        original_height=original_height,
    )
