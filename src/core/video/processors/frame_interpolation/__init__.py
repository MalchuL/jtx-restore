"""
Frame interpolation for increasing video frame rate.

This package provides various algorithms for video frame interpolation,
from basic methods to advanced deep learning approaches.
"""

# Import the core classes and interfaces
from src.core.video.processors.frame_interpolation.frame_interpolator import FrameInterpolator
from src.core.video.processors.frame_interpolation.interpolated_frame import InterpolatedFrame

# Import basic interpolators
from src.core.video.processors.frame_interpolation.interpolators import OpenCVFrameInterpolator

# Import AI-based interpolators
from src.core.video.processors.frame_interpolation.ai.rife import PracticalRIFEFrameInterpolator425

__all__ = [
    "FrameInterpolator",
    "InterpolatedFrame",
    "OpenCVFrameInterpolator",
    "PracticalRIFEFrameInterpolator425"
]