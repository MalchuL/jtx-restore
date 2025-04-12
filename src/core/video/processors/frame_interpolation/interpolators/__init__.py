"""
Basic frame interpolation methods based on conventional computer vision techniques.

This package contains implementations of frame interpolation methods that use
traditional computer vision techniques rather than deep learning approaches.
"""

from src.core.video.processors.frame_interpolation.interpolators.opencv_interpolator import OpenCVFrameInterpolator

__all__ = ["OpenCVFrameInterpolator"] 