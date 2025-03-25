#!/usr/bin/env python
"""
Frame processor package.

This package provides a framework for processing video frames through
various enhancers and transformations.
"""

# Core processor classes
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.base import FrameProcessor

# Import enhancers subpackage
from src.core.video.processors.enhancers import ColorCorrectionProcessor, DenoiseProcessor, UpscaleProcessor

__all__ = [
    # Frame data class
    'ProcessedFrame',
    # Core processor classes
    'FrameProcessor',
    # Enhancers
    'ColorCorrectionProcessor',
    'DenoiseProcessor',
    'UpscaleProcessor',
] 