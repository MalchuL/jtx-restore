#!/usr/bin/env python
"""
Frame processor package.

This package provides a framework for processing video frames through
various enhancers and transformations.
"""

# Core processor classes
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.processor import FrameProcessor
from src.core.video.processors.pipeline import ProcessorPipeline

# Import enhancers subpackage
from src.core.video.processors.enhancers import ColorCorrectionProcessor, DenoiseProcessor, UpscaleProcessor

__all__ = [
    # Frame data class
    'ProcessedFrame',
    # Core processor classes
    'FrameProcessor',
    'ProcessorPipeline',
    # Enhancers
    'ColorCorrectionProcessor',
    'DenoiseProcessor',
    'UpscaleProcessor',
] 