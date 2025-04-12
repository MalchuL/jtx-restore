#!/usr/bin/env python
"""
Frame processor package.

This package provides a framework for processing video frames through
various enhancers and transformations.
"""

# Core processor classes
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.processor import FrameProcessor

# Import AI processors
from src.core.video.processors.single_frame.ai import AIProcessor, HFAIProcessor

# Import enhancers subpackage
from src.core.video.processors.single_frame.enhancers import ColorCorrectionProcessor, DenoiseProcessor, UpscaleProcessor
from src.core.video.processors.single_frame.ai import RealESRGANProcessor

__all__ = [
    # Frame data class
    'ProcessedFrame',
    # Core processor classes
    'FrameProcessor',
    # AI processors
    'AIProcessor',
    'HFAIProcessor',
    'RealESRGANProcessor',
    # Enhancers
    'ColorCorrectionProcessor',
    'DenoiseProcessor',
    'UpscaleProcessor',
] 