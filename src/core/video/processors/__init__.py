#!/usr/bin/env python
"""
Frame processor package.

This package provides a framework for processing video frames through
various enhancers and transformations.
"""

# Core processor classes
from src.core.video.processors.frame import ProcessorFrame
from src.core.video.processors.base import FrameProcessor
from src.core.video.processors.batch import BatchFrameProcessor
from src.core.video.processors.parallel import ParallelFrameProcessor
from src.core.video.processors.pipeline import ProcessorPipeline
from src.core.video.processors.manager import BatchProcessingManager

# Import enhancers subpackage
from src.core.video.processors.enhancers import *

__all__ = [
    # Frame data class
    'ProcessorFrame',
    # Core processor classes
    'FrameProcessor',
    'BatchFrameProcessor',
    'ParallelFrameProcessor',
    'ProcessorPipeline',
    'BatchProcessingManager',
] 