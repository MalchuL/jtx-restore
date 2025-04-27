#!/usr/bin/env python
"""
AI-based frame processors.

This package provides frame processors that use deep learning models
for video frame processing.
"""
from src.core.video.frames.processors.single_frame.ai.ai_processor import AIProcessor
from src.core.video.frames.processors.single_frame.ai.hf_processor import HFAIProcessor
from src.core.video.frames.processors.single_frame.ai.upscale import RealESRGANProcessor   
from src.core.video.frames.processors.single_frame.ai.jpeg_removal import FBCNNProcessor
from src.core.video.frames.processors.single_frame.ai.upscale import APISRProcessor
from src.core.video.frames.processors.single_frame.ai.upscale import SpandrelProcessor

__all__ = [
    'AIProcessor',
    'HFAIProcessor',
    'RealESRGANProcessor',
    'FBCNNProcessor',
    'APISRProcessor',
    'SpandrelProcessor',
] 