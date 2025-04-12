#!/usr/bin/env python
"""
AI-based frame processors.

This package provides frame processors that use deep learning models
for video frame processing.
"""
from src.core.video.processors.single_frame.ai.ai_processor import AIProcessor
from src.core.video.processors.single_frame.ai.hf_processor import HFAIProcessor
from src.core.video.processors.single_frame.ai.upscale import RealESRGANProcessor
__all__ = [
    'AIProcessor',
    'HFAIProcessor',
    'RealESRGANProcessor',
] 