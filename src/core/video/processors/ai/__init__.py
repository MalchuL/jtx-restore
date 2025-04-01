#!/usr/bin/env python
"""
AI-based frame processors.

This package provides frame processors that use deep learning models
for video frame processing.
"""

from src.core.video.processors.ai_processor import AIProcessor
from src.core.video.processors.ai.hf_processor import HFAIProcessor

__all__ = [
    'AIProcessor',
    'HFAIProcessor',
] 