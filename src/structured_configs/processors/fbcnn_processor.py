"""
Structured config for FBCNNProcessor.

This module provides a dataclass for FBCNNProcessor configuration to be used with Hydra.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from src.structured_configs.processors.ai_processor import AIProcessorConfig

@dataclass
class FBCNNProcessorConfig(AIProcessorConfig):
    """Hydra config dataclass for FBCNNProcessor."""

    _target_: str = "src.core.video.frames.processors.single_frame.ai.jpeg_removal.fbcnn.FBCNNProcessor"
    
    # Compression factor of the JPEG image (0-100, lower means more compression)
    # If None, auto-detection will be used
    compression_factor: Optional[float] = None
    
    # Override the model_name requirement from parent class since it has a default
    model_name: Optional[str] = None 