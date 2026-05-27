"""
Structured config for RealESRGANProcessor.

This module provides a dataclass for RealESRGANProcessor configuration to be used with Hydra.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from src.structured_configs.processors.ai_processor import AIProcessorConfig

@dataclass
class RealESRGANProcessorConfig(AIProcessorConfig):
    """Hydra config dataclass for RealESRGANProcessor."""

    _target_: str = "src.core.video.frames.processors.single_frame.ai.upscale.realesrgan.RealESRGANProcessor"
    
    # Scale factor for the upscaling (2, 4, or 8)
    scale: int = 2
    
    # Override the model_name requirement from parent class since it has a default
    model_name: Optional[str] = None 
    
    # Whether to use the whole image upscaling model
    use_whole_image: bool = True