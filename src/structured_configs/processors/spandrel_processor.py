"""
Structured configuration for Spandrel processor.

This module defines the configuration structure for the Spandrel processor,
which provides a unified interface for various AI upscaling models.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SpandrelProcessorConfig:
    """Configuration for Spandrel processor.
    
    This processor uses Spandrel to load and run various AI upscaling models like
    Real-ESRGAN, ESRGAN, SwinIR, etc., with a unified interface.
    """
    
    _target_: str = "src.core.video.frames.processors.single_frame.ai.upscale.spandrel_processor.SpandrelProcessor"
    
    # Path to the model file
    model_path: str = ""
    
    # Processing device (cuda, cpu, or auto)
    device: Optional[str] = None
    
    # Batch size for processing - number of frames to process simultaneously
    # Higher values can improve throughput but require more GPU memory
    batch_size: int = 1
    
    # Explicitly specify model type (auto-detected if None)
    model_type: Optional[str] = None
    


