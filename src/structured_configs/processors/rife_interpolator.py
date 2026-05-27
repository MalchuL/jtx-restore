"""
Structured config for PracticalRIFEFrameInterpolator425.

This module provides a dataclass for PracticalRIFEFrameInterpolator425 configuration to be used with Hydra.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

@dataclass
class PracticalRIFEFrameInterpolatorConfig:
    """Hydra config dataclass for PracticalRIFEFrameInterpolator425."""

    _target_: str = "src.core.video.frames.processors.frame_interpolation.ai.rife.practical_rife_interpolator.PracticalRIFEFrameInterpolator425"
    
    # The frame rate increase factor (e.g., 2 doubles the frame rate).
    # Must be a positive integer.
    factor: int = 2
    
    # Path to the pre-trained RIFE model directory.
    # If None, uses the default path.
    model_path: Optional[str] = None
    
    # Device to run the model on ('cuda' or 'cpu').
    # Default uses CUDA if available, otherwise CPU.
    device: str = "cuda"
    
    # Scale factor for the model (1.0 for original resolution).
    scale: float = 1.0 