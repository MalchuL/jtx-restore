"""
Structured config for UpscaleProcessor.

This module provides a dataclass for UpscaleProcessor configuration to be used with Hydra.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

from omegaconf import MISSING

@dataclass
class UpscaleProcessorConfig:
    """Hydra config dataclass for UpscaleProcessor."""

    _target_: str = "src.core.video.frames.processors.single_frame.enhancers.upscale.scaling.UpscaleProcessor"
    
    # Factor by which to increase frame dimensions.
    # Must be greater than 1.0.
    scale_factor: float = 2.0
    
    # Interpolation method to use.
    # Options: "nearest", "bilinear", "bicubic", "lanczos"
    interpolation: str = "lanczos"
    
    # Number of worker processes for parallel processing.
    num_workers: int = 1 