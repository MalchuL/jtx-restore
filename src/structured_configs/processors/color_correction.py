"""
Structured config for ColorCorrectionProcessor.

This module provides a dataclass for ColorCorrectionProcessor configuration to be used with Hydra.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

@dataclass
class ColorCorrectionProcessorConfig:
    """Hydra config dataclass for ColorCorrectionProcessor."""

    _target_: str = "src.core.video.frames.processors.single_frame.enhancers.color.color.ColorCorrectionProcessor"
    
    # Brightness adjustment. Range: -1.0 to 1.0.
    # Negative values darken, positive values brighten.
    brightness: float = 0.0
    
    # Contrast adjustment. Range: 0.0 to 3.0.
    # Values below 1.0 reduce contrast, above 1.0 increase it.
    contrast: float = 1.0
    
    # Saturation adjustment. Range: 0.0 to 3.0.
    # Values below 1.0 reduce saturation, above 1.0 increase it.
    saturation: float = 1.0
    
    # Whether to apply automatic white balance correction.
    white_balance: bool = False
    
    # Gamma correction value. Range: 0.1 to 3.0.
    # Values below 1.0 brighten shadows, above 1.0 darken midtones.
    gamma: float = 1.0
    
    # Whether to apply automatic exposure correction.
    auto_exposure: bool = False
    
    # Number of worker processes for parallel processing.
    num_workers: int = 1 