"""
Structured config for DenoiseProcessor.

This module provides a dataclass for DenoiseProcessor configuration to be used with Hydra.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

@dataclass
class DenoiseProcessorConfig:
    """Hydra config dataclass for DenoiseProcessor."""

    _target_: str = "src.core.video.frames.processors.single_frame.enhancers.denoise.denoising.DenoiseProcessor"
    
    # Overall denoising strength. Range: 0.0 to 20.0.
    # Higher values apply stronger denoising.
    strength: float = 10.0
    
    # Color denoising strength. Range: 0.0 to 20.0.
    # Higher values apply stronger color denoising.
    color_strength: float = 10.0
    
    # Size of template patch for non-local means.
    # Must be odd number.
    template_window_size: int = 7
    
    # Size of search window for non-local means.
    # Must be odd number.
    search_window_size: int = 21
    
    # Whether to use fast non-local means denoising.
    # If False, uses standard non-local means.
    use_fast_nl_means: bool = True
    
    # Number of worker processes for parallel processing.
    num_workers: int = 1 