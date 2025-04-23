#!/usr/bin/env python
"""
Structured configuration for APISR processor.

This module defines the configuration dataclass for the APISR processor,
which is used for upscaling anime-style video frames.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

from src.structured_configs.processors.ai_processor import AIProcessorConfig


@dataclass
class APISRProcessorConfig(AIProcessorConfig):
    """Configuration for the APISR processor.

    This processor uses APISR models to upscale anime-style video frames with high quality,
    leveraging specialized architectures trained for anime content.
    """

    # Override processor_type
    _target_: str = "src.core.video.frames.processors.single_frame.ai.upscale.apisr.APISRProcessor"
    
    # APISR specific parameters
    model_name: Literal["2xRRDB", "4xRRDB", "4xGRL", "4xDAT"] = "2xRRDB"
    use_custom_model: bool = False
    custom_model_path: Optional[str] = None
    downsample_threshold: int = 720
    crop_for_4x: bool = True

