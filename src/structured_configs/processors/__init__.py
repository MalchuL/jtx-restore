"""
Structured configs for frame processors.

This module provides dataclass configs for frame processors to be used with Hydra.
"""

from src.structured_configs.processors.color_correction import ColorCorrectionProcessorConfig
from src.structured_configs.processors.denoise import DenoiseProcessorConfig
from src.structured_configs.processors.upscale import UpscaleProcessorConfig
from src.structured_configs.processors.rife_interpolator import PracticalRIFEFrameInterpolatorConfig

# AI processors
from src.structured_configs.processors.ai_processor import AIProcessorConfig
from src.structured_configs.processors.apisr_processor import APISRProcessorConfig
from src.structured_configs.processors.fbcnn_processor import FBCNNProcessorConfig
from src.structured_configs.processors.realesrgan_processor import RealESRGANProcessorConfig

__all__ = [
    # Basic processors
    "ColorCorrectionProcessorConfig",
    "DenoiseProcessorConfig",
    "UpscaleProcessorConfig", 
    "PracticalRIFEFrameInterpolatorConfig",
    
    # AI processors
    "AIProcessorConfig",
    "APISRProcessorConfig",
    "FBCNNProcessorConfig",
    "RealESRGANProcessorConfig"
] 