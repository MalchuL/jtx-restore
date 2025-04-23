"""
Structured configs for hydra configuration.

This module provides structured configurations for various components of the application.
"""

from src.structured_configs.task.task import Pipeline
from src.structured_configs.config_store import CS
from src.structured_configs.processors import (
    # Basic processors
    ColorCorrectionProcessorConfig,
    DenoiseProcessorConfig,
    UpscaleProcessorConfig,
    PracticalRIFEFrameInterpolatorConfig,
    
    # AI processors
    AIProcessorConfig,
    APISRProcessorConfig,
    FBCNNProcessorConfig,
    RealESRGANProcessorConfig
)

def register_structured_configs():
    """Register all structured configs with Hydra's config store."""
    # Register processor configs
    processors_group = "processors"
    
    # Basic processors
    CS.store(group=processors_group, name="color_correction", node=ColorCorrectionProcessorConfig)
    CS.store(group=processors_group, name="denoise", node=DenoiseProcessorConfig)
    CS.store(group=processors_group, name="upscale", node=UpscaleProcessorConfig)
    CS.store(group=processors_group, name="rife_interpolator", node=PracticalRIFEFrameInterpolatorConfig)
    
    # AI processors
    CS.store(group=processors_group, name="ai_processor", node=AIProcessorConfig)
    CS.store(group=processors_group, name="apisr_processor", node=APISRProcessorConfig)
    CS.store(group=processors_group, name="fbcnn_processor", node=FBCNNProcessorConfig)
    CS.store(group=processors_group, name="realesrgan_processor", node=RealESRGANProcessorConfig)
    
    # Register pipeline/task configs
    # task_group = "task"
    # CS.store(group=task_group, name="default_task", node=Pipeline)


__all__ = [
    "register_structured_configs",
    
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
