"""AI-based upscaling processors.

This package provides processors for AI-based image upscaling,
including RealESRGAN and other upscaling models.
"""

from src.core.video.processors.ai.upscale.realesrgan import RealESRGANProcessor

__all__ = ['RealESRGANProcessor'] 