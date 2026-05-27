from src.core.video.frames.processors.single_frame.enhancers.color.color import ColorCorrectionProcessor
from src.core.video.frames.processors.single_frame.enhancers.denoise.denoising import DenoiseProcessor
from src.core.video.frames.processors.single_frame.enhancers.upscale.scaling import UpscaleProcessor

__all__ = [
    'ColorCorrectionProcessor',
    'DenoiseProcessor',
    'UpscaleProcessor',
]