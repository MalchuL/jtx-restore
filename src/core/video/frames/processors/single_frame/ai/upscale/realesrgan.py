#!/usr/bin/env python
"""
RealESRGAN-based frame processor.

This module provides a frame processor that uses RealESRGAN for high-quality
image upscaling. RealESRGAN is particularly effective for anime and general
image upscaling.
"""

import warnings
from typing import Any, Optional, List
import numpy as np
from PIL import Image

from src.core.video.frames.processors.single_frame.ai.upscale.models.whole_image_realesrgan import WholeImageRealESRGAN



# Check for PyTorch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found. Install with: pip install torch", RuntimeWarning)

# Check for basicsr
try:
    from RealESRGAN import RealESRGAN

    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    warnings.warn(
        "RealESRGAN not found. Install with: pip install git+https://github.com/doevent/Real-ESRGAN.git",
        RuntimeWarning,
    )

# Both dependencies are required
REALESRGAN_AVAILABLE = TORCH_AVAILABLE and REALESRGAN_AVAILABLE

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.processor_info import ProcessorInfo
from src.core.video.frames.processors.single_frame.ai.ai_processor import AIProcessor


class RealESRGANProcessor(AIProcessor):
    """Frame processor using RealESRGAN for upscaling.
    Based on https://huggingface.co/ai-forever/Real-ESRGAN

    This processor uses RealESRGAN to upscale video frames with high quality.
    It supports both anime and general image upscaling models.
    """

    def __init__(
        self,
        scale: int = 2,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_whole_image: bool = True,
        batch_size: int = 1,
    ):
        """Initialize RealESRGAN processor.

        Args:
            scale: Scale factor for the upscaling
            model_name: Name of the RealESRGAN model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            batch_size: Number of frames to process in each batch, but batch means batch size for patching
            use_whole_image: Whether to use the whole image upscaling model
        Raises:
            RuntimeError: If RealESRGAN dependencies are not installed
        """
        if not REALESRGAN_AVAILABLE:
            missing_deps = []
            if not TORCH_AVAILABLE:
                missing_deps.append("torch")
            if not REALESRGAN_AVAILABLE:
                missing_deps.append("git+https://github.com/doevent/Real-ESRGAN.git")
            raise RuntimeError(
                f"Missing required dependencies: {', '.join(missing_deps)}. "
                "Install with: pip install " + " ".join(missing_deps)
            )

        if model_name is None:
            model_name = f"RealESRGAN_x{scale}"
        if model_name not in ["RealESRGAN_x2", "RealESRGAN_x4", "RealESRGAN_x8"]:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                "Supported models: RealESRGAN_x2, RealESRGAN_x4, RealESRGAN_x8"
            )
        self.use_whole_image = use_whole_image
        self.upsampler = None
        self.scale = scale
        # Set default device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        self._num_patches = batch_size  # Number of patches to process in each batch
        bs = batch_size if self.use_whole_image else 1
        super().__init__(
            model_name=model_name, device=device, batch_size=bs
        )

    def _load_model(self) -> None:
        """Load the RealESRGAN model and upscaler.

        This method loads the model and initializes the RealESRGAN upscaler.
        """
        # Initialize model
        if self.use_whole_image:
            model = WholeImageRealESRGAN(self.device, scale=self.scale)
        else:
            model = RealESRGAN(self.device, scale=self.scale)
        model.load_weights(f"weights/{self.model_name}.pth", download=True)

        self.upsampler = model

    def _preprocess(self, frame: ProcessedFrame) -> Any:
        """Preprocess a frame for RealESRGAN input.

        Args:
            frame: Input frame to preprocess

        Returns:
            Preprocessed data ready for model input
        """
        # RealESRGAN expects numpy arrays in BGR format
        if frame.data.ndim == 2:  # Grayscale
            frame_data = np.stack([frame.data] * 3, axis=-1)
        else:
            frame_data = frame.data

        return frame_data

    def _postprocess(self, model_output: Any) -> np.ndarray:
        """Postprocess RealESRGAN output into a frame.

        Args:
            model_output: Raw model output to postprocess

        Returns:
            Processed frame data as numpy array
        """
        # RealESRGAN outputs BGR format
        assert isinstance(model_output, Image.Image), f"Model output is not an instance of Image.Image, but {type(model_output)}"
        out = np.array(model_output)

        return out

    def _infer_model(self, inputs: List[Any]) -> List[Any]:
        """Run RealESRGAN inference on a batch of inputs.

        Args:
            inputs: List of preprocessed inputs

        Returns:
            List of model outputs
        """
        print(len(inputs))
        if self.use_whole_image:
            outputs = self.upsampler.predict(inputs)
        else:
            outputs = []
            for img in inputs:
                # RealESRGAN processes one image at a time
                output = self.upsampler.predict(img, batch_size=self._num_patches)
                outputs.append(output)
        assert len(outputs) == len(inputs), f"Number of outputs is not equal to number of inputs, got {len(outputs)} and {len(inputs)}"
        return outputs

    def update_processor_info(self, processor_info: ProcessorInfo) -> ProcessorInfo:
        return processor_info.set_frame_width_scale(self.scale).set_frame_height_scale(self.scale)
