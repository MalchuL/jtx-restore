#!/usr/bin/env python
"""
APISR-based frame processor.

This module provides a frame processor that uses APISR for high-quality
image upscaling. APISR (Anime Production Inspired Real-world Anime Super-Resolution)
is designed specifically for anime-style images.
"""

import warnings
import os
import tempfile
from typing import Any, Optional, List, Literal
import numpy as np
from PIL import Image
import rootutils

# Check for PyTorch
try:
    import torch
    from torchvision.utils import save_image

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found. Install with: pip install torch torchvision", RuntimeWarning)

# Check for OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    warnings.warn(
        "OpenCV not found. Install with: pip install opencv-python",
        RuntimeWarning,
    )

# Both dependencies are required
APISR_AVAILABLE = TORCH_AVAILABLE and OPENCV_AVAILABLE

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.processor_info import ProcessorInfo
from src.core.video.frames.processors.single_frame.ai.ai_processor import AIProcessor
from src.core.video.frames.processors.single_frame.ai.upscale.models.apisr_models import (
    super_resolve_img, load_rrdb, load_grl, load_dat, 
)

class APISRProcessor(AIProcessor):
    """Frame processor using APISR for upscaling.
    Based on https://github.com/Kiteretsu77/APISR

    This processor uses APISR to upscale anime-style video frames with high quality,
    leveraging specialized architectures trained for anime content.
    """

    MODELS = {
        "2xRRDB": {
            "url": "https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/2x_APISR_RRDB_GAN_generator.pth",
            "path": "2x_APISR_RRDB_GAN_generator.pth",
            "scale": 2,
            "loader": load_rrdb
        },
        "4xRRDB": {
            "url": "https://github.com/Kiteretsu77/APISR/releases/download/v0.2.0/4x_APISR_RRDB_GAN_generator.pth",
            "path": "4x_APISR_RRDB_GAN_generator.pth",
            "scale": 4,
            "loader": load_rrdb
        },
        "4xGRL": {
            "url": "https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/4x_APISR_GRL_GAN_generator.pth",
            "path": "4x_APISR_GRL_GAN_generator.pth",
            "scale": 4,
            "loader": load_grl
        },
        "4xDAT": {
            "url": "https://github.com/Kiteretsu77/APISR/releases/download/v0.3.0/4x_APISR_DAT_GAN_generator.pth",
            "path": "4x_APISR_DAT_GAN_generator.pth",
            "scale": 4,
            "loader": load_dat
        }
    }
    WEIGHTS_PATH = "weights/aspir"

    def __init__(
        self,
        model_name: Literal["2xRRDB", "4xRRDB", "4xGRL", "4xDAT"] = "2xRRDB",
        device: Optional[str] = None,
        use_custom_model: bool = False,
        custom_model_path: Optional[str] = None,
        downsample_threshold: int = 720,
        crop_for_4x: bool = True,
    ):
        """Initialize APISR processor.

        Args:
            model_type: Type of APISR model to use ("2xRRDB", "4xRRDB", "4xGRL", "4xDAT")
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            use_custom_model: Whether to use a custom model instead of the pretrained ones
            custom_model_path: Path to a custom model checkpoint (used only if use_custom_model is True)
            downsample_threshold: Images with height or width above this value will be downsampled
            crop_for_4x: Whether to crop images to ensure dimensions are divisible by the scale factor
        Raises:
            RuntimeError: If APISR dependencies are not installed
        """
        if not APISR_AVAILABLE:
            missing_deps = []
            if not TORCH_AVAILABLE:
                missing_deps.append("torch torchvision")
            if not OPENCV_AVAILABLE:
                missing_deps.append("opencv-python")
            raise RuntimeError(
                f"Missing required dependencies: {', '.join(missing_deps)}. "
                "Install with: pip install " + " ".join(missing_deps)
            )

        if model_name not in self.MODELS:
            raise ValueError(f"Unsupported model type: {model_name}. "
                           f"Choose from: {', '.join(self.MODELS.keys())}")

        self.use_custom_model = use_custom_model
        self.custom_model_path = custom_model_path
        self.downsample_threshold = downsample_threshold
        self.crop_for_4x = crop_for_4x
        
        # Extract scale from model type
        self.scale = self.MODELS[model_name]["scale"]
            
        # Set default device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Store model reference
        self.generator = None
        self.weight_dtype = torch.float32
                    
        super().__init__(
            model_name=model_name, device=device, batch_size=1
        )

    def _load_model(self) -> None:
        """Load the APISR model.

        This method loads the appropriate APISR model type.
        """
        # Auto-download the model if needed
        if not self.use_custom_model:
            weights_path = self._download_model()
        else:
            if not self.custom_model_path or not os.path.exists(self.custom_model_path):
                raise ValueError(f"Custom model path not valid: {self.custom_model_path}")
            weights_path = self.custom_model_path
            
        try:
            # Load the model according to the model type
            model_info = self.MODELS[self.model_name]
            loader_func = model_info["loader"]
            self.generator = loader_func(weights_path, scale=self.scale)
                
            # Move the model to the specified device and set data type
            self.generator = self.generator.to(device=self.device, dtype=self.weight_dtype)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load APISR model: {str(e)}")
            
    def _download_model(self) -> str:
        """Download the pretrained model if it doesn't exist.
        
        Returns:
            Path to the downloaded model
        """
        model_info = self.MODELS[self.model_name]
        path = rootutils.find_root(search_from=__file__, indicator=".project-root")
        weights_path = os.path.join(path, self.WEIGHTS_PATH)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path, exist_ok=True)
        model_path = os.path.join(weights_path, model_info["path"])
        
        if not os.path.exists(model_path):
            print(f"Downloading APISR model {self.model_name}...")
            
            import urllib.request
            try:
                urllib.request.urlretrieve(model_info["url"], model_path)
                print(f"Model downloaded successfully to {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {str(e)}")
        
        return model_path

    def _preprocess(self, frame: ProcessedFrame) -> Any:
        """Preprocess a frame for APISR input.

        Args:
            frame: Input frame to preprocess

        Returns:
            Preprocessed data ready for model input (numpy array)
        """
        # Get the numpy array from the processed frame
        if frame.data.ndim == 2:  # Grayscale
            frame_data = np.stack([frame.data] * 3, axis=-1)
        else:
            frame_data = frame.data
            
        return frame_data

    def _postprocess(self, model_output: Any) -> np.ndarray:
        """Postprocess APISR output into a frame.

        Args:
            model_output: Raw model output to postprocess (torch tensor)

        Returns:
            Processed frame data as numpy array
        """
        # Convert tensor to numpy array if it's a tensor
        if isinstance(model_output, torch.Tensor):
            result = model_output.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        else:
            result = model_output
            
        return result

    def _infer_model(self, inputs: List[Any]) -> List[Any]:
        """Run APISR inference on a batch of inputs.

        Args:
            inputs: List of preprocessed inputs (numpy arrays in RGB format)

        Returns:
            List of output tensors as numpy arrays
        """
        outputs = []
        for img_np in inputs:
            # Run the actual inference
            output = super_resolve_img(
                self.generator, 
                img_np,
                output_path=None,
                weight_dtype=self.weight_dtype,
                downsample_threshold=self.downsample_threshold,
                crop_for_4x=self.crop_for_4x
            )
            
            # Convert to numpy and prepare output
            outputs.append(output)
            
        return outputs

    def update_processor_info(self, processor_info: ProcessorInfo) -> ProcessorInfo:
        """Update processor info with scaling information.

        Args:
            processor_info: Current processor info

        Returns:
            Updated processor info
        """
        return processor_info.set_frame_width_scale(self.scale).set_frame_height_scale(self.scale) 