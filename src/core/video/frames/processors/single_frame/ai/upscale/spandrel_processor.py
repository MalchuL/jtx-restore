#!/usr/bin/env python
"""
Spandrel-based frame processor.

This module provides a frame processor that uses Spandrel for loading and running
various upscaling models like RealESRGAN, ESRGAN, SwinIR, etc. Spandrel provides
a unified interface for working with different AI upscaling architectures.
"""

import warnings
import os
from typing import Any, Optional, List, Literal, Dict, Union, Tuple
import numpy as np
from PIL import Image
import rootutils

# Check for PyTorch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found. Install with: pip install torch", RuntimeWarning)

# Check for Spandrel
try:
    import spandrel
    from spandrel import ModelLoader, ImageModelDescriptor

    SPANDREL_AVAILABLE = True
except ImportError:
    SPANDREL_AVAILABLE = False
    warnings.warn(
        "Spandrel not found. Install with: pip install spandrel",
        RuntimeWarning,
    )

# Both dependencies are required
DEPENDENCIES_AVAILABLE = TORCH_AVAILABLE and SPANDREL_AVAILABLE

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.processor_info import ProcessorInfo
from src.core.video.frames.processors.single_frame.ai.ai_processor import AIProcessor
from src.core.video.frames.utils.padder import Padder


class SpandrelProcessor(AIProcessor):
    """Frame processor using Spandrel for upscaling.
    Can be used to load and run various AI upscaling models, from https://openmodeldb.info/ hub
    
    providing a unified interface for different model architectures like
    Real-ESRGAN, ESRGAN, SwinIR, etc.
    """

    # Common model types supported by Spandrel
    MODEL_TYPES = [
        "realesrgan", "esrgan", "swinir", "restormer",
        "hat", "dat", "omnisr", "span", "srvgg", "rrdb"
    ]

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        batch_size: int = 1,
        model_type: Optional[str] = None,
    ):
        """Initialize Spandrel processor.

        Args:
            model_path: Path to the model file
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            batch_size: Number of frames to process in each batch
            model_type: Explicitly specify model type (auto-detected if None)
            
        Raises:
            RuntimeError: If Spandrel dependencies are not installed
            ValueError: If the model path is invalid or the model type is not supported
        """
        if not DEPENDENCIES_AVAILABLE:
            missing_deps = []
            if not TORCH_AVAILABLE:
                missing_deps.append("torch")
            if not SPANDREL_AVAILABLE:
                missing_deps.append("spandrel")
            raise RuntimeError(
                f"Missing required dependencies: {', '.join(missing_deps)}. "
                "Install with: pip install " + " ".join(missing_deps)
            )

        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.model_path = os.path.join(rootutils.find_root(search_from=__file__, indicator=".project-root"), self.model_path)
        
        self.model_type = model_type
        
        # Set default device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(device)
        
        # These will be set after loading the model
        self.model = None
        self.model_descriptor = None
        self.scale = None
        self.padder = None
        
        super().__init__(
            model_name=os.path.basename(model_path), 
            device=device, 
            batch_size=batch_size
        )

    def _load_model(self) -> None:
        """Load the upscaling model using Spandrel.

        This method loads the model and initializes all required components.
        """
        # Create the model loader
        loader = ModelLoader()
        
        try:
            # Load and parse the model
            self.model_descriptor = loader.load_from_file(self.model_path)
            
            # Check if we need to enforce a specific model type
            if self.model_type and self.model_descriptor.architecture != self.model_type:
                warnings.warn(
                    f"Model architecture is {self.model_descriptor.architecture} "
                    f"but requested {self.model_type}. Using as requested.",
                    RuntimeWarning,
                )
            
            # Extract the model and its properties
            self.model = self.model_descriptor.model
            self.model.to(device=self.torch_device)
            self.model.eval()
            
            # Get the scale factor
            self.scale = self.model_descriptor.scale
            
            # Initialize the padder for making input dimensions divisible by a specific value
            # Real-ESRGAN usually requires input dimensions to be divisible by 32
            self.padder = Padder(mod_pad=32, pad_size=15, scale_factor=self.scale)
            
            print(f"Loaded {self.model_descriptor.architecture} model with scale {self.scale}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _preprocess(self, frame: ProcessedFrame) -> Any:
        """Preprocess a frame for model input.

        Args:
            frame: Input frame to preprocess

        Returns:
            Preprocessed data ready for model input
        """
        # Get the numpy array from the frame
        if frame.data.ndim == 2:  # Grayscale
            frame_data = np.stack([frame.data] * 3, axis=-1)
        else:
            frame_data = frame.data
            
        # Apply padding using Padder
        padded_data = self.padder.pad_image(frame_data)
        
        # Convert to tensor format expected by the model (RGB)
        # Most models expect input in the range [0, 1]
        input_tensor = torch.from_numpy(padded_data).float().div(255.0)
        
        # Change from HWC to NCHW format
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return input_tensor

    def _postprocess(self, model_output: Any) -> np.ndarray:
        """Postprocess model output into a frame.

        Args:
            model_output: Raw model output to postprocess

        Returns:
            Processed frame data as numpy array
        """
        # Convert output tensor to numpy array
        # Model output is usually in NCHW format with values in [0, 1]
        if isinstance(model_output, torch.Tensor):
            output_np = model_output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
            output_np = output_np.astype(np.uint8)
        else:
            output_np = model_output
            
        # Remove padding
        result = self.padder.unpad_image(output_np)
        
        return result

    def _infer_model(self, inputs: List[Any]) -> List[Any]:
        """Run model inference on a batch of inputs.

        Args:
            inputs: List of preprocessed inputs (torch tensors)

        Returns:
            List of model outputs
        """
        outputs = []
        
        # Process in a batch if there are multiple inputs
        if len(inputs) > 1:
            try:
                # Stack inputs into a single batch tensor
                batch_size = len(inputs)
                if batch_size != self.batch_size:
                    print(f"Warning: Actual batch size ({batch_size}) differs from configured batch size ({self.batch_size})")
                
                batch_tensor = torch.cat(inputs, dim=0).to(self.torch_device)
                
                # Process the batch
                with torch.no_grad():
                    batch_output = self.model(batch_tensor)
                
                # Split the output back into individual results
                for i in range(batch_size):
                    outputs.append(batch_output[i:i+1])
                    
                print(f"Processed batch of {batch_size} frames")
            except RuntimeError as e:
                # If we encounter an error (like CUDA OOM), fall back to individual processing
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory when processing batch. Falling back to individual processing.")
                else:
                    print(f"Error during batch processing: {str(e)}. Falling back to individual processing.")
                
                outputs = []
                for input_tensor in inputs:
                    # Process one by one
                    with torch.no_grad():
                        output = self.model(input_tensor.to(self.torch_device))
                    outputs.append(output)
        else:
            # Process a single input
            input_tensor = inputs[0].to(self.torch_device)
            with torch.no_grad():
                output = self.model(input_tensor)
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