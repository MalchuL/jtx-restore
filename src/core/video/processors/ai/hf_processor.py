#!/usr/bin/env python
"""
HuggingFace-based frame processor.

This module provides a frame processor that uses HuggingFace's transformers
library to process video frames with deep learning models.
"""

import warnings
from typing import Any, Optional, List
import numpy as np

# Check for PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not found. Install with: pip install torch",
        RuntimeWarning
    )

# Check for Transformers
try:
    from transformers import AutoImageProcessor, AutoModelForImageProcessing
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "Transformers not found. Install with: pip install transformers",
        RuntimeWarning
    )

# Both dependencies are required
HF_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE

from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.ai_processor import AIProcessor


class HFAIProcessor(AIProcessor):
    """Frame processor using HuggingFace models.
    
    This processor uses HuggingFace's transformers library to load and run
    image processing models. It handles model loading, preprocessing, and
    postprocessing automatically.
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 1,
        fill_batch: bool = False
    ):
        """Initialize HuggingFace processor.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            batch_size: Number of frames to process in each batch
            fill_batch: Whether to pad incomplete batches to batch_size
            
        Raises:
            RuntimeError: If HuggingFace dependencies are not installed
        """
        if not HF_AVAILABLE:
            missing_deps = []
            if not TORCH_AVAILABLE:
                missing_deps.append("torch")
            if not TRANSFORMERS_AVAILABLE:
                missing_deps.append("transformers")
            raise RuntimeError(
                f"Missing required dependencies: {', '.join(missing_deps)}. "
                "Install with: pip install " + " ".join(missing_deps)
            )
            
        # Set default device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        super().__init__(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            fill_batch=fill_batch
        )
    
    def _load_model(self) -> None:
        """Load the HuggingFace model and processor.
        
        This method loads the model and processor using the HuggingFace
        transformers library.
        """
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageProcessing.from_pretrained(self.model_name)
        self.model.to(self.device)

    def _preprocess(self, frame: ProcessedFrame) -> Any:
        """Preprocess a frame for HuggingFace model input.
        
        Args:
            frame: Input frame to preprocess
            
        Returns:
            Preprocessed data ready for model input
        """
        return self.processor(
            images=frame.data,
            return_tensors="pt"
        ).to(self.device)

    def _postprocess(self, model_output: Any) -> np.ndarray:
        """Postprocess HuggingFace model output into a frame.
        
        Args:
            model_output: Raw model output to postprocess
            
        Returns:
            Processed frame data as numpy array
        """
        # Convert model output to numpy array
        if isinstance(model_output, torch.Tensor):
            output = model_output.detach().cpu().numpy()
        else:
            output = model_output
            
        # Ensure output is in correct format [H, W, C]
        if output.ndim == 4:  # [B, C, H, W]
            output = output[0].transpose(1, 2, 0)
        elif output.ndim == 3 and output.shape[0] == 3:  # [C, H, W]
            output = output.transpose(1, 2, 0)
            
        # Normalize to [0, 255] range if needed
        if output.max() <= 1.0:
            output = (output * 255).astype(np.uint8)
            
        return output

    def _infer_model(self, inputs: List[Any]) -> List[Any]:
        """Run HuggingFace model inference on a batch of inputs.
        
        Args:
            inputs: List of preprocessed inputs
            
        Returns:
            List of model outputs
        """
        # Stack inputs if they're tensors
        if isinstance(inputs[0], torch.Tensor):
            inputs = torch.stack(inputs)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Handle different output formats
        if isinstance(outputs, torch.Tensor):
            # Split batched tensor into list
            return [outputs[i] for i in range(outputs.size(0))]
        elif isinstance(outputs, dict):
            # Handle dictionary outputs (common in transformers)
            return [outputs[key] for key in outputs.keys()]
        else:
            # Assume outputs is already a list
            return outputs 