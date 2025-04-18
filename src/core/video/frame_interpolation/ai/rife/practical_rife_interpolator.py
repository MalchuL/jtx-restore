"""
Practical RIFE (Real-Time Intermediate Flow Estimation) frame interpolator.

This module provides a frame interpolator implementation based on the RIFE v4.25
algorithm, which is designed for real-time frame interpolation.
"""

import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple, Dict, Any

from src.core.video.frame_interpolation.frame_interpolator import StreamingFrameInterpolator
from src.core.video.frame_interpolation.interpolated_frame import InterpolatedFrame
from src.core.video.frame_interpolation.ai.rife.practical_rife_4_25 import Model


class PracticalRIFEFrameInterpolator425(StreamingFrameInterpolator[InterpolatedFrame]):
    """
    Practical RIFE (Real-Time Intermediate Flow Estimation) frame interpolator.
    
    This interpolator uses the RIFE v4.25 algorithm to generate intermediate frames
    between two input frames, providing high-quality frame interpolation.
    
    Attributes:
        model_path (str): Path to the pre-trained RIFE model
        device (str): Device to run the model on ('cuda' or 'cpu')
        _model (Model): The loaded RIFE model
        _factor (float): The frame rate increase factor
        scale (float): Scale factor for the model (1.0 for original resolution)
    """
    
    def __init__(
        self, 
        factor: float = 2.0,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scale: float = 1.0
    ):
        """
        Initialize the RIFE interpolator.
        
        Args:
            factor: The frame rate increase factor (e.g., 2.0 doubles the frame rate)
            model_path: Path to the pre-trained RIFE model directory
            device: Device to run the model on ('cuda' or 'cpu')
            scale: Scale factor for the model (1.0 for original resolution)
        """
        super().__init__(factor=factor)
        self.model_path = model_path
        self.device = device
        self.scale = scale
        self._model = None
        self.logger = logging.getLogger(__name__)
        
        # Load the model if a path is provided
        if model_path:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the RIFE model from the specified path."""
        try:
            self.logger.info(f"Loading RIFE model from {self.model_path}")
            self._model = Model()
            self._model.load_model(self.model_path)
            self._model.eval()
            self.logger.info("RIFE model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load RIFE model: {e}")
            raise
    
    def _preprocess_frames(self, frame1: InterpolatedFrame, frame2: InterpolatedFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess frames for the RIFE model.
        
        Args:
            frame1: First input frame
            frame2: Second input frame
            
        Returns:
            Tuple of preprocessed tensors for both frames
        """
        h, w, _ = frame1.data.shape
        tmp = max(128, int(128 / self.scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        
        
        
        # Convert frames to tensors and normalize
        frame1_tensor = torch.from_numpy(frame1.data).permute(2, 0, 1).float() / 255.0
        frame2_tensor = torch.from_numpy(frame2.data).permute(2, 0, 1).float() / 255.0
        
        frame1_tensor = self._pad_image(frame1_tensor, padding)
        frame2_tensor = self._pad_image(frame2_tensor, padding)
        
        # Add batch dimension
        frame1_tensor = frame1_tensor.unsqueeze(0).to(self.device)
        frame2_tensor = frame2_tensor.unsqueeze(0).to(self.device)
        
        return frame1_tensor, frame2_tensor
    
    @staticmethod
    def _pad_image(img, padding, fp16=False):
        if(fp16):
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)
    
    def _postprocess_frame(self, tensor: torch.Tensor, frame_id: float, dt: float, original_shape: Tuple[int, int, int]) -> InterpolatedFrame:
        """
        Convert a model output tensor to an InterpolatedFrame.
        
        Args:
            tensor: Output tensor from the model
            frame_id: ID for the interpolated frame
            dt: Time difference between the two frames
            original_shape: Shape of the original frames
        Returns:
            InterpolatedFrame object
        """
        h, w, _ = original_shape
        
        # Convert tensor to numpy array
        with torch.no_grad():
            tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
        tensor = tensor[:h, :w, :]
        
        # Create metadata for the interpolated frame
        metadata = {
            "interpolated": True,
            "interpolation_method": "rife_4.25",
            "model_path": self.model_path,
            "scale": self.scale,
            "dt": dt
        }
        
        return InterpolatedFrame(data=tensor, frame_id=frame_id, dt=dt, metadata=metadata)
    
    def _interpolate_window(self, window: Sequence[InterpolatedFrame]) -> List[InterpolatedFrame]:
        """
        Interpolate frames within a window using the RIFE model.
        
        Args:
            window: Sequence of frames to interpolate between
            
        Returns:
            List of interpolated frames
        """
        if len(window) < 2:
            self.logger.warning("Not enough frames for interpolation")
            return list(window)
        
        frame1, frame2 = window[0], window[1]
        
        # If model is not loaded, return original frames
        if self._model is None:
            self.logger.warning("RIFE model not loaded, returning original frames")
            return [frame1, frame2]
        
        # Preprocess frames
        frame1_tensor, frame2_tensor = self._preprocess_frames(frame1, frame2)
        
        # Generate interpolated frames
        interpolated_frames = []
        
        # Add the first original frame
        interpolated_frames.append(frame1)
        
        # Calculate the number of intermediate frames to generate
        num_intermediate = int(self.factor) - 1
        
        # Generate intermediate frames
        for i in range(1, num_intermediate + 1):
            # Calculate interpolation factor (t)
            t = i / (num_intermediate + 1)
            # Run model inference
            with torch.no_grad():
                intermediate_tensor = self._model.inference(
                    frame1_tensor, 
                    frame2_tensor, 
                    timestep=t, 
                    scale=self.scale
                )
            
            # Calculate the frame ID for the interpolated frame
            frame_id = frame1.frame_id
            
            # Convert the tensor to an InterpolatedFrame
            intermediate_frame = self._postprocess_frame(intermediate_tensor, frame_id, t, frame1.shape)
            interpolated_frames.append(intermediate_frame)
        
        # Add the second original frame
        interpolated_frames.append(frame2)
        
        return interpolated_frames 