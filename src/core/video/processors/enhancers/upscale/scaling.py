#!/usr/bin/env python
"""
Scaling frame processors.

This module provides frame processors for upscaling and downscaling video frames.
All processors expect and return frames in RGB format.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import cv2
import numpy as np

from src.core.video.processors.processor import FrameProcessor
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.parallel import ParallelProcessor


class UpscaleProcessor(ParallelProcessor):
    """Frame processor for upscaling video frames.
    
    This processor applies standard interpolation-based upscaling to increase
    frame resolution while preserving image quality. Works with frames in RGB format.
    Supports parallel processing for better performance.
    """
    
    def __init__(self,
                scale_factor: float = 2.0,
                interpolation: str = "lanczos",
                num_workers: Optional[int] = None):
        """Initialize upscaling processor.
        
        Args:
            scale_factor: Factor by which to increase frame dimensions.
                Must be greater than 1.0. Default: 2.0.
            interpolation: Interpolation method to use. Options:
                - "nearest": Nearest neighbor (fastest, lowest quality)
                - "bilinear": Bilinear interpolation (fast, medium quality)
                - "bicubic": Bicubic interpolation (slower, better quality)
                - "lanczos": Lanczos interpolation (slowest, best quality)
            num_workers: Number of worker processes for parallel processing.
        """
        super().__init__(num_workers)
        
        # Validate and store parameters
        if scale_factor <= 1.0:
            raise ValueError("Scale factor must be greater than 1.0")
        self.scale_factor = float(scale_factor)
        
        # Map interpolation method to OpenCV constant
        self.interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4
        }.get(interpolation.lower(), cv2.INTER_LANCZOS4)

    def _get_parallel_kwargs(self, frame: ProcessedFrame) -> Dict[str, Any]:
        """Generate kwargs for parallel processing.
        
        Args:
            frame: The frame to process
            
        Returns:
            Dictionary containing the upscaling parameters
        """
        return {
            "scale_factor": self.scale_factor,
            "interpolation": self.interpolation
        }

    @staticmethod
    def _process_frame_parallel(
        frame: ProcessedFrame,
        scale_factor: float,
        interpolation: int
    ) -> ProcessedFrame:
        """Process a single frame in a worker process.
        
        Args:
            frame: The frame to process
            scale_factor: Factor by which to increase frame dimensions
            interpolation: OpenCV interpolation constant
            
        Returns:
            The processed frame
        """
            
        # Calculate new dimensions
        height, width = frame.data.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Apply upscaling
        result_data = cv2.resize(
            frame.data,
            (new_width, new_height),
            interpolation=interpolation
        )
            
        return ProcessedFrame(
            data=result_data,
            frame_id=frame.frame_id,
            metadata=frame.metadata
        )

    def process_frame(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Apply upscaling to a single frame."""
            
        # Use the parallel processing method with instance parameters
        return self._process_frame_parallel(
            frame,
            scale_factor=self.scale_factor,
            interpolation=self.interpolation
        )
