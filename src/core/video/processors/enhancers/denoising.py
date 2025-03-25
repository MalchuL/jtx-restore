#!/usr/bin/env python
"""
Denoising frame processors.

This module provides frame processors for reducing noise in video frames.
All processors expect and return frames in RGB format.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from src.core.video.processors.frame import ProcessorFrame
from src.core.video.processors.parallel import ParallelProcessor


class DenoiseProcessor(ParallelProcessor):
    """Frame processor for reducing noise in video frames.
    
    This processor applies various denoising algorithms to reduce noise
    while preserving image details. Works with frames in RGB format.
    Supports parallel processing for better performance.
    """
    
    def __init__(self,
                strength: float = 10.0,
                color_strength: float = 10.0,
                template_window_size: int = 7,
                search_window_size: int = 21,
                use_fast_nl_means: bool = True,
                num_workers: Optional[int] = None):
        """Initialize denoising processor.
        
        Args:
            strength: Overall denoising strength. Range: 0.0 to 20.0.
                Higher values apply stronger denoising.
            color_strength: Color denoising strength. Range: 0.0 to 20.0.
                Higher values apply stronger color denoising.
            template_window_size: Size of template patch for non-local means.
                Must be odd number. Default: 7.
            search_window_size: Size of search window for non-local means.
                Must be odd number. Default: 21.
            use_fast_nl_means: Whether to use fast non-local means denoising.
                If False, uses standard non-local means.
            num_workers: Number of worker processes for parallel processing.
        """
        super().__init__(num_workers)
        
        # Validate and store parameters
        self.strength = max(0.0, min(20.0, float(strength)))
        self.color_strength = max(0.0, min(20.0, float(color_strength)))
        
        # Ensure window sizes are odd
        self.template_window_size = template_window_size + (1 - template_window_size % 2)
        self.search_window_size = search_window_size + (1 - search_window_size % 2)
        
        self.use_fast_nl_means = bool(use_fast_nl_means)
    
    def _apply_fast_nl_means(self, frame_data: np.ndarray) -> np.ndarray:
        """Apply fast non-local means denoising."""
        return cv2.fastNlMeansDenoisingColored(
            frame_data,
            h=self.strength,
            hColor=self.color_strength,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size
        )
    
    def _apply_nl_means(self, frame_data: np.ndarray) -> np.ndarray:
        """Apply standard non-local means denoising."""
        # Convert to LAB color space for better denoising
        lab = cv2.cvtColor(frame_data, cv2.COLOR_RGB2LAB)
        
        # Denoise each channel separately
        l, a, b = cv2.split(lab)
        l = cv2.fastNlMeansDenoising(
            l,
            h=self.strength,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size
        )
        
        # Reconstruct LAB image
        denoised_lab = cv2.merge([l, a, b])
        
        # Convert back to RGB
        return cv2.cvtColor(denoised_lab, cv2.COLOR_LAB2RGB)
    
    def process_frame(self, frame: ProcessorFrame) -> ProcessorFrame:
        """Apply denoising to a single frame."""
        if frame is None:
            return None
            
        # Apply appropriate denoising algorithm
        if self.use_fast_nl_means:
            result_data = self._apply_fast_nl_means(frame.data)
        else:
            result_data = self._apply_nl_means(frame.data)
            
        return ProcessorFrame(
            data=result_data,
            frame_id=frame.frame_id,
            metadata=frame.metadata
        )
