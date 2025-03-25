#!/usr/bin/env python
"""
Denoising frame processors.

This module provides frame processors for reducing noise in video frames.
All processors expect and return frames in RGB format.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import cv2
import numpy as np

from src.core.video.processors.frame import ProcessedFrame
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

    def _get_parallel_kwargs(self, frame: ProcessedFrame) -> Dict[str, Any]:
        """Generate kwargs for parallel processing.
        
        Args:
            frame: The frame to process
            
        Returns:
            Dictionary containing the denoising parameters
        """
        return {
            "strength": self.strength,
            "color_strength": self.color_strength,
            "template_window_size": self.template_window_size,
            "search_window_size": self.search_window_size,
            "use_fast_nl_means": self.use_fast_nl_means
        }

    @staticmethod
    def _process_frame_parallel(
        frame: ProcessedFrame,
        strength: float,
        color_strength: float,
        template_window_size: int,
        search_window_size: int,
        use_fast_nl_means: bool
    ) -> ProcessedFrame:
        """Process a single frame in a worker process.
        
        Args:
            frame: The frame to process
            strength: Overall denoising strength
            color_strength: Color denoising strength
            template_window_size: Size of template patch
            search_window_size: Size of search window
            use_fast_nl_means: Whether to use fast non-local means
            
        Returns:
            The processed frame
        """
            
        # Apply appropriate denoising algorithm
        if use_fast_nl_means:
            result_data = cv2.fastNlMeansDenoisingColored(
                frame.data,
                h=strength,
                hColor=color_strength,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
        else:
            # Convert to LAB color space for better denoising
            lab = cv2.cvtColor(frame.data, cv2.COLOR_RGB2LAB)
            
            # Denoise each channel separately
            l, a, b = cv2.split(lab)
            l = cv2.fastNlMeansDenoising(
                l,
                h=strength,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
            
            # Reconstruct LAB image
            denoised_lab = cv2.merge([l, a, b])
            
            # Convert back to RGB
            result_data = cv2.cvtColor(denoised_lab, cv2.COLOR_LAB2RGB)
            
        return ProcessedFrame(
            data=result_data,
            frame_id=frame.frame_id,
            metadata=frame.metadata
        )

    def process_frame(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Apply denoising to a single frame."""
            
        # Use the parallel processing method with instance parameters
        return self._process_frame_parallel(
            frame,
            strength=self.strength,
            color_strength=self.color_strength,
            template_window_size=self.template_window_size,
            search_window_size=self.search_window_size,
            use_fast_nl_means=self.use_fast_nl_means
        )
