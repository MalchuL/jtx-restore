#!/usr/bin/env python
"""
Color correction frame processors.

This module provides frame processors for color correction and enhancement
of video frames. All processors expect and return frames in RGB format, 
while internally converting to other color spaces as needed.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.parallel import ParallelProcessor


class ColorCorrectionProcessor(ParallelProcessor):
    """Frame processor for color correction and enhancement.
    
    This processor adjusts brightness, contrast, saturation, and white balance
    of video frames in RGB format. Supports parallel processing for better performance.
    """
    
    def __init__(self, 
                brightness: float = 0.0,
                contrast: float = 1.0,
                saturation: float = 1.0,
                white_balance: bool = False,
                gamma: float = 1.0,
                auto_exposure: bool = False,
                num_workers: Optional[int] = None):
        """Initialize color correction processor.
        
        Args:
            brightness: Brightness adjustment. Range: -1.0 to 1.0.
                Negative values darken, positive values brighten.
            contrast: Contrast adjustment. Range: 0.0 to 3.0.
                Values below 1.0 reduce contrast, above 1.0 increase it.
            saturation: Saturation adjustment. Range: 0.0 to 3.0.
                Values below 1.0 reduce saturation, above 1.0 increase it.
            white_balance: Whether to apply automatic white balance correction.
            gamma: Gamma correction value. Range: 0.1 to 3.0.
                Values below 1.0 brighten shadows, above 1.0 darken midtones.
            auto_exposure: Whether to apply automatic exposure correction.
            num_workers: Number of worker processes for parallel processing.
        """
        super().__init__(num_workers)
        
        # Color correction parameters
        self.brightness = max(-1.0, min(1.0, float(brightness)))
        self.contrast = max(0.0, min(3.0, float(contrast)))
        self.saturation = max(0.0, min(3.0, float(saturation)))
        self.white_balance = bool(white_balance)
        self.gamma = max(0.1, min(3.0, float(gamma)))
        self.auto_exposure = bool(auto_exposure)

    def _get_parallel_kwargs(self, frame: ProcessedFrame) -> Dict[str, Any]:
        """Generate kwargs for parallel processing.
        
        Args:
            frame: The frame to process
            
        Returns:
            Dictionary containing the color correction parameters
        """
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "white_balance": self.white_balance,
            "gamma": self.gamma,
            "auto_exposure": self.auto_exposure
        }

    @staticmethod
    def _process_frame_parallel(
        frame: ProcessedFrame,
        brightness: float,
        contrast: float,
        saturation: float,
        white_balance: bool,
        gamma: float,
        auto_exposure: bool
    ) -> ProcessedFrame:
        """Process a single frame in a worker process.
        
        Args:
            frame: The frame to process
            brightness: Brightness adjustment value
            contrast: Contrast adjustment value
            saturation: Saturation adjustment value
            white_balance: Whether to apply white balance
            gamma: Gamma correction value
            auto_exposure: Whether to apply auto exposure
            
        Returns:
            The processed frame
        """
            
        # Apply color corrections in sequence
        result_data = frame.data.copy()
        
        # Apply auto exposure correction if enabled
        if auto_exposure:
            gray = cv2.cvtColor(result_data, cv2.COLOR_RGB2GRAY)
            mean, std = cv2.meanStdDev(gray)
            
            target_mean = 127
            alpha = 1.0
            if mean[0][0] > 0:
                alpha = target_mean / mean[0][0]
            
            alpha = min(2.0, max(0.5, alpha))
            result_data = cv2.convertScaleAbs(result_data, alpha=alpha, beta=0)
            
        # Apply brightness and contrast adjustment
        if brightness != 0.0 or contrast != 1.0:
            alpha = contrast
            beta = int(brightness * 127.0)
            result_data = cv2.convertScaleAbs(result_data, alpha=alpha, beta=beta)
            
        # Apply saturation adjustment
        if saturation != 1.0:
            hsv = cv2.cvtColor(result_data, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            result_data = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
        # Apply white balance correction if enabled
        if white_balance:
            lab = cv2.cvtColor(result_data, cv2.COLOR_RGB2LAB)
            avg_a = np.average(lab[:, :, 1])
            avg_b = np.average(lab[:, :, 2])
            
            lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * 0.7)
            lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * 0.7)
            
            result_data = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
        # Apply gamma correction
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]).astype(np.uint8)
            
            result_data = cv2.LUT(result_data, table)
            
        return ProcessedFrame(
            data=result_data,
            frame_id=frame.frame_id,
            metadata=frame.metadata
        )

    def process_frame(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Apply color correction to a single frame."""
            
        # Use the parallel processing method with instance parameters
        return self._process_frame_parallel(
            frame,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            white_balance=self.white_balance,
            gamma=self.gamma,
            auto_exposure=self.auto_exposure
        ) 