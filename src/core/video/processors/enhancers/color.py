#!/usr/bin/env python
"""
Color correction frame processors.

This module provides frame processors for color correction and enhancement
of video frames. All processors expect and return frames in RGB format, 
while internally converting to other color spaces as needed.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from src.core.video.processors.frame import ProcessorFrame
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
    
    def _adjust_brightness_contrast(self, frame_data: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustments."""
        alpha = self.contrast
        beta = int(self.brightness * 127.0)
        return cv2.convertScaleAbs(frame_data, alpha=alpha, beta=beta)
    
    def _adjust_saturation(self, frame_data: np.ndarray) -> np.ndarray:
        """Apply saturation adjustment."""
        if self.saturation == 1.0:
            return frame_data
            
        hsv = cv2.cvtColor(frame_data, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * self.saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def _apply_gamma_correction(self, frame_data: np.ndarray) -> np.ndarray:
        """Apply gamma correction."""
        if self.gamma == 1.0:
            return frame_data
            
        inv_gamma = 1.0 / self.gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype(np.uint8)
        
        return cv2.LUT(frame_data, table)
    
    def _apply_white_balance(self, frame_data: np.ndarray) -> np.ndarray:
        """Apply automatic white balance correction."""
        if not self.white_balance:
            return frame_data
            
        lab = cv2.cvtColor(frame_data, cv2.COLOR_RGB2LAB)
        avg_a = np.average(lab[:, :, 1])
        avg_b = np.average(lab[:, :, 2])
        
        lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * 0.7)
        lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * 0.7)
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _apply_auto_exposure(self, frame_data: np.ndarray) -> np.ndarray:
        """Apply automatic exposure correction."""
        if not self.auto_exposure:
            return frame_data
            
        gray = cv2.cvtColor(frame_data, cv2.COLOR_RGB2GRAY)
        mean, std = cv2.meanStdDev(gray)
        
        target_mean = 127
        alpha = 1.0
        if mean[0][0] > 0:
            alpha = target_mean / mean[0][0]
        
        alpha = min(2.0, max(0.5, alpha))
        return cv2.convertScaleAbs(frame_data, alpha=alpha, beta=0)
    
    def process_frame(self, frame: ProcessorFrame) -> ProcessorFrame:
        """Apply color correction to a single frame."""
        if frame is None:
            return None
        
        # Apply color corrections in sequence
        result_data = frame.data.copy()
        
        # Apply auto exposure correction if enabled
        if self.auto_exposure:
            result_data = self._apply_auto_exposure(result_data)
            
        # Apply brightness and contrast adjustment
        if self.brightness != 0.0 or self.contrast != 1.0:
            result_data = self._adjust_brightness_contrast(result_data)
            
        # Apply saturation adjustment
        if self.saturation != 1.0:
            result_data = self._adjust_saturation(result_data)
            
        # Apply white balance correction if enabled
        if self.white_balance:
            result_data = self._apply_white_balance(result_data)
            
        # Apply gamma correction
        if self.gamma != 1.0:
            result_data = self._apply_gamma_correction(result_data)
            
        return ProcessorFrame(
            data=result_data,
            frame_id=frame.frame_id,
            metadata=frame.metadata
        ) 