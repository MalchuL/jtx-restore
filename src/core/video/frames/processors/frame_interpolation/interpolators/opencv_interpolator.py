"""
OpenCV-based frame interpolation for increasing video frame rate.

This module implements a frame interpolator that uses OpenCV's optical flow
methods to generate intermediate frames between existing frames.
"""

from typing import List, Sequence, Optional
import numpy as np
import cv2

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.frame_interpolation.frame_interpolator import FrameInterpolator
from src.core.video.frames.processors.frame_interpolation.interpolated_frame import InterpolatedFrame




class OpenCVFrameInterpolator(FrameInterpolator):
    """
    A frame interpolator that uses OpenCV's optical flow to generate intermediate frames.
    
    This implementation uses Farneback's optical flow algorithm to estimate motion
    between consecutive frames and then warps the frames to create intermediate frames.
    
    Attributes:
        optical_flow_params (dict): Parameters for the optical flow algorithm
    """
    
    def __init__(self, factor: float = 2, optical_flow_method: str = 'farneback'):
        """
        Initialize the OpenCV frame interpolator.
        
        Args:
            factor: The frame rate increase factor (e.g., 2 doubles the frame rate)
            optical_flow_method: The optical flow method to use ('farneback' or 'dis')
        """
        super().__init__(factor=factor)
        self.optical_flow_method = optical_flow_method
        
        # Default parameters for Farneback optical flow
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # Default parameters for DIS optical flow
        self.dis_params = {
            'preset': cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
            'finest_scale': 2,
            'gradient_desc_iterations': 25,
            'patch_size': 8,
            'patch_stride': 4
        }
    
    def _calculate_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Calculate optical flow between two frames.
        
        Args:
            frame1: The first frame (as a numpy array)
            frame2: The second frame (as a numpy array)
            
        Returns:
            np.ndarray: Optical flow field as a numpy array
        """
        # Convert frames to grayscale for optical flow calculation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        if self.optical_flow_method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                gray1, 
                gray2, 
                None,
                self.farneback_params['pyr_scale'],
                self.farneback_params['levels'],
                self.farneback_params['winsize'],
                self.farneback_params['iterations'],
                self.farneback_params['poly_n'],
                self.farneback_params['poly_sigma'],
                self.farneback_params['flags']
            )
        elif self.optical_flow_method == 'dis':
            # Use DIS optical flow algorithm
            flow_calculator = cv2.DISOpticalFlow_create(self.dis_params['preset'])
            flow_calculator.setFinestScale(self.dis_params['finest_scale'])
            flow_calculator.setGradientDescentIterations(self.dis_params['gradient_desc_iterations'])
            flow_calculator.setPatchSize(self.dis_params['patch_size'])
            flow_calculator.setPatchStride(self.dis_params['patch_stride'])
            flow = flow_calculator.calc(gray1, gray2, None)
        else:
            raise ValueError(f"Unsupported optical flow method: {self.optical_flow_method}")
            
        return flow
    
    def _warp_frame(self, frame: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
        """
        Warp a frame using optical flow with interpolation factor t.
        
        Args:
            frame: The frame to warp
            flow: The optical flow field
            t: Interpolation factor (0-1) where 0 is the first frame and 1 is the second
            
        Returns:
            np.ndarray: Warped frame
        """
        h, w = frame.shape[:2]
        
        # Create grid of pixel coordinates
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Scale the flow by the interpolation factor
        flow_scaled = flow * t
        
        # Calculate new coordinates
        x_new = x_coords + flow_scaled[..., 0]
        y_new = y_coords + flow_scaled[..., 1]
        
        # Stack coordinates for remap
        map_xy = np.stack([x_new, y_new], axis=-1)
        
        # Warp the frame using remap
        warped = cv2.remap(frame, map_xy, None, cv2.INTER_LINEAR)
        
        return warped
    
    def _interpolate_window(self, window: Sequence[ProcessedFrame]) -> List[InterpolatedFrame]:
        """
        Interpolate frames within a window using optical flow.
        
        Args:
            window: Sequence of frames to interpolate between
            
        Returns:
            List[InterpolatedFrame]: List of interpolated frames including originals
        """
        if len(window) < 2:
            self.logger.warning(f"Need at least 2 frames for interpolation, got {len(window)}")
            return list(window)
        
        # Get the two frames to interpolate between
        frame1, frame2 = window[0], window[1]
        
        # Calculate optical flow between frames
        flow_forward = self._calculate_optical_flow(frame1.data, frame2.data)
        flow_backward = self._calculate_optical_flow(frame2.data, frame1.data)
        
        # Number of intermediate frames to generate
        n_interp = int(self.factor - 1)
        
        result = [frame1]  # Start with the first frame
        
        # Generate intermediate frames
        for i in range(1, n_interp + 1):
            t = i / float(n_interp + 1)  # Interpolation factor
            
            # Warp both frames towards the intermediate position
            warped1 = self._warp_frame(frame1.data, flow_forward, t)
            warped2 = self._warp_frame(frame2.data, flow_backward, 1 - t)
            
            # Blend the warped frames based on the interpolation factor
            blended = (1 - t) * warped1 + t * warped2
            blended = blended.astype(np.uint8)
            
            # Create a new interpolated frame
            interp_frame = InterpolatedFrame(
                data=blended,
                frame_id=int(frame1.frame_id),  # Keep the base frame_id
                dt=t,  # Store the offset in dt
                metadata={
                    **frame1.metadata,
                    'interpolated': True,
                    'interp_factor': t
                }
            )
            
            result.append(interp_frame)
        
        # Add the second frame to the result
        result.append(frame2)
        
        return result 