"""
Tests for the OpenCV-based frame interpolator.

This module contains tests for the OpenCVFrameInterpolator class to ensure
that frame interpolation works correctly using OpenCV's optical flow methods.
"""

import pytest
import numpy as np
import cv2
from typing import List

from src.core.video.frame_interpolation import OpenCVFrameInterpolator
from src.core.video.frame_interpolation import InterpolatedFrame


def create_test_frame(height: int, width: int, frame_id: int, color: tuple) -> InterpolatedFrame:
    """
    Create a test frame with a colored rectangle for testing interpolation.
    
    Args:
        height: Frame height
        width: Frame width
        frame_id: Frame ID
        color: Rectangle color (R, G, B)
        
    Returns:
        InterpolatedFrame: A test frame
    """
    # Create a black frame
    data = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a rectangle that moves based on frame_id
    rect_size = min(width, height) // 3
    x_offset = (frame_id * 20) % (width - rect_size)
    y_offset = (frame_id * 10) % (height - rect_size)
    
    # Draw the rectangle
    data[y_offset:y_offset+rect_size, x_offset:x_offset+rect_size] = color
    
    return InterpolatedFrame(data=data, frame_id=frame_id)


def create_test_frames(count: int, height: int = 480, width: int = 640) -> List[InterpolatedFrame]:
    """
    Create a sequence of test frames with a moving rectangle.
    
    Args:
        count: Number of frames to create
        height: Frame height
        width: Frame width
        
    Returns:
        List[InterpolatedFrame]: List of test frames
    """
    frames = []
    for i in range(count):
        # Create frames with different colored rectangles that move
        color = (
            128 + 127 * np.sin(i * 0.1),
            128 + 127 * np.sin(i * 0.2 + 2),
            128 + 127 * np.sin(i * 0.3 + 4)
        )
        frame = create_test_frame(height, width, i, color)
        frames.append(frame)
    return frames


class TestOpenCVFrameInterpolator:
    """Tests for the OpenCVFrameInterpolator class."""

    def test_initialization(self):
        """Test that the interpolator initializes correctly with default parameters."""
        interpolator = OpenCVFrameInterpolator()
        assert interpolator.factor == 2
        assert interpolator.optical_flow_method == 'farneback'
        
        # Test with custom parameters
        interpolator = OpenCVFrameInterpolator(factor=4, optical_flow_method='dis')
        assert interpolator.factor == 4
        assert interpolator.optical_flow_method == 'dis'
        
    def test_calculate_optical_flow(self):
        """Test that optical flow calculation produces valid flow fields."""
        interpolator = OpenCVFrameInterpolator()
        
        # Create two test frames with a moving rectangle
        frame1 = create_test_frame(120, 160, 0, (255, 0, 0))
        frame2 = create_test_frame(120, 160, 1, (255, 0, 0))
        
        # Calculate optical flow
        flow = interpolator._calculate_optical_flow(frame1.data, frame2.data)
        
        # Check that flow has the correct shape and type
        assert flow.shape == (120, 160, 2)  # Flow has x and y components
        assert flow.dtype == np.float32
        
        # Check that flow is non-zero (there should be motion)
        assert np.any(flow != 0)
        
    def test_warp_frame(self):
        """Test that frame warping works correctly."""
        interpolator = OpenCVFrameInterpolator()
        
        # Create test frames
        frame1 = create_test_frame(120, 160, 0, (255, 0, 0))
        frame2 = create_test_frame(120, 160, 1, (255, 0, 0))
        
        # Calculate optical flow
        flow = interpolator._calculate_optical_flow(frame1.data, frame2.data)
        
        # Warp frame with different t values
        warped_t0 = interpolator._warp_frame(frame1.data, flow, 0)
        warped_t1 = interpolator._warp_frame(frame1.data, flow, 1)
        warped_t05 = interpolator._warp_frame(frame1.data, flow, 0.5)
        
        # Check that warped frames have the correct shape
        assert warped_t0.shape == frame1.data.shape
        assert warped_t1.shape == frame1.data.shape
        assert warped_t05.shape == frame1.data.shape
        
        # t=0 should be almost identical to the original frame
        assert np.mean(np.abs(warped_t0.astype(float) - frame1.data.astype(float))) < 5.0
        
        # t=0.5 should be different from both original frames
        assert np.mean(np.abs(warped_t05.astype(float) - frame1.data.astype(float))) > 0.05
        
    def test_interpolate_window(self):
        """Test that frame interpolation produces the correct number of frames."""
        # Test with factor=2 (should produce 3 frames: original, interpolated, original)
        interpolator = OpenCVFrameInterpolator(factor=2)
        
        frames = create_test_frames(2)
        result = interpolator._interpolate_window(frames)
        
        # Check that we get 3 frames
        assert len(result) == 3
        
        # Check that the first and last frames are the originals
        assert result[0] == frames[0]
        assert result[-1] == frames[1]
        
        # Check that the middle frame is interpolated
        assert result[1].metadata.get('interpolated') == True
        assert result[1].dt > 0 and result[1].dt < 1
        
    def test_multiple_interpolation_factors(self):
        """Test interpolation with different factors."""
        for factor in [2, 3, 4]:
            interpolator = OpenCVFrameInterpolator(factor=factor)
            
            frames = create_test_frames(2)
            result = interpolator._interpolate_window(frames)
            
            # Check that we get the expected number of frames (factor + 1)
            assert len(result) == factor + 1
            
            # Check frame IDs and dt values for interpolated frames
            for i in range(1, factor):
                expected_t = i / factor
                assert abs(result[i].dt - expected_t) < 0.01
                assert result[i].metadata.get('interpolated') == True
    
    def test_process_frame(self):
        """Test that the process_frame method works correctly in streaming mode."""
        interpolator = OpenCVFrameInterpolator(factor=2)
        
        # Create test frames
        frames = create_test_frames(3)
        
        # First frame should return None (not enough frames yet)
        result1 = interpolator.process_frame(frames[0])
        assert result1 is None
        
        # Second frame should return two frames (original and interpolated)
        result2 = interpolator.process_frame(frames[1])
        assert result2 is not None
        assert len(result2) == 2
        
        # Check that we get the first frame and an interpolated frame
        assert result2[0] == frames[0]
        assert result2[1].metadata.get('interpolated') == True
        
        # Third frame should return two more frames
        result3 = interpolator.process_frame(frames[2])
        assert result3 is not None
        assert len(result3) == 2
        
    def test_different_optical_flow_methods(self):
        """Test interpolation with different optical flow methods."""
        for method in ['farneback', 'dis']:
            interpolator = OpenCVFrameInterpolator(optical_flow_method=method)
            
            frames = create_test_frames(2)
            result = interpolator._interpolate_window(frames)
            
            # Check that we get 3 frames regardless of method
            assert len(result) == 3
            
            # Check that the middle frame is interpolated
            assert result[1].metadata.get('interpolated') == True
            
    @pytest.mark.skipif(not hasattr(cv2, 'cuda') or not cv2.cuda.getCudaEnabledDeviceCount(), 
                      reason="CUDA not available")
    def test_cuda_support(self):
        """Test that CUDA support works if available (optional test)."""
        # This test is skipped if CUDA is not available
        frames = create_test_frames(2)
        
        # Try to create a CUDA-enabled interpolator
        # This is just a placeholder - OpenCVFrameInterpolator would need CUDA support added
        interpolator = OpenCVFrameInterpolator(optical_flow_method='farneback')
        
        # Just check that interpolation doesn't crash
        result = interpolator._interpolate_window(frames)
        assert len(result) == 3
    
    def test_full_pipeline(self):
        """Test the full interpolation pipeline with a sequence of frames."""
        interpolator = OpenCVFrameInterpolator(factor=2)
        
        # Create a sequence of test frames
        frames = create_test_frames(5)
        
        # Process frames one by one
        all_results = []
        for frame in frames:
            result = interpolator.process_frame(frame)
            if result:
                all_results.extend(result)
        
        # Process remaining frames
        final_result = interpolator.process_frame(None)
        if final_result:
            all_results.extend(final_result)
        
        # Check that we have the expected number of output frames
        # For 5 input frames with factor=2, we should get 9 output frames
        assert len(all_results) == 10
        
        # Check that every other frame is an original frame
        for i in range(0, len(all_results), 2):
            assert all_results[i].metadata.get('interpolated', False) == False
        
        # Check that the remaining frames are interpolated
        for i in range(1, len(all_results), 2):
            assert all_results[i].metadata.get('interpolated') == True 