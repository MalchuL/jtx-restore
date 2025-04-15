#!/usr/bin/env python
"""
Tests for the upscaling frame processor.
"""

import numpy as np
import pytest
import cv2

from src.core.video.frames.processors.enhancers.upscale.scaling import UpscaleProcessor
from src.core.video.frames.processors.frame import ProcessedFrame

raise NotImplementedError("Upscaling processor is not implemented yet.")
@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    # Create a simple test pattern
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[25:75, 25:75] = 255  # White square in the middle
    return frame


@pytest.fixture
def sample_frames(sample_frame):
    """Create a batch of sample frames for testing."""
    return [sample_frame.copy() for _ in range(5)]


@pytest.fixture
def processor():
    """Create a default upscale processor."""
    return UpscaleProcessor(scale_factor=2.0)


def test_upscale_processor_initialization():
    """Test proper initialization of the upscale processor."""
    # Test valid initialization
    processor = UpscaleProcessor(scale_factor=2.0)
    assert processor.scale_factor == 2.0
    assert processor.interpolation == cv2.INTER_LANCZOS4  # Default interpolation

    # Test invalid scale factor
    with pytest.raises(ValueError):
        UpscaleProcessor(scale_factor=0.5)

    # Test different interpolation methods
    processor = UpscaleProcessor(interpolation="nearest")
    assert processor.interpolation == cv2.INTER_NEAREST

    processor = UpscaleProcessor(interpolation="bilinear")
    assert processor.interpolation == cv2.INTER_LINEAR

    processor = UpscaleProcessor(interpolation="bicubic")
    assert processor.interpolation == cv2.INTER_CUBIC


def test_upscale_processor_output_shape(processor, sample_frame):
    """Test that the output frame has the correct shape."""
    input_frame = ProcessedFrame(data=sample_frame, frame_id=0)
    output_frame = processor.process_frame(input_frame)

    expected_height = int(sample_frame.shape[0] * processor.scale_factor)
    expected_width = int(sample_frame.shape[1] * processor.scale_factor)

    assert output_frame.data.shape == (expected_height, expected_width, 3)
    assert output_frame.frame_id == 0


def test_upscale_processor_preserves_data(processor, sample_frame):
    """Test that the upscaling preserves the general structure of the image."""
    input_frame = ProcessedFrame(data=sample_frame, frame_id=0)
    output_frame = processor.process_frame(input_frame)

    # Check that the white square is still visible and roughly in the same position
    center_y = output_frame.data.shape[0] // 2
    center_x = output_frame.data.shape[1] // 2

    # The white square should be roughly in the center
    assert (
        np.mean(
            output_frame.data[
                center_y - 25 : center_y + 25, center_x - 25 : center_x + 25
            ]
        )
        > 200
    )


def test_upscale_processor_with_none_frame(processor):
    """Test handling of None input frame."""
    output_frame = processor.process_frame(None)
    assert output_frame is None


def test_upscale_processor_with_different_scale_factors(sample_frame):
    """Test upscaling with different scale factors."""
    scale_factors = [1.5, 2.0, 3.0, 4.0]

    for scale_factor in scale_factors:
        processor = UpscaleProcessor(scale_factor=scale_factor)
        input_frame = ProcessedFrame(data=sample_frame, frame_id=0)
        output_frame = processor.process_frame(input_frame)

        expected_height = int(sample_frame.shape[0] * scale_factor)
        expected_width = int(sample_frame.shape[1] * scale_factor)

        assert output_frame.data.shape == (expected_height, expected_width, 3)


def test_upscale_processor_with_different_interpolation_methods(sample_frame):
    """Test upscaling with different interpolation methods."""
    interpolation_methods = ["nearest", "bilinear", "bicubic", "lanczos"]

    for method in interpolation_methods:
        processor = UpscaleProcessor(interpolation=method)
        input_frame = ProcessedFrame(data=sample_frame, frame_id=0)
        output_frame = processor.process_frame(input_frame)

        # Check that the output has the correct shape
        expected_height = int(sample_frame.shape[0] * processor.scale_factor)
        expected_width = int(sample_frame.shape[1] * processor.scale_factor)
        assert output_frame.data.shape == (expected_height, expected_width, 3)

        # Check that the output is valid (not all zeros or all ones)
        assert np.any(output_frame.data > 0)
        assert np.any(output_frame.data < 255)


def test_upscale_processor_preserves_frame_id(sample_frame):
    """Test that the frame ID is preserved during processing."""
    frame_id = 42
    input_frame = ProcessedFrame(data=sample_frame, frame_id=frame_id)
    processor = UpscaleProcessor()
    output_frame = processor.process_frame(input_frame)

    assert output_frame.frame_id == frame_id


def test_upscale_processor_parallel_processing(sample_frames):
    """Test parallel processing of frames."""
    # Create processor with 2 workers
    processor = UpscaleProcessor(scale_factor=2.0, num_workers=2)

    # Create input frames with ProcessorFrame wrapper
    input_frames = [
        ProcessedFrame(data=frame, frame_id=i) for i, frame in enumerate(sample_frames)
    ]

    # Process frames in parallel
    output_frames = processor.process_batch(input_frames)

    # Check results
    assert len(output_frames) == len(input_frames)
    for i, (input_frame, output_frame) in enumerate(zip(input_frames, output_frames)):
        # Check frame ID preservation
        assert output_frame.frame_id == i

        # Check output shape
        expected_height = int(input_frame.data.shape[0] * processor.scale_factor)
        expected_width = int(input_frame.data.shape[1] * processor.scale_factor)
        assert output_frame.data.shape == (expected_height, expected_width, 3)


def test_upscale_processor_sequential_vs_parallel(sample_frames):
    """Test that parallel processing produces same results as sequential."""
    # Create processor with 2 workers
    parallel_processor = UpscaleProcessor(scale_factor=2.0, num_workers=2)
    sequential_processor = UpscaleProcessor(scale_factor=2.0, num_workers=0)

    # Create input frames
    input_frames = [
        ProcessedFrame(data=frame, frame_id=i) for i, frame in enumerate(sample_frames)
    ]

    # Process frames both ways
    parallel_output = parallel_processor.process_batch(input_frames)
    sequential_output = sequential_processor.process_batch(input_frames)

    # Compare results
    assert len(parallel_output) == len(sequential_output)
    for p_out, s_out in zip(parallel_output, sequential_output):
        # Results should be identical
        assert np.array_equal(p_out.data, s_out.data)
        assert p_out.frame_id == s_out.frame_id


def test_upscale_processor_empty_batch(processor):
    """Test processing of empty batch."""
    output_frames = processor.process_batch([])
    assert len(output_frames) == 0


def test_upscale_processor_single_frame_batch(processor, sample_frame):
    """Test processing of single frame batch."""
    input_frame = ProcessedFrame(data=sample_frame, frame_id=0)
    output_frames = processor.process_batch([input_frame])

    assert len(output_frames) == 1
    assert output_frames[0].frame_id == 0

    # Check output shape
    expected_height = int(sample_frame.shape[0] * processor.scale_factor)
    expected_width = int(sample_frame.shape[1] * processor.scale_factor)
    assert output_frames[0].data.shape == (expected_height, expected_width, 3)


def test_upscale_processor_large_batch(sample_frame):
    """Test processing of large batch."""
    # Create a large batch of frames
    large_batch = [sample_frame.copy() for _ in range(20)]
    input_frames = [
        ProcessedFrame(data=frame, frame_id=i) for i, frame in enumerate(large_batch)
    ]

    # Process with different worker counts
    for num_workers in [0, 2, 4]:
        processor = UpscaleProcessor(scale_factor=2.0, num_workers=num_workers)
        output_frames = processor.process_batch(input_frames)

        assert len(output_frames) == len(input_frames)
        for i, output_frame in enumerate(output_frames):
            assert output_frame.frame_id == i
            expected_height = int(sample_frame.shape[0] * processor.scale_factor)
            expected_width = int(sample_frame.shape[1] * processor.scale_factor)
            assert output_frame.data.shape == (expected_height, expected_width, 3)
