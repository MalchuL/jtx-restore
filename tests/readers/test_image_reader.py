#!/usr/bin/env python
"""
Tests for the image reader.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any

import numpy as np
import pytest
import cv2

from src.core.video.readers.image_reader import ImageReader
from src.core.video.types import FrameType


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test frame with random data in RGB format.
    
    Args:
        width: Frame width
        height: Frame height
        
    Returns:
        Test frame with random data in RGB format
    """
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def save_test_frames(
    output_dir: Path,
    count: int,
    frame_name_template: str = "frame_{:08d}.{ext}",
    format: str = "png"
) -> None:
    """Save test frames to a directory.
    
    Args:
        output_dir: Directory to save frames
        count: Number of frames to save
        frame_name_template: Template for frame filenames
        format: Image format
    """
    for i in range(count):
        frame = create_test_frame()
        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = frame_name_template.format(i, ext=format)
        output_dir = Path(output_dir)
        cv2.imwrite(str(output_dir / filename), frame_bgr)


def test_reader_initialization():
    """Test reader initialization and validation."""
    with TemporaryDirectory() as temp_dir:
        # Test valid initialization
        reader = ImageReader(
            source_path=temp_dir,
            frame_name_template="frame_{:08d}.{ext}",
            format="png"
        )
        assert reader.format == "png"
        assert reader.frame_name_template == "frame_{:08d}.{ext}"
        assert not reader.is_finished
        
        # Test invalid template (missing frame index)
        with pytest.raises(ValueError, match="must contain a placeholder for the frame index"):
            ImageReader(
                source_path=temp_dir,
                frame_name_template="frame.{ext}",
                format="png"
            )
            
        # Test invalid template (missing extension)
        with pytest.raises(ValueError, match="must contain a placeholder for the file extension"):
            ImageReader(
                source_path=temp_dir,
                frame_name_template="frame_{:08d}",
                format="png"
            )


def test_reader_with_empty_directory():
    """Test reader behavior with empty directory."""
    with TemporaryDirectory() as temp_dir:
        reader = ImageReader(source_path=temp_dir)
        
        # Should raise FileNotFoundError when trying to read
        with pytest.raises(FileNotFoundError):
            reader.open()


def test_frame_reading():
    """Test basic frame reading functionality."""
    with TemporaryDirectory() as temp_dir:
        # Save test frames
        save_test_frames(temp_dir, count=3)
        
        # Initialize reader
        reader = ImageReader(source_path=temp_dir)
        reader.open()
        assert not reader.is_finished
        
        # Read frames
        frame1 = reader.read_frame()
        assert frame1 is not None
        assert not reader.is_finished
        
        frame2 = reader.read_frame()
        assert frame2 is not None
        assert not reader.is_finished
        
        frame3 = reader.read_frame()
        assert frame3 is not None
        assert not reader.is_finished
        
        frame4 = reader.read_frame()  # Should be None
        assert frame4 is None
        assert reader.is_finished
        
        # Verify frames
        assert frame1 is not None
        assert frame2 is not None
        assert frame3 is not None
        assert frame4 is None
        
        # Verify frame shapes
        assert frame1.shape == (480, 640, 3)
        assert frame2.shape == (480, 640, 3)
        assert frame3.shape == (480, 640, 3)
        
        # Verify frames are in RGB format
        assert np.array_equal(frame1, cv2.cvtColor(cv2.imread(str(Path(temp_dir) / "frame_00000000.png")), cv2.COLOR_BGR2RGB))


def test_metadata():
    """Test metadata handling."""
    with TemporaryDirectory() as temp_dir:
        # Save test frames
        save_test_frames(temp_dir, count=3)
        
        # Initialize reader
        reader = ImageReader(source_path=temp_dir)
        reader.open()
        
        # Get metadata
        metadata = reader.metadata
        
        # Verify metadata
        assert metadata.width == 640
        assert metadata.height == 480
        assert metadata.frame_count == 3
        assert metadata.codec == "png"
        assert metadata.color_space == "RGB"
        assert metadata.fps is None
        assert metadata.duration is None


def test_random_access():
    """Test random frame access functionality."""
    with TemporaryDirectory() as temp_dir:
        # Save test frames
        save_test_frames(temp_dir, count=3)
        
        # Initialize reader
        reader = ImageReader(source_path=temp_dir)
        reader.open()
        assert not reader.is_finished
        
        # Test get_frame_at_index
        frame0 = reader.get_frame_at_index(0)
        assert frame0 is not None
        assert not reader.is_finished
        
        frame1 = reader.get_frame_at_index(1)
        assert frame1 is not None
        assert not reader.is_finished
        
        frame2 = reader.get_frame_at_index(2)
        assert frame2 is not None
        assert not reader.is_finished
        
        frame3 = reader.get_frame_at_index(3)  # Should be None
        assert frame3 is None
        assert not reader.is_finished  # get_frame_at_index doesn't affect is_finished
        
        # Test set_frame_index
        reader._last_frame = None
        assert reader.set_frame_index(1)
        assert reader.current_index == 1
        assert reader._last_frame is not None
        assert not reader.is_finished
        assert not reader.set_frame_index(3)  # Invalid index
        assert not reader.is_finished
        
        # Test reset
        assert reader.reset()
        assert reader.current_index == 0
        assert not reader.is_finished


def test_batch_reading():
    """Test batch frame reading functionality."""
    with TemporaryDirectory() as temp_dir:
        # Save test frames
        save_test_frames(temp_dir, count=5)
        
        # Initialize reader
        reader = ImageReader(source_path=temp_dir)
        reader.open()
        assert not reader.is_finished
        
        # Test read_frames
        frames = reader.read_frames(3)
        assert len(frames) == 3
        assert all(frame.shape == (480, 640, 3) for frame in frames)
        assert not reader.is_finished
        
        # Test read_frames beyond available frames
        frames = reader.read_frames(5)
        assert len(frames) == 2  # Only 2 frames left
        assert reader.is_finished
        
        # Test yield_frames
        chunks = list(reader.yield_frames(chunk_size=2))
        assert len(chunks) == 0  # No frames left after previous reads
        assert reader.is_finished


def test_custom_format():
    """Test reading frames in different formats."""
    with TemporaryDirectory() as temp_dir:
        # Save test frames in JPG format
        save_test_frames(temp_dir, count=3, format="jpg")
        
        # Initialize reader with JPG format
        reader = ImageReader(
            source_path=temp_dir,
            frame_name_template="frame_{:08d}.{ext}",
            format="jpg"
        )
        reader.open()
        
        # Read and verify frames
        frame = reader.read_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert reader.metadata.codec == "jpg"


def test_custom_naming():
    """Test custom frame naming template."""
    with TemporaryDirectory() as temp_dir:
        # Save test frames with custom naming
        save_test_frames(
            temp_dir,
            count=3,
            frame_name_template="img_{:04d}.{ext}"
        )
        
        # Initialize reader with matching template
        reader = ImageReader(
            source_path=temp_dir,
            frame_name_template="img_{:04d}.{ext}"
        )
        reader.open()
        
        # Read and verify frames
        frame = reader.read_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        
        # Verify frame count
        assert reader.metadata.frame_count == 3


def test_context_manager():
    """Test using the reader as a context manager."""
    with TemporaryDirectory() as temp_dir:
        # Save test frames
        save_test_frames(temp_dir, count=3)
        
        # Test context manager
        with ImageReader(source_path=temp_dir) as reader:
            assert reader.is_open
            assert not reader.is_finished
            
            # Read all frames
            while reader.read_frame() is not None:
                pass
            
            assert reader.is_finished
        
        # Should be closed after exiting context
        assert not reader.is_open
        assert reader.is_finished


def test_timestamp_metadata():
    """Test timestamp-related metadata."""
    with TemporaryDirectory() as temp_dir:
        # Save test frames
        save_test_frames(temp_dir, count=3)
        
        # Initialize reader
        reader = ImageReader(source_path=temp_dir)
        reader.open()
        
        # Verify FPS and duration
        assert reader.metadata.fps is None
        assert reader.metadata.duration is None
        
        # Verify frame count
        assert reader.metadata.frame_count == 3 