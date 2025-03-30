#!/usr/bin/env python
"""
Tests for the image writer.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any

import numpy as np
import pytest
import cv2

from src.core.video.writers.image_writer import ImageWriter
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


def test_writer_initialization():
    """Test writer initialization and directory creation."""
    with TemporaryDirectory() as temp_dir:
        writer = ImageWriter(
            output_path=temp_dir,
            fps=30.0,
            frame_size=(640, 480)
        )
        
        # Check directories were created
        assert Path(temp_dir).exists()
        assert not (Path(temp_dir) / "images").exists()
        
        # Check writer state
        assert writer.format == "png"
        assert writer.fps == 30.0
        assert writer.frame_size == (640, 480)
        assert writer.current_frame == 0


def test_frame_saving():
    """Test frame saving and file structure."""
    with TemporaryDirectory() as temp_dir:
        writer = ImageWriter(
            output_path=temp_dir,
            fps=30.0,
            frame_size=(640, 480)
        )
        
        # Create and save test frame
        frame = create_test_frame()
        writer.write_frame(frame)
        
        # Check frame file was created
        frame_path = Path(temp_dir) / "images" / "frame_00000000.png"
        assert frame_path.exists()
        
        # Verify frame data (should be in BGR format when read back)
        saved_frame = cv2.imread(str(frame_path))
        assert saved_frame is not None
        assert saved_frame.shape == frame.shape
        
        # Convert back to RGB for comparison
        saved_frame_rgb = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(saved_frame_rgb, frame)


def test_metadata_handling():
    """Test metadata creation and saving."""
    with TemporaryDirectory() as temp_dir:
        writer = ImageWriter(
            output_path=temp_dir,
            fps=30.0,
            frame_size=(640, 480)
        )
        
        # Create and save test frame
        frame = create_test_frame()
        writer.write_frame(frame)
        writer.close()
        
        # Check metadata file was created
        metadata_path = Path(temp_dir) / "frames.json"
        assert metadata_path.exists()
        
        # Load and verify metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        # Check video-level metadata
        assert metadata["fps"] == 30.0
        assert metadata["frame_size"] == [640, 480]
        assert metadata["format"] == "png"
        assert metadata["frame_count"] == 1
        assert metadata["duration"] == 1/30.0
        assert metadata["frame_name_template"] == "frame_{:08d}.{ext}"
        
        # Check frames array
        assert len(metadata["frames"]) == 1
        frame_metadata = metadata["frames"][0]
        
        # Check frame-specific metadata
        assert frame_metadata["frame_id"] == 0
        assert frame_metadata["frame_number"] == 0
        assert frame_metadata["timestamp"] is None
        assert frame_metadata["shape"] == [480, 640, 3]
        assert frame_metadata["dtype"] == "uint8"
        assert frame_metadata["output_path"] == "images/frame_00000000.png"


def test_multiple_frames():
    """Test saving multiple frames."""
    with TemporaryDirectory() as temp_dir:
        writer = ImageWriter(
            output_path=temp_dir,
            fps=30.0,
            frame_size=(640, 480)
        )
        
        # Save multiple frames
        for i in range(3):
            frame = create_test_frame()
            writer.write_frame(frame)
        writer.close()
        
        # Check all frame files were created
        for i in range(3):
            frame_path = Path(temp_dir) / "images" / f"frame_{i:08d}.png"
            assert frame_path.exists()
        
        # Check metadata
        metadata_path = Path(temp_dir) / "frames.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        # Check video-level metadata
        assert metadata["frame_count"] == 3
        assert metadata["duration"] == 3/30.0
        
        # Check frames array
        assert len(metadata["frames"]) == 3
        for i, frame_metadata in enumerate(metadata["frames"]):
            assert frame_metadata["frame_id"] == i
            assert frame_metadata["frame_number"] == i
            assert frame_metadata["output_path"] == f"images/frame_{i:08d}.png"


def test_empty_frame():
    """Test handling of empty frames."""
    with TemporaryDirectory() as temp_dir:
        writer = ImageWriter(
            output_path=temp_dir,
            fps=30.0,
            frame_size=(640, 480)
        )
        
        # Try to save None frame
        with pytest.raises(ValueError):
            writer.write_frame(None)
        writer.close()


def test_custom_format():
    """Test saving frames in different formats."""
    with TemporaryDirectory() as temp_dir:
        writer = ImageWriter(
            output_path=temp_dir,
            fps=30.0,
            frame_size=(640, 480),
            format="jpg"
        )
        
        # Save test frame
        frame = create_test_frame()
        writer.write_frame(frame)
        
        # Check file extension
        frame_path = Path(temp_dir) / "images" / "frame_00000000.jpg"
        assert frame_path.exists()
        
        writer.close()
        # Check metadata format
        metadata_path = Path(temp_dir) / "frames.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        assert metadata["format"] == "jpg"


def test_custom_naming():
    """Test custom frame naming template."""
    with TemporaryDirectory() as temp_dir:
        writer = ImageWriter(
            output_path=temp_dir,
            fps=30.0,
            frame_size=(640, 480),
            frame_name_template="img_{:04d}.{ext}"
        )
        
        # Save test frame
        frame = create_test_frame()
        writer.write_frame(frame)

        writer.close()
        
        # Check file naming
        frame_path = Path(temp_dir) / "images" / "img_0000.png"
        assert frame_path.exists()
        
        # Check metadata
        metadata_path = Path(temp_dir) / "frames.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        assert metadata["frame_name_template"] == "img_{:04d}.{ext}"
        assert metadata["frames"][0]["output_path"] == "images/img_0000.png"

