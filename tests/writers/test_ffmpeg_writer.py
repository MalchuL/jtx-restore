#!/usr/bin/env python
"""
Tests for the FFmpeg video writer.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import numpy as np
import pytest

from src.core.video.frames.writers.ffmpeg_writer import FFmpegVideoWriter
from src.core.video.frames.types import FrameType


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test frame with random data in RGB format.
    
    Args:
        width: Frame width
        height: Frame height
        
    Returns:
        Test frame with random data in RGB format
    """
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def create_test_frames(count: int) -> List[FrameType]:
    """Create a list of test frames.
    
    Args:
        count: Number of frames to create
        
    Returns:
        List of test frames
    """
    return [create_test_frame() for _ in range(count)]


def test_writer_initialization():
    """Test writer initialization."""
    with TemporaryDirectory() as temp_dir:
        # Test basic initialization
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.mp4",
            fps=30.0,
            frame_size=(640, 480)
        )
        assert writer.fps == 30.0
        assert writer.frame_size == (640, 480)
        assert writer.codec == "libx264"
        assert writer.ffmpeg_args == []
        
        # Test with custom codec and args
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.mp4",
            fps=30.0,
            codec="libx265",
            ffmpeg_args=["-preset", "slow"]
        )
        assert writer.codec == "libx265"
        assert writer.ffmpeg_args == ["-preset", "slow"]


def test_writer_with_custom_temp_dir():
    """Test writer with custom temporary directory."""
    with TemporaryDirectory() as temp_dir:
        temp_images_dir = Path(temp_dir)  / "temp_images"
        temp_images_dir.mkdir()
        
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.mp4",
            fps=30.0,
            temp_dir=temp_images_dir
        )
        
        # Write some frames
        writer.open()
        frames = create_test_frames(3)
        for frame in frames:
            writer.write_frame(frame)
        writer.close()
        
        # Verify temporary files were created
        assert (temp_images_dir / "images").exists()
        assert len(list((temp_images_dir / "images").glob("frame_*.png"))) == 3
        assert (temp_images_dir / "frames.json").exists()


def test_writer_with_auto_temp_dir():
    """Test writer with automatic temporary directory."""
    with TemporaryDirectory() as temp_dir:
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.mp4",
            fps=30.0
        )
        
        # Write some frames
        writer.open()
        frames = create_test_frames(3)
        for frame in frames:
            writer.write_frame(frame)
        writer.close()
        
        # Verify output video was created
        assert (Path(temp_dir)  / "output.mp4").exists()
        
        # Verify temporary directory was cleaned up
        assert not writer._temp_dir_path.exists()


def test_writer_with_different_codecs():
    """Test writer with different codecs."""
    with TemporaryDirectory() as temp_dir:
        # Test H.264
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output_h264.mp4",
            fps=30.0,
            codec="libx264"
        )
        writer.open()
        writer.write_frame(create_test_frame())
        writer.close()
        assert (Path(temp_dir)  / "output_h264.mp4").exists()
        
        # Test H.265
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output_h265.mp4",
            fps=30.0,
            codec="libx265"
        )
        writer.open()
        writer.write_frame(create_test_frame())
        writer.close()
        assert (Path(temp_dir)  / "output_h265.mp4").exists()


def test_writer_with_custom_ffmpeg_args():
    """Test writer with custom FFmpeg arguments."""
    with TemporaryDirectory() as temp_dir:
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.mp4",
            fps=30.0,
            ffmpeg_args=["-preset", "slow", "-crf", "18"]
        )
        
        writer.open()
        writer.write_frame(create_test_frame())
        writer.close()
        
        # Verify output video was created
        assert (Path(temp_dir)  / "output.mp4").exists()


def test_writer_error_handling():
    """Test writer error handling."""
    with TemporaryDirectory() as temp_dir:
        # Test with invalid output path
        with pytest.raises(RuntimeError):
            writer = FFmpegVideoWriter(
                output_path="/invalid/path/output.mp4",
                fps=30.0
            )
            writer.open()
            writer.write_frame(create_test_frame())
            writer.close()
        
        # Test with invalid codec
        with pytest.raises(ValueError):
            writer = FFmpegVideoWriter(
                output_path=Path(temp_dir)  / "output.mp4",
                fps=30.0,
                codec="invalid_codec"
            )


def test_writer_context_manager():
    """Test writer as context manager."""
    with TemporaryDirectory() as temp_dir:
        with FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.mp4",
            fps=30.0
        ) as writer:
            writer.write_frame(create_test_frame())
            
        # Verify output video was created
        assert (Path(temp_dir)  / "output.mp4").exists()
        
        # Verify writer was closed
        assert not writer.is_open


def test_codec_selection():
    """Test automatic codec selection based on file extension."""
    with TemporaryDirectory() as temp_dir:
        # Test MP4 with H.264
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.mp4",
            fps=30.0
        )
        assert writer.codec in ["libx264", "libx265", "mpeg4"]
        
        # Test WebM with VP9
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.webm",
            fps=30.0
        )
        assert writer.codec in ["libvpx-vp9", "libvpx", "libx264"]
        
        # Test AVI with fallback
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.avi",
            fps=30.0
        )
        assert writer.codec in ["libx264", "libx265", "mpeg4", "msmpeg4v2"]


def test_codec_availability():
    """Test codec availability checking."""
    with TemporaryDirectory() as temp_dir:
        # Test with explicitly specified codec
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.mp4",
            fps=30.0,
            codec="libx264"
        )
        assert writer.codec == "libx264"
        
        # Test with unavailable codec
        with pytest.raises(ValueError, match="not available"):
            FFmpegVideoWriter(
                output_path=Path(temp_dir)  / "output.mp4",
                fps=30.0,
                codec="nonexistent_codec"
            )


def test_codec_presets():
    """Test codec preset arguments."""
    with TemporaryDirectory() as temp_dir:
        # Test H.264 preset
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.mp4",
            fps=30.0,
            codec="libx264"
        )
        writer.open()
        writer.write_frame(create_test_frame())
        writer.close()
        assert (Path(temp_dir)  / "output.mp4").exists()
        
        # Test VP9 preset
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.webm",
            fps=30.0,
            codec="libvpx-vp9"
        )
        writer.open()
        writer.write_frame(create_test_frame())
        writer.close()
        assert (Path(temp_dir)  / "output.webm").exists()


def test_custom_ffmpeg_args_override():
    """Test that custom FFmpeg arguments override codec presets."""
    with TemporaryDirectory() as temp_dir:
        writer = FFmpegVideoWriter(
            output_path=Path(temp_dir)  / "output.mp4",
            fps=30.0,
            codec="libx264",
            ffmpeg_args=["-preset", "slow", "-crf", "18"]
        )
        
        writer.open()
        writer.write_frame(create_test_frame())
        writer.close()
        
        # Verify output video was created
        assert (Path(temp_dir)  / "output.mp4").exists() 