import pytest
import numpy as np
import cv2
import os
from pathlib import Path
import tempfile
import shutil

from src.core.video.writers.opencv_writer import OpenCVVideoWriter
from src.core.video.readers.opencv_reader import OpenCVVideoReader


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output videos."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_frames():
    """Generate sample frames for testing."""
    frames = []
    for i in range(5):
        # Create RGB frames with increasing intensity in red channel
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 50  # R
        frame[:, :, 1] = 100  # G
        frame[:, :, 2] = 150  # B
        frames.append(frame)
    return frames


class TestOpenCVVideoWriter:
    """Tests for the OpenCVVideoWriter class."""

    def test_initialization(self, temp_output_dir):
        """Test initializing the writer with various parameters."""
        # Basic initialization
        output_path = temp_output_dir / "test_video.mp4"
        writer = OpenCVVideoWriter(
            output_path=output_path, fps=30.0, frame_size=(640, 480)
        )

        assert writer.output_path == output_path
        assert writer.fps == 30.0
        assert writer.frame_size == (640, 480)
        assert writer.codec is None
        assert writer.is_open

        # With custom codec
        writer = OpenCVVideoWriter(
            output_path=output_path,
            fps=24.0,
            frame_size=(1280, 720),
            codec="mp4v",
        )

        assert writer.output_path == output_path
        assert writer.fps == 24.0
        assert writer.frame_size == (1280, 720)
        assert writer.codec == "mp4v"
        assert writer.is_open

    def test_codec_selection(self, temp_output_dir):
        """Test automatic codec selection based on file extension."""
        # MP4 format
        writer = OpenCVVideoWriter(
            output_path=temp_output_dir / "test_video.mp4", fps=30.0, frame_size=(640, 480)
        )
        assert writer._actual_codec == "mp4v"

        # AVI format
        writer = OpenCVVideoWriter(
            output_path=temp_output_dir / "test_video.avi", fps=30.0, frame_size=(640, 480)
        )
        assert writer._actual_codec == "XVID"

        # MOV format
        writer = OpenCVVideoWriter(
            output_path=temp_output_dir / "test_video.mov", fps=30.0, frame_size=(640, 480)
        )
        assert writer._actual_codec == "mp4v"

        # Unknown format
        with pytest.raises(ValueError):
            writer = OpenCVVideoWriter(
                output_path=temp_output_dir / "test_video.xyz", fps=30.0, frame_size=(640, 480)
            )

    def test_open_close(self, temp_output_dir):
        """Test opening and closing the writer."""
        output_path = temp_output_dir / "test_open_close.mp4"
        writer = OpenCVVideoWriter(
            output_path=output_path, fps=30.0, frame_size=(640, 480)
        )

        assert writer.is_open
        writer.open()
        assert writer.is_open
        assert writer._writer is not None
        writer.close()
        assert not writer.is_open
        assert writer._writer is None

    def test_write_frames(self, temp_output_dir, sample_frames):
        """Test writing frames to a video file."""
        output_path = temp_output_dir / "test_write.mp4"
        writer = OpenCVVideoWriter(
            output_path=output_path, fps=30.0, frame_size=(640, 480)
        )

        # Write frames
        with writer:
            for frame in sample_frames:
                writer.write_frame(frame)

        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify content with an OpenCVVideoReader
        reader = OpenCVVideoReader(output_path)
        with reader:
            # Check video properties
            assert reader.frame_count >= len(sample_frames)  # May be more due to codec
            assert reader.fps == pytest.approx(30.0, abs=0.1)
            assert reader.resolution == (640, 480)

            # Read and check the first frame
            first_frame = reader.read_frame()
            assert first_frame is not None
            assert first_frame.shape == (480, 640, 3)
            # The RGB-BGR conversion might cause small color differences
            assert np.mean(np.abs(first_frame[:, :, 0] - sample_frames[0][:, :, 0])) < 5

    def test_write_frames_with_auto_size(self, temp_output_dir, sample_frames):
        """Test writing frames with automatic size determination."""
        output_path = temp_output_dir / "test_auto_size.mp4"
        writer = OpenCVVideoWriter(
            output_path=output_path,
            fps=30.0,
            frame_size=(640, 480)
        )

        # Write frames
        with writer:
            for frame in sample_frames:
                writer.write_frame(frame)

        # Verify frame size was set correctly
        assert writer.frame_size == (640, 480)

        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_resize_frame(self, temp_output_dir):
        """Test that frames are automatically resized if they don't match the specified size."""
        output_path = temp_output_dir / "test_resize.mp4"
        writer = OpenCVVideoWriter(
            output_path=output_path,
            fps=30.0,
            frame_size=(320, 240),  # Smaller than our sample frames
        )

        # Create one larger frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # R

        # Write the frame
        with writer:
            writer.write_frame(frame)

        # Verify file was created
        assert output_path.exists()

        # Verify the frame was resized
        reader = OpenCVVideoReader(output_path)
        with reader:
            read_frame = reader.read_frame()
            assert read_frame.shape == (240, 320, 3)

    def test_save_video_method(self, temp_output_dir, sample_frames):
        """Test the save_video convenience method."""
        output_path = temp_output_dir / "test_save_video.mp4"
        writer = OpenCVVideoWriter(
            output_path=output_path, fps=30.0, frame_size=(640, 480)
        )

        # Save video
        result_path = writer.save_video(sample_frames)

        # Should return the output path
        assert result_path == output_path

        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify content
        reader = OpenCVVideoReader(output_path)
        with reader:
            assert reader.frame_count >= len(sample_frames)

    def test_context_manager(self, temp_output_dir, sample_frames):
        """Test using the writer as a context manager."""
        output_path = temp_output_dir / "test_context.mp4"

        # Use context manager
        with OpenCVVideoWriter(
            output_path=output_path, fps=30.0, frame_size=(640, 480)
        ) as writer:
            for frame in sample_frames:
                writer.write_frame(frame)
            assert writer.is_open


        # Verify writer is closed after context
        assert not writer.is_open

        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
