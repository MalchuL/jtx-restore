from dataclasses import dataclass
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from typing import NamedTuple

from src.core.video.readers.video_reader import VideoMetadata, VideoReader



class TestVideoMetadata:
    """Tests for the VideoMetadata class."""

    def test_initialization(self):
        """Test initializing VideoMetadata."""
        metadata = VideoMetadata(
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=300,
            duration=10.0,
            codec="mp4v",
            color_space="RGB",
            bit_depth=8
        )
        
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.fps == 30.0
        assert metadata.frame_count == 300
        assert metadata.duration == 10.0
        assert metadata.codec == "mp4v"
        assert metadata.color_space == "RGB"
        assert metadata.bit_depth == 8
    
    def test_str_representation(self):
        """Test string representation of VideoMetadata."""
        metadata = VideoMetadata(
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=300,
            duration=10.0,
            codec="mp4v"
        )
        
        result = str(metadata)
        assert "1920x1080" in result
        assert "30.0 fps" in result
        assert "300 frames" in result
        assert "10.00s" in result
        assert "mp4v" in result
        assert "RGB" in result  # Default value
        assert "8-bit" in result  # Default value


class MockVideoReader(VideoReader):
    """Mock implementation of VideoReader for testing."""
    
    def __init__(self, source_path):
        super().__init__(source_path)
        self._mock_metadata = VideoMetadata(
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=300,
            duration=10.0,
            codec="mp4v"
        )
    
    @property
    def metadata(self):
        return self._mock_metadata
    
    def open(self):
        self._is_open = True
    
    def close(self):
        self._is_open = False
    
    def read_frame(self):
        return MagicMock()
    
    def read_frames(self, count):
        return [MagicMock() for _ in range(count)]
    
    def yield_frames(self, chunk_size):
        chunk = [MagicMock() for _ in range(chunk_size)]
        yield chunk
    
    def get_frame_at_timestamp(self, timestamp_sec):
        return MagicMock()
    
    def get_frame_at_index(self, index):
        return MagicMock()

    def set_frame_index(self, index: int) -> bool:
        return True

    def set_frame_timestamp(self, timestamp_sec: float) -> bool:
        return True

    @property
    def current_index(self) -> int:
        return 0

    def reset(self) -> bool:
        return True



class TestVideoReader:
    """Tests for the VideoReader base class."""

    def test_initialization(self):
        """Test initializing VideoReader."""
        source_path = Path("/path/to/video.mp4")
        reader = MockVideoReader(source_path)
        
        assert reader.source_path == source_path
        assert not reader.is_open
    
    def test_context_manager(self):
        """Test using VideoReader as a context manager."""
        reader = MockVideoReader(Path("/path/to/video.mp4"))
        
        with reader as r:
            assert r.is_open
        
        assert not reader.is_open
    
    def test_properties(self):
        """Test VideoReader properties."""
        reader = MockVideoReader(Path("/path/to/video.mp4"))
        
        assert reader.frame_count == 300
        assert reader.fps == 30.0
        assert reader.duration == 10.0
        assert reader.resolution == (1920, 1080) 