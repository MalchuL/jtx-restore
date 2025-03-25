from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict, Any, Union, TypeVar, Generic, NamedTuple
import logging
from dataclasses import dataclass

from src.core.video.types import FrameType


# Type variable for frame data
T = TypeVar('T')
@dataclass
class VideoMetadata:
    """Metadata for a video file."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # in seconds
    codec: str
    color_space: str = 'RGB'
    bit_depth: int = 8
    
    def __str__(self) -> str:
        return (f"Video: {self.width}x{self.height}, {self.fps} fps, {self.frame_count} frames, "
                f"{self.duration:.2f}s, {self.codec}, {self.color_space}, {self.bit_depth}-bit")


class VideoReader(ABC):
    """Abstract base class for video readers."""
    
    def __init__(self, source_path: Union[str, Path]):
        """Initialize the video reader.
        
        Args:
            source_path: Path to the source video file
        """
        self.source_path = Path(source_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._metadata: Optional[VideoMetadata] = None
        self._is_open = False
        
    @property
    @abstractmethod
    def metadata(self) -> VideoMetadata:
        """Get metadata about the video."""
        pass
    
    @abstractmethod
    def open(self) -> None:
        """Open the video file."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the video file and release resources."""
        pass
    
    def __enter__(self) -> 'VideoReader':
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    @abstractmethod
    def read_frame(self) -> Optional[FrameType]:
        """Read a single frame from the video."""
        pass
    
    @abstractmethod
    def read_frames(self, count: int) -> List[FrameType]:
        """Read multiple frames from the video."""
        pass
    
    @abstractmethod
    def yield_frames(self, chunk_size: int) -> Iterator[List[FrameType]]:
        """Yield frames from the video in chunks of specified size.
        
        Args:
            chunk_size: Number of frames to include in each chunk
        """
        pass
    
    @abstractmethod
    def get_frame_at_timestamp(self, timestamp_sec: float) -> Optional[FrameType]:
        """Get the frame at a specific timestamp."""
        pass
    
    @abstractmethod
    def get_frame_at_index(self, index: int) -> Optional[FrameType]:
        """Get the frame at a specific index."""
        pass
    
    @abstractmethod
    def set_frame_index(self, index: int) -> bool:
        """Set the current position to a specific frame index.
        
        Args:
            index: The frame index to seek to
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def set_frame_timestamp(self, timestamp_sec: float) -> bool:
        """Set the current position to a specific timestamp.
        
        Args:
            timestamp_sec: The timestamp in seconds to seek to
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def current_index(self) -> int:
        """Get the current frame index position in the video.
        
        Returns:
            int: The current frame index (0-based)
        """
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """Reset the reader position to the beginning of the video.
        
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        pass
    
    @property
    def is_open(self) -> bool:
        """Check if the video is open."""
        return self._is_open
    
    @property
    def frame_count(self) -> int:
        """Get the total number of frames in the video."""
        return self.metadata.frame_count
    
    @property
    def fps(self) -> float:
        """Get the frames per second of the video."""
        return self.metadata.fps
    
    @property
    def duration(self) -> float:
        """Get the duration of the video in seconds."""
        return self.metadata.duration
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the resolution of the video as (width, height)."""
        return (self.metadata.width, self.metadata.height)
