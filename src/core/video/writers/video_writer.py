from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple, Iterator, TypeVar, Generic
import numpy as np

from src.core.video.types import FrameType

# Type variable for frame types
T = TypeVar('T')

class VideoWriter(ABC, Generic[T]):
    """Abstract base class for video writers.
    
    This class defines the interface for writing frames to a video file,
    with implementations responsible for handling codec selection,
    frame formatting, and other format-specific considerations.
    """
    
    def __init__(self, 
                 output_path: Union[str, Path], 
                 fps: float,
                 frame_size: Optional[Tuple[int, int]] = None,
                 codec: Optional[str] = None):
        """Initialize the video writer.
        
        Args:
            output_path (Union[str, Path]): Path where the video will be saved
            fps (float): Frames per second
            frame_size (Optional[Tuple[int, int]], optional): Size of video frames as (width, height), 
                or None to determine from first frame. Defaults to None.
            codec (Optional[str], optional): Codec to use for encoding, or None for auto-selection.
                Defaults to None.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self._codec = codec
        self._is_open = False
        
    @property
    def is_open(self) -> bool:
        """Check if the writer is open and ready to write frames.
        
        Returns:
            bool: True if the writer is open, False otherwise
        """
        return self._is_open
    
    @abstractmethod
    def open(self) -> None:
        """Open the writer and prepare for writing frames."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the writer and finalize the video file."""
        pass
    
    def __enter__(self) -> 'VideoWriter':
        """Context manager entry.
        
        Returns:
            VideoWriter: The video writer instance
        """
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    @abstractmethod
    def write_frame(self, frame: FrameType) -> None:
        """Write a single frame to the video.
        
        Args:
            frame (FrameType): The frame to write
        """
        pass
    
    def _check_frame_size(self, frame: FrameType) -> None:
        if self.frame_size is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
        elif self.frame_size != (frame.shape[1], frame.shape[0]):
            raise ValueError(f"Frame size mismatch: {self.frame_size} != {frame.shape[1]}x{frame.shape[0]}")
    
    def write_frames(self, frames: List[FrameType]) -> None:
        """Write multiple frames to the video.
        
        Args:
            frames (List[FrameType]): List of frames to write
        """
        for frame in frames:
            self.write_frame(frame)
    
    def save_video(self, frames: Union[List[FrameType], Iterator[FrameType]]) -> Path:
        """Save frames directly to a video file in a memory-efficient way.
        
        This method handles the complete workflow of opening the writer,
        writing all frames, and finalizing the video file. It's designed to
        be memory-efficient when working with iterators, as it processes
        one frame at a time without loading the entire video into memory.
        
        Args:
            frames (Union[List[FrameType], Iterator[FrameType]]): 
                List or iterator of frames to save to video
                
        Returns:
            Path: Path to the saved video file
            
        Raises:
            IOError: If the video could not be created or is invalid
        """
        frame_count = 0
        
        try:
            with self:
                # Handle both list and iterator inputs
                for frame in frames:
                    self.write_frame(frame)
                    frame_count += 1
                    
            return self.output_path
            
        except Exception as e:
            raise
    
    @property
    @abstractmethod
    def codec(self) -> str:
        """Get the codec that was actually used for encoding.
        
        Returns:
            str: The codec used for encoding
        """
        pass
