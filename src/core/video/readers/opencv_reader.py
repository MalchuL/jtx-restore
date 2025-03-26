import cv2
import os
import time
from typing import Iterator, List, Optional, Union
from pathlib import Path
import numpy as np
import logging

from src.core.video.types import FrameType

from src.core.video.readers.reader import VideoReader, VideoMetadata


class OpenCVVideoReader(VideoReader):
    """Video reader implementation using OpenCV."""

    def __init__(self, source_path: Union[str, Path]):
        """Initialize the OpenCV video reader.

        Args:
            source_path (Union[str, Path]): Path to the source video file
        """
        super().__init__(source_path)
        self._cap = None
        self._metadata_cache = None

    @property
    def metadata(self) -> VideoMetadata:
        """Get metadata about the video."""
        if self._metadata_cache is None:
            if not self.is_open:
                self.open()
                should_close = True
            else:
                should_close = False

            # Extract metadata
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            # Get codec as a string
            fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            # These might not be available for all videos/OpenCV builds
            color_space = "RGB"

            try:
                bit_depth = 8  # Default assumption
                # Could try to determine based on the first frame
                ret, frame = self._cap.read()
                if ret:
                    bit_depth = 8 * frame.itemsize
                    # Reset position
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except:
                bit_depth = 8

            self._metadata_cache = VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                codec=codec,
                color_space=color_space,
                bit_depth=bit_depth,
            )

            if should_close:
                self.close()

        return self._metadata_cache

    def open(self) -> None:
        """Open the video file."""
        if not self.is_open:
            if not self.source_path.exists():
                raise FileNotFoundError(f"Video file not found: {self.source_path}")

            self._cap = cv2.VideoCapture(str(self.source_path))
            if not self._cap.isOpened():
                raise IOError(f"Failed to open video file: {self.source_path}")

            self._is_open = True
            self.logger.debug(f"Opened video: {self.source_path}")

    def close(self) -> None:
        """Close the video file and release resources."""
        if self.is_open and self._cap is not None:
            self._cap.release()
            self._is_open = False
            self.logger.debug(f"Closed video: {self.source_path}")

    def read_frame(self) -> Optional[FrameType]:
        """Read a single frame from the video and convert from BGR to RGB.

        Returns:
            Optional[FrameType]: The frame in RGB format if successful, None otherwise
        """
        if not self.is_open:
            self.open()

        ret, frame = self._cap.read()
        if ret:
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        return None


    def get_frame_at_timestamp(self, timestamp_sec: float) -> Optional[FrameType]:
        """Get the frame at a specific timestamp."""
        if not self.is_open:
            self.open()

        if timestamp_sec < 0 or timestamp_sec > self.duration:
            self.logger.warning(
                f"Timestamp {timestamp_sec} is out of range [0, {self.duration}]"
            )
            return None

        # Calculate frame index from timestamp
        frame_idx = int(timestamp_sec * self.fps)
        return self.get_frame_at_index(frame_idx)

    def get_frame_at_index(self, index: int) -> Optional[FrameType]:
        """Get the frame at a specific index and convert from BGR to RGB.

        Args:
            index (int): The index of the frame to get

        Returns:
            Optional[FrameType]: The frame in RGB format if successful, None otherwise
        """
        if not self.is_open:
            self.open()

        if index < 0 or index >= self.frame_count:
            self.logger.warning(
                f"Frame index {index} is out of range [0, {self.frame_count-1}]"
            )
            return None

        # Store current position
        current_position = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Set the video position
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()

        # Restore original position
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)

        if not ret:
            self.logger.error(f"Failed to read frame at index {index}")
            return None

        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def set_frame_index(self, index: int) -> bool:
        """Set the current position to a specific frame index.
        
        Args:
            index: The frame index to seek to
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if not self.is_open:
            self.open()
            
        if index < 0 or index >= self.frame_count:
            self.logger.warning(
                f"Frame index {index} is out of range [0, {self.frame_count-1}]"
            )
            return False
            
        # Set the video position using OpenCV's CAP_PROP_POS_FRAMES
        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        
        if not success:
            self.logger.error(f"Failed to set frame position to index {index}")
            return False
            
        self.logger.debug(f"Set frame position to index {index}")
        return True
    
    def set_frame_timestamp(self, timestamp_sec: float) -> bool:
        """Set the current position to a specific timestamp.
        
        Args:
            timestamp_sec: The timestamp in seconds to seek to
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if not self.is_open:
            self.open()
            
        if timestamp_sec < 0 or timestamp_sec > self.duration:
            self.logger.warning(
                f"Timestamp {timestamp_sec} is out of range [0, {self.duration}]"
            )
            return False
            
        # Convert timestamp to milliseconds for OpenCV's CAP_PROP_POS_MSEC
        msec = timestamp_sec * 1000.0
        success = self._cap.set(cv2.CAP_PROP_POS_MSEC, msec)
        
        if not success:
            # If setting by milliseconds fails, try using frames
            frame_idx = int(timestamp_sec * self.fps)
            success = self.set_frame_index(frame_idx)
            
            if not success:
                self.logger.error(f"Failed to set frame position to timestamp {timestamp_sec} seconds")
                return False
                
        self.logger.debug(f"Set frame position to timestamp {timestamp_sec} seconds")
        return True
    
    @property
    def current_index(self) -> int:
        """Get the current frame index position in the video.
        
        Returns:
            int: The current frame index (0-based)
        """
        if not self.is_open:
            self.open()
            
        current_index = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.logger.debug(f"Current frame position: {current_index}")
        return current_index
    
    def reset(self) -> bool:
        """Reset the reader position to the beginning of the video.
        
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        return self.set_frame_index(0)
