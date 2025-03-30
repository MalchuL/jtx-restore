import os
import cv2
from pathlib import Path
from typing import Iterator, List, Optional, Union, Dict, Any
import logging

from src.core.video.types import FrameType
from src.core.video.readers.video_reader import VideoReader, VideoMetadata


class FolderCacheReader(VideoReader):
    """
    A reader wrapper that caches frames in a folder for faster subsequent access.

    This reader wraps another VideoReader and saves frames to disk as they are read.
    On subsequent reads, it will check if the frame exists in the cache before
    requesting it from the base reader.
    """

    def __init__(
        self,
        base_reader: VideoReader,
        cache_folder: Union[str, Path],
        image_format: str = "png",
        frame_name_template: str = "frame_{:08d}.{ext}",
    ):
        """Initialize the folder cache reader.

        Args:
            base_reader (VideoReader): The underlying video reader to wrap
            cache_folder (Union[str, Path]): Path to the folder where frames will be cached
            image_format (str, optional): Format to save images as (png, jpg, etc). Defaults to "png".
            frame_name_template (str, optional): Template for frame filenames. Defaults to "frame_{:08d}.{ext}".
        """
        super().__init__(base_reader.source_path)
        self.base_reader = base_reader
        self.cache_folder = Path(cache_folder)
        self.image_format = image_format
        self.frame_name_template = frame_name_template

        # Ensure the cache folder exists
        self.cache_folder.mkdir(parents=True, exist_ok=True)

    @property
    def metadata(self) -> VideoMetadata:
        """Get metadata from the base reader."""
        return self.base_reader.metadata

    @property
    def frame_count(self) -> int:
        """Get frame count from the base reader."""
        return self.base_reader.frame_count

    @property
    def fps(self) -> float:
        """Get fps from the base reader."""
        return self.base_reader.fps

    @property
    def duration(self) -> float:
        """Get duration from the base reader."""
        return self.base_reader.duration

    @property
    def resolution(self) -> tuple:
        """Get resolution from the base reader."""
        return self.base_reader.resolution

    @property
    def is_open(self) -> bool:
        """Check if the reader is open.
        
        Returns:
            bool: True if the reader is open, False otherwise
        """
        return self.base_reader.is_open

    def open(self) -> None:
        """Open the base reader."""
        if not self.is_open:
            self.base_reader.open()
            self.logger.debug(
                f"Opened FolderCacheReader with cache folder: {self.cache_folder}"
            )

    def close(self) -> None:
        """Close the base reader."""
        if self.is_open:
            self.base_reader.close()
            self.logger.debug(
                f"Closed FolderCacheReader with cache folder: {self.cache_folder}"
            )

    def get_frame_path(self, frame_index: int) -> Path:
        """Get the path where a frame would be stored in the cache.

        Args:
            frame_index (int): The index of the frame

        Returns:
            Path: Path to the cached frame file
        """
        filename = self.frame_name_template.format(frame_index, ext=self.image_format)
        return self.cache_folder / filename

    def frame_is_cached(self, frame_index: int) -> bool:
        """Check if a frame exists in the cache.

        Args:
            frame_index (int): The index of the frame to check

        Returns:
            bool: True if the frame is in the cache, False otherwise
        """
        frame_path = self.get_frame_path(frame_index)
        return frame_path.exists()

    def save_frame_to_cache(self, frame: FrameType, frame_index: int) -> Path:
        """Save a frame to the cache folder.

        Args:
            frame (FrameType): The frame to save (in RGB format)
            frame_index (int): The index of the frame

        Returns:
            Path: Path to the saved frame file
        """
        frame_path = self.get_frame_path(frame_index)
        # Convert RGB to BGR for OpenCV's imwrite
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)
        self.logger.debug(f"Cached frame {frame_index} to {frame_path}")
        return frame_path

    def load_frame_from_cache(self, frame_index: int) -> Optional[FrameType]:
        """Load a frame from the cache.

        Args:
            frame_index (int): The index of the frame to load

        Returns:
            Optional[FrameType]: The loaded frame in RGB format if available, None otherwise
        """
        frame_path = self.get_frame_path(frame_index)
        if not frame_path.exists():
            return None
            
        # OpenCV loads in BGR format, convert to RGB
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            self.logger.warning(f"Failed to load cached frame from {frame_path}")
            return None
        
        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.logger.debug(f"Loaded frame {frame_index} from cache {frame_path}")
        return frame_rgb

    def read_frame(self) -> Optional[FrameType]:
        """Read a single frame, using cache if available.

        Returns:
            Optional[FrameType]: The next frame in RGB format if available, None otherwise
        """
        if not self.is_open:
            self.open()

        # Get current frame index using the abstract property instead of direct _cap access
        frame_index = self.base_reader.current_index

        # Check if frame is in cache
        if self.frame_is_cached(frame_index):
            frame = self.load_frame_from_cache(frame_index)
            if frame is not None:
                # Advance the base reader position by reading a frame and discarding it
                # This preserves the internal state of the base reader
                self.base_reader.read_frame()
                return frame

        # If not in cache or failed to load, get from base reader and cache it
        frame = self.base_reader.read_frame()
        if frame is not None:
            self.save_frame_to_cache(frame, frame_index)

        return frame

    def read_frames(self, count: int) -> List[FrameType]:
        """Read multiple frames, using cache where available.

        Args:
            count (int): Number of frames to read

        Returns:
            List[FrameType]: List of frames
        """
        if not self.is_open:
            self.open()

        frames = []
        for _ in range(count):
            frame = self.read_frame()
            if frame is None:
                break
            frames.append(frame)

        return frames

    def yield_frames(self, chunk_size: int) -> Iterator[List[FrameType]]:
        """Yield chunks of frames from the video, using cache where available.

        Args:
            chunk_size: Number of frames to include in each chunk
        """
        if not self.is_open:
            self.open()

        frames = []

        while True:
            frame = self.read_frame()
            if frame is None:
                # End of video reached
                if frames:  # Yield any remaining frames
                    yield frames
                break

            frames.append(frame)
            if len(frames) >= chunk_size:
                yield frames
                frames = []

    def get_frame_at_timestamp(self, timestamp_sec: float) -> Optional[FrameType]:
        """Get the frame at a specific timestamp, using cache if available."""
        # Convert timestamp to frame index
        frame_idx = int(timestamp_sec * self.fps)
        return self.get_frame_at_index(frame_idx)

    def get_frame_at_index(self, index: int) -> Optional[FrameType]:
        """Get a frame at a specific index, using cache if available.

        Args:
            index (int): The index of the frame to get

        Returns:
            Optional[FrameType]: The frame if available, None otherwise
        """
        if not self.is_open:
            self.open()

        if index < 0 or index >= self.frame_count:
            self.logger.warning(
                f"Frame index {index} is out of range [0, {self.frame_count-1}]"
            )
            return None

        # Check if frame is in cache
        if self.frame_is_cached(index):
            frame = self.load_frame_from_cache(index)
            if frame is not None:
                return frame

        # If not in cache or failed to load, get from base reader and cache it
        frame = self.base_reader.get_frame_at_index(index)
        if frame is not None:
            self.save_frame_to_cache(frame, index)

        return frame

    def set_frame_index(self, index: int) -> bool:
        """Set the current position to a specific frame index.
        
        Args:
            index: The frame index to seek to
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if not self.is_open:
            self.open()
        
        # Delegate to base reader
        success = self.base_reader.set_frame_index(index)
        
        if success:
            self.logger.debug(f"Set frame position to index {index}")
        
        return success

    def set_frame_timestamp(self, timestamp_sec: float) -> bool:
        """Set the current position to a specific timestamp.
        
        Args:
            timestamp_sec: The timestamp in seconds to seek to
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if not self.is_open:
            self.open()
        
        # Delegate to base reader
        success = self.base_reader.set_frame_timestamp(timestamp_sec)
        
        if success:
            self.logger.debug(f"Set frame position to timestamp {timestamp_sec} seconds")
        
        return success

    @property
    def current_index(self) -> int:
        """Get the current frame index position in the video.
        
        Returns:
            int: The current frame index (0-based)
        """
        if not self.is_open:
            self.open()
        
        # Delegate to base reader
        return self.base_reader.current_index

    def reset(self) -> bool:
        """Reset the reader position to the beginning of the video.
        
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if not self.is_open:
            self.open()
        
        # Delegate to base reader
        success = self.base_reader.reset()
        
        if success:
            self.logger.debug("Reset reader position to the beginning")
        
        return success

    def clear_cache(self) -> None:
        """Clear all cached frames from the cache folder."""
        # Extract the extension part from the template
        ext_part = self.image_format
        
        # Create a more robust pattern by using a glob pattern that matches the static parts
        # of the filename and ignores the format placeholder
        parts = self.frame_name_template.split("{")
        
        # Get the prefix before any format placeholder
        prefix = parts[0]
        
        # Look for all files in the cache directory with the right prefix and extension
        # This is more robust than trying to replace format placeholders
        for cached_file in self.cache_folder.glob(f"{prefix}*{ext_part}"):
            if cached_file.is_file():
                cached_file.unlink()
        
        self.logger.info(f"Cleared cache folder: {self.cache_folder}")
