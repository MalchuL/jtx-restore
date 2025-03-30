#!/usr/bin/env python
"""
Image reader for reading frames from a directory of images.
"""

import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import re

import cv2
import numpy as np

from src.core.video.readers.video_reader import VideoReader, VideoMetadata
from src.core.video.types import FrameType


class ImageReader(VideoReader):
    """Reader for reading frames from a directory of images."""

    # Regex pattern to find the last integer in a string (including leading zeros)
    FRAME_NUMBER_PATTERN = re.compile(r"(\d+)(?!.*\d)")

    def __init__(
        self,
        source_path: Union[str, Path],
        frame_name_template: str = "frame_{:08d}.{ext}",
        format: str = "png",
        fps: Optional[float] = None,
    ):
        """Initialize the image reader.

        Args:
            source_path: Path to the directory containing frame images
            frame_name_template: Template for frame filenames
            format: Image format (extension)
        """
        super().__init__(source_path)
        self.frame_name_template = frame_name_template
        self.format = format
        self._fps = fps
        self._current_index = 0
        self._number_of_images = 0
        self._last_frame = None  # last frame read, used to fill gaps if images are missing
        if "{:0" not in frame_name_template:
            raise ValueError(
                "frame_name_template must contain a placeholder for the frame index, it's importand to use {:0 for because we use sorting"
            )
        if "{ext}" not in frame_name_template:
            raise ValueError(
                "frame_name_template must contain a placeholder for the file extension"
            )

    def _get_frame_path(self, index: int) -> Path:
        """Get the path for a frame at the given index.

        Args:
            index: Frame index

        Returns:
            Path to the frame file
        """
        filename = self.frame_name_template.format(index, ext=self.format)
        return self.source_path / filename

    def _get_frame_number(self, filename: str) -> int:
        """Extract the frame number from a filename.

        Args:
            filename: Name of the frame file

        Returns:
            The frame number as an integer

        Raises:
            ValueError: If no frame number is found in the filename
        """
        match = self.FRAME_NUMBER_PATTERN.search(str(filename))
        if match is None:
            raise ValueError(f"No frame number found in filename: {filename}")
        return int(match.group(1))

    def _scan_frame_files(self) -> None:
        """Scan the directory for frame files."""
        if not self.source_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.source_path}")

        # Get all files with the correct extension and sort them by frame number
        frame_files = list(self.source_path.glob(f"*{self.format}"))
        ids = list(map(self._get_frame_number, frame_files))
        if len(ids) == 0:
            raise FileNotFoundError(
                f"No frame files found in {self.source_path} with format {self.format}"
            )
        self._number_of_images = max(ids) + 1


        # Read first frame to get dimensions
        first_frame_path = self._get_frame_path(0)
        first_frame = cv2.imread(str(first_frame_path))
        if first_frame is None:
            raise ValueError(f"Failed to read first frame: {first_frame_path}")

        if self._last_frame is None:
            self._last_frame = first_frame

        if self._fps is None:
            duration = None
        else:
            duration = self._number_of_images / self._fps

        self._metadata = VideoMetadata(
            width=first_frame.shape[1],
            height=first_frame.shape[0],
            fps=self._fps,
            frame_count=self._number_of_images,
            duration=duration,
            codec=self.format,
            color_space="RGB",  # We convert to RGB
        )

    def _convert_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame from BGR to RGB.

        Args:
            frame: Frame in BGR format

        Returns:
            Frame in RGB format
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @property
    def metadata(self) -> VideoMetadata:
        """Get metadata about the video."""
        if self._metadata is None:
            self._scan_frame_files()
        return self._metadata

    def open(self) -> None:
        """Open the image directory and scan for frames."""
        if not self._is_open:
            self._last_frame = None
            self._scan_frame_files()
            self._is_open = True
            self._current_index = 0

    def close(self) -> None:
        """Close the reader and release resources."""
        self._is_open = False
        self._current_index = 0
        self._last_frame = None

    def read_frame(self) -> Optional[FrameType]:
        """Read a single frame from the video.

        Returns:
            The next frame as a numpy array in RGB format, or None if no more frames
        """
        if not self._is_open:
            self.open()

        if self._current_index >= self._number_of_images:
            return None

        frame_path = self._get_frame_path(self._current_index)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            self._current_index += 1
            return self._last_frame
        else:
            self._last_frame = frame
            # Convert to RGB
            frame = self._convert_to_rgb(frame)
            self._current_index += 1
            return frame

    def get_frame_at_index(self, index: int) -> Optional[FrameType]:
        """Get the frame at a specific index.

        Args:
            index: Frame index

        Returns:
            The frame at the given index in RGB format, or None if not found
        """
        if index < 0 or index >= self._number_of_images:
            return None

        while index >= 0:
            frame_path = self._get_frame_path(index)
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                return self._convert_to_rgb(frame)
            index -= 1
        return None

    def set_frame_index(self, index: int) -> bool:
        """Set the current position to a specific frame index.

        Args:
            index: The frame index to seek to

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if index < 0 or index >= self._number_of_images:
            return False

        self._current_index = index
        while index >= 0:
            frame_path = self._get_frame_path(index)
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                self._last_frame = self._convert_to_rgb(frame)
            index -= 1
        return True

    @property
    def current_index(self) -> int:
        """Get the current frame index position.

        Returns:
            int: The current frame index (0-based)
        """
        return self._current_index

    def reset(self) -> bool:
        """Reset the reader position to the beginning.

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        self._current_index = 0
        return True

    def get_frame_at_timestamp(self, timestamp_sec: float) -> Optional[FrameType]:
        raise NotImplementedError(
            "ImageReader does not support timestamp-based frame access"
        )

    def set_frame_timestamp(self, timestamp_sec: float) -> bool:
        raise NotImplementedError(
            "ImageReader does not support timestamp-based frame access"
        )
