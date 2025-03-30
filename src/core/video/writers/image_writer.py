#!/usr/bin/env python
"""
Image writer for saving frames as individual images.

This module provides the ImageWriter class for saving video frames as
individual image files in a specified directory.
"""

import json
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple

import cv2
import numpy as np

from src.core.video.writers.video_writer import VideoWriter
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.types import FrameType


class ImageWriter(VideoWriter[FrameType]):
    """Writer for saving frames as individual image files.

    This class saves each frame as a separate image file in a specified
    directory. It supports common image formats and provides options for
    frame numbering and file naming.
    """
    IMAGE_FOLDER = "images"


    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: Optional[str] = None,
        format: str = "png",
        frame_name_template: str = "frame_{:08d}.{ext}",
        metadata_file: str = "frames.json",
        saving_freq: int = 10000,
    ):
        """Initialize the image writer.

        Args:
            output_path: Directory to save image files
            fps: Frames per second (used for metadata only)
            frame_size: Size of video frames as (width, height)
            codec: Codec to use (not used for images)
            format: Image format to use (defaults to "png")
            frame_name_template: Template for frame filenames
            metadata_file: Name of the metadata JSON file
            saving_freq: Number of frames between saves to avoid missing frames (default 10000)
        """
        super().__init__(
            output_path=output_path, fps=fps, frame_size=frame_size, codec=codec
        )

        self.output_dir = Path(output_path)
        self.images_dir = self.output_dir / self.IMAGE_FOLDER
        self.format = format.lower()
        self.frame_name_template = frame_name_template
        self.metadata_file = metadata_file
        self.current_frame = 0
        self.metadata_list: List[Dict[str, Any]] = []
        self.saving_freq = saving_freq
        

    def open(self) -> None:
        """Open the writer and prepare for writing frames."""
        if not self._is_open:
            # Create output directories if they don't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)
        self._is_open = True


    def _get_frame_path(self, frame_id: int) -> Path:
        """Generate the path for a frame file.

        Args:
            frame_id: Frame ID to generate path for

        Returns:
            Path to the frame file
        """
        filename = self.frame_name_template.format(frame_id, ext=self.format)
        return self.images_dir / filename

    def _get_relative_path(self, path: Path) -> str:
        """Get path relative to output directory.

        Args:
            path: Path to convert

        Returns:
            Relative path string
        """
        return str(path.relative_to(self.output_dir))

    def _convert_to_bgr(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame from RGB to BGR for OpenCV.

        Args:
            frame: Frame in RGB format

        Returns:
            Frame in BGR format
        """
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _create_frame_metadata(
        self, frame: FrameType, frame_id: int, frame_path: Path
    ) -> Dict[str, Any]:
        """Create metadata for a frame.

        Args:
            frame: Frame to create metadata for
            frame_id: Frame ID
            frame_path: Path to the frame file

        Returns:
            Frame metadata dictionary
        """
        metadata = {
            "frame_id": frame_id,
            "timestamp": frame.timestamp if hasattr(frame, "timestamp") else None,
            "frame_number": frame.frame_id if hasattr(frame, "frame_id") else frame_id,
            "shape": frame.shape if hasattr(frame, "shape") else None,
            "dtype": str(frame.dtype) if hasattr(frame, "dtype") else None,
            "output_path": self._get_relative_path(frame_path),
        }

        # Add any additional metadata from the frame
        if hasattr(frame, "metadata"):
            metadata.update(frame.metadata)

        return metadata

    def _create_video_metadata(self) -> Dict[str, Any]:
        """Create metadata for the video.

        Returns:
            Video metadata dictionary
        """
        return {
            "fps": self.fps,
            "frame_size": self.frame_size,
            "format": self.format,
            "frame_count": self.current_frame,
            "duration": self.current_frame / self.fps if self.fps > 0 else None,
            "frame_name_template": self.frame_name_template,
        }

    def _save_metadata(self) -> None:
        """Save all metadata to JSON file."""
        metadata_path = self.output_dir / self.metadata_file
        metadata = {**self._create_video_metadata(), "frames": self.metadata_list}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def write_frame(self, frame: FrameType) -> None:
        """Write a frame to disk as an image file.

        Args:
            frame: Frame to write (in RGB format)
        """
        if not self.is_open:
            self.open()

        if frame is None:
            raise ValueError("Frame is None")
        elif not isinstance(frame, FrameType):
            raise ValueError(f"Frame is not a FrameType, got: {type(frame)}")
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Image folder does not exist: {self.images_dir}")

        # Generate output path
        output_path = self._get_frame_path(self.current_frame)

        # Convert to BGR and save frame
        bgr_frame = self._convert_to_bgr(frame)
        cv2.imwrite(str(output_path), bgr_frame)

        # Create and store metadata
        metadata = self._create_frame_metadata(frame, self.current_frame, output_path)
        self.metadata_list.append(metadata)

        self.current_frame += 1

        # Save updated metadata
        if self.current_frame % self.saving_freq == 0:
            self._save_metadata()

    def close(self) -> None:
        """Close the writer and release resources."""
        if self.is_open:
            # Ensure metadata is saved
            self._save_metadata()
            self._is_open = False

    @property
    def codec(self) -> str:
        """Get the codec that was actually used for encoding.

        Returns:
            str: The codec used for encoding (always "image" for ImageWriter)
        """
        return "image"
