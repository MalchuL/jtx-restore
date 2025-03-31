"""Concrete implementation of the video processing pipeline using OpenCV and FFmpeg.

This module provides a concrete implementation of the video pipeline using
OpenCVVideoReader for reading frames and FFmpegVideoWriter for writing frames.
"""

from pathlib import Path
from typing import Optional, Sequence

from src.core.video.readers.opencv_reader import OpenCVVideoReader
from src.core.video.readers.video_reader import VideoMetadata
from src.core.video.writers.ffmpeg_writer import FFmpegVideoWriter
from src.core.video.processors.processor import FrameProcessor
from src.core.video.pipeline.video_pipeline import DefaultVideoPipeline
from src.core.video.types import FrameType


class OpenCVFFmpegPipeline(DefaultVideoPipeline):
    """Concrete implementation of the video pipeline using OpenCV and FFmpeg.

    This class implements a video processing pipeline that uses OpenCVVideoReader
    for reading frames and FFmpegVideoWriter for writing frames. It provides
    high-quality video processing with support for various codecs and formats.

    Attributes:
        input_path (Path): Path to the input video file
        output_path (Path): Path where the processed video will be saved
        processors (Optional[Sequence[FrameProcessor]]): Optional sequence of frame processors
        reader (Optional[OpenCVVideoReader]): The OpenCV video reader instance
        writer (Optional[FFmpegVideoWriter]): The FFmpeg video writer instance
    """
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        processors: Optional[Sequence[FrameProcessor]] = None,
        batch_size: int = 1,
    ) -> None:
        """Initialize the OpenCV FFmpeg pipeline.

        Args:
            input_path: Path to the input video file
            output_path: Path where the processed video will be saved
            processors: Optional sequence of frame processors to apply
            batch_size: Number of frames to process in each batch
        """
        super().__init__(processors=processors, batch_size=batch_size)
        self.input_path = input_path
        self.output_path = output_path

    def _create_reader(self) -> OpenCVVideoReader:
        """Create an OpenCV video reader instance.

        Returns:
            OpenCVVideoReader: A new OpenCV video reader instance
        """
        return OpenCVVideoReader(self.input_path)

    def _create_writer(self, metadata: VideoMetadata) -> FFmpegVideoWriter:
        """Create an FFmpeg video writer instance.

        Args:
            metadata (VideoMetadata): Metadata from the input video

        Returns:
            FFmpegVideoWriter: A new FFmpeg video writer instance
        """
        return FFmpegVideoWriter(
            output_path=self.output_path,
            fps=metadata.fps,
            frame_size=(metadata.width, metadata.height),
            # Use default codec based on file extension
            codec=None,
            # Use default FFmpeg arguments for high quality
            ffmpeg_args=None,
            # Use PNG format for temporary images to maintain quality
            image_format="png"
        )

    def update_metadata(self, metadata: VideoMetadata) -> VideoMetadata:
        """Update the metadata of the video.

        This implementation preserves the original metadata from the input video.
        Subclasses can override this method to modify metadata for the output video.

        Args:
            metadata (VideoMetadata): The metadata of the video

        Returns:
            VideoMetadata: The updated metadata of the video
        """
        return metadata 