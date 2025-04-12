"""Concrete implementation of the video processing pipeline.

This module provides a concrete implementation of the video pipeline using
the existing reader, writer, and processor classes from the codebase.
"""

from abc import abstractmethod
from collections import deque
import logging
from pathlib import Path
from typing import List, Optional, Sequence

from src.core.video.processors.frame_processor_manager import FrameProcessorManager
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.readers.video_reader import VideoMetadata, VideoReader
from src.core.video.writers.video_writer import VideoWriter
from src.core.video.processors.processor import FrameProcessor
from src.core.video.types import FrameType


class DefaultVideoPipeline:
    """Default implementation of the video processing pipeline.

    This class implements a video processing pipeline that uses the existing
    reader, writer, and processor classes from the codebase.
    """

    def __init__(
        self,
        processors: Optional[Sequence[FrameProcessor]] = None,
        batch_size: int = 1,
    ) -> None:
        """Initialize the video pipeline.

        Args:
            processors: Optional sequence of frame processors to apply.
            batch_size: Number of frames to process in each batch.
        """
        self.frame_processors = FrameProcessorManager(processors or [])
        self.reader: Optional[VideoReader] = None
        self.writer: Optional[VideoWriter] = None

        self.logger = logging.getLogger(__name__)
        
        # We need to know if the reader is finished to know when to stop processing
        self._is_reader_finished = False

    @abstractmethod
    def _create_reader(self) -> VideoReader:
        """Create a video reader instance.

        Returns:
            VideoReader: A video reader instance.
        """
        pass

    @abstractmethod
    def _create_writer(self, metadata: VideoMetadata) -> VideoWriter:
        """Create a video writer instance.

        Args:
            metadata: Video metadata from the reader.

        Returns:
            VideoWriter: A video writer instance.
        """
        pass

    def update_metadata(self, metadata: VideoMetadata) -> VideoMetadata:
        """Update the metadata of the video.

        Args:
            metadata (VideoMetadata): The metadata of the video.

        Returns:
            VideoMetadata: The updated metadata of the video.
        """
        return metadata

    def setup(self) -> None:
        """Set up the pipeline components."""
        self.logger.info("Setting up video pipeline...")

        # Initialize reader and writer
        self.reader = self._create_reader()
        self.reader.open()

        # Create writer with metadata from reader
        metadata = self.update_metadata(self.reader.metadata)
        self.writer = self._create_writer(metadata=metadata)
        self.writer.open()

        self.logger.info(
            f"Pipeline configured for {metadata.width}x{metadata.height} "
            f"video at {metadata.fps} FPS"
        )

    def __get_frame(self) -> Optional[ProcessedFrame]:
        # Not all readers can always return None after the end of the video
            # So we need to check if the reader is finished
        if not self._is_reader_finished:
            frame = self.reader.read_frame()
            if frame is None:
                self._is_reader_finished = True
        else:
            frame = None
        return frame

    def process(self) -> None:
        """Process the video through the pipeline."""
        if not self.reader or not self.writer:
            raise RuntimeError("Pipeline not properly set up")

        frame_count = 0
        self.logger.info("Starting video processing...")

        while True:
            
            frame = self.__get_frame()
            if frame_count % 50 == 0:
                self.logger.info(
                    f"Processed {frame_count}/{self.reader.metadata.frame_count} frames"
                )
            if frame is not None:
                processed_frame = ProcessedFrame(frame, frame_id=frame_count)
                frame_count += 1
            else:
                processed_frame = None
            
            # Prepare frame for processing
            processed_window = self.frame_processors(processed_frame)
            if processed_window is None:
                continue
            
            for processed_frame in processed_window:
                self.writer.write_frame(processed_frame.data)
            
            if self.frame_processors.is_finished:
                break
        self.logger.info(f"Completed processing {frame_count} frames")

    def finish(self) -> None:
        """Realize resources used by the pipeline."""
        self.logger.info("Realizing pipeline resources...")
        if self.reader:
            self.reader.close()
        if self.writer:
            self.writer.close()
        self.logger.info("Realization complete")

    def run(self) -> None:
        """Run the complete video processing pipeline.

        This method executes the pipeline in the correct order:
        1. Setup
        2. Process
        3. Finish
        """
        try:
            self.setup()
            self.process()
        finally:
            self.finish()
