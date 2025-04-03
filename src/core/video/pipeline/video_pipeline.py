"""Concrete implementation of the video processing pipeline.

This module provides a concrete implementation of the video pipeline using
the existing reader, writer, and processor classes from the codebase.
"""

from abc import abstractmethod
from collections import deque
import logging
from pathlib import Path
from typing import List, Optional, Sequence

from src.core.video.frame_interpolation.frame_cutter import FrameCutter
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.pipeline import ProcessorPipeline
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
        self.single_frame_processors = ProcessorPipeline(processors or [])
        self.frame_cutter = FrameCutter(window_size=batch_size)
        self.reader: Optional[VideoReader] = None
        self.writer: Optional[VideoWriter] = None

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        self.batch_size = batch_size
        self.use_batch_processing = batch_size > 1
        self.logger = logging.getLogger(__name__)

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

    def _process_frame(
        self, frame: Optional[ProcessedFrame]
    ) -> Optional[List[ProcessedFrame]]:
        """Process a single frame and return a processed frame if available."""
        if frame is None:
            windows = self.frame_cutter.get_remaining_windows()
        else:
            batch = self.frame_cutter.process_frame(frame)
            if batch is None:
                windows = []
            else:
                windows = [batch]  # Single window
        if len(windows) == 0:
            return None

        # Now we process multiple batches, because we have multiple windows at the end
        processed_frames = []
        for batch in windows:
            processed_frames.extend(self.single_frame_processors.process_batch(batch))
        return processed_frames

    def process(self) -> None:
        """Process the video through the pipeline."""
        if not self.reader or not self.writer:
            raise RuntimeError("Pipeline not properly set up")

        frame_count = 0
        self.logger.info("Starting video processing...")

        non_processed_frames = 0
        is_reader_finished = False
        while True:
            # Not all readers can always return None after the end of the video
            # So we need to check if the reader is finished
            if not is_reader_finished:
                frame = self.reader.read_frame()
                if frame is None:
                    is_reader_finished = True
            else:
                frame = None

            # Prepare frame for processing
            frame_to_process = None
            if frame is not None:
                frame_to_process = ProcessedFrame(frame, frame_id=frame_count)
                non_processed_frames += 1

            window = self.frame_cutter.process_frame(frame_to_process)
            if window.finished:
                break

            if frame_count % 100 == 0:
                self.logger.info(
                    f"Processed {frame_count}/{self.reader.metadata.frame_count} frames"
                )
            frame_count += 1
            
            if not window.ready:
                continue
            processed_frames = self.single_frame_processors.process_batch(window.frames)
            processed_frames = processed_frames[:non_processed_frames]
            
            for processed_frame in processed_frames:
                self.writer.write_frame(processed_frame.data)
            non_processed_frames = 0

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
