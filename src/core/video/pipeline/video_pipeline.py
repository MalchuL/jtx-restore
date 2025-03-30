"""Concrete implementation of the video processing pipeline.

This module provides a concrete implementation of the video pipeline using
the existing reader, writer, and processor classes from the codebase.
"""

from abc import abstractmethod
import logging
from pathlib import Path
from typing import Optional, Sequence

from src.core.video.processors.frame import ProcessedFrame
from src.core.video.readers.video_reader import VideoMetadata, VideoReader
from src.core.video.writers.video_writer import VideoWriter
from src.core.video.processors.processor import FrameProcessor
from src.core.video.types import FrameType
from src.core.utils.progress import ProgressBar


class DefaultVideoPipeline:
    """Default implementation of the video processing pipeline.

    This class implements a video processing pipeline that uses the existing
    reader, writer, and processor classes from the codebase.
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        processors: Optional[Sequence[FrameProcessor]] = None,
        batch_size: int = 1,
    ) -> None:
        """Initialize the video pipeline.

        Args:
            input_path: Path to the input video file.
            output_path: Path where the processed video will be saved.
            processors: Optional sequence of frame processors to apply.
            batch_size: Number of frames to process in each batch.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.processors = processors or []
        self.reader: Optional[VideoReader] = None
        self.writer: Optional[VideoWriter] = None
        self.progress_bar: Optional[ProgressBar] = None

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        self.batch_size = batch_size
        self.use_batch_processing = batch_size > 1
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def _create_reader(self, input_path: Path) -> VideoReader:
        """Create a video reader instance.

        Args:
            input_path: Path to the input video file.

        Returns:
            VideoReader: A video reader instance.
        """
        pass

    @abstractmethod
    def _create_writer(self, output_path: Path, metadata: VideoMetadata) -> VideoWriter:
        """Create a video writer instance.

        Args:
            output_path: Path where the processed video will be saved.
            metadata: Video metadata from the reader.

        Returns:
            VideoWriter: A video writer instance.
        """
        pass

    def _create_progress_bar(self, total_frames: int) -> ProgressBar:
        """Create a progress bar instance.

        This method can be overridden by child classes to customize the progress bar.
        By default, it creates a progress bar with statistics panel showing:
        - Frame count and progress
        - Processing speed
        - Time elapsed and estimated time remaining
        - Video metadata

        Args:
            total_frames: Total number of frames to process.

        Returns:
            ProgressBar: A progress bar instance.
        """
        progress = ProgressBar(
            total=total_frames,
            description="Processing video",
            show_time=True,
            show_speed=True,
            show_eta=True,
            show_count=True,
            show_percentage=True,
            refresh_per_second=10,
            transient=False,
            expand=True
        )
        
        # Add initial metadata to stats
        if self.reader:
            metadata = self.reader.metadata
            progress.add_stat("Resolution", f"{metadata.width}x{metadata.height}")
            progress.add_stat("FPS", f"{metadata.fps:.2f}")
            progress.add_stat("Duration", f"{metadata.duration:.2f}s")
            progress.add_stat("Codec", metadata.codec)
        
        return progress

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
        self.reader = self._create_reader(self.input_path)
        self.reader.open()
        
        # Create writer with metadata from reader
        metadata = self.update_metadata(self.reader.metadata)
        self.writer = self._create_writer(
            output_path=self.output_path,
            metadata=metadata
        )
        self.writer.open()
        
        # Create progress bar
        total_frames = int(metadata.frame_count)
        self.progress_bar = self._create_progress_bar(total_frames)
        self.progress_bar.start()
        
        self.logger.info(
            f"Pipeline configured for {metadata.width}x{metadata.height} "
            f"video at {metadata.fps} FPS"
        )

    def process(self) -> None:
        """Process the video through the pipeline."""
        if not self.reader or not self.writer:
            raise RuntimeError("Pipeline not properly set up")

        frame_count = 0
        self.logger.info("Starting video processing...")

        while True:
            frame = self.reader.read_frame()
            if frame is None:
                break

            # Apply all processors in sequence
            processed_frame = ProcessedFrame(frame, frame_id=frame_count)
            for processor in self.processors:
                processed_frame = processor.process_frame(processed_frame)
            self.writer.write_frame(processed_frame.data)
            frame_count += 1

            # Update progress bar
            if self.progress_bar:
                self.progress_bar.update()
                self.progress_bar.add_stat("Frames Processed", frame_count)
                self.progress_bar.add_stat("Processing Speed", f"{frame_count / self.progress_bar.get_elapsed_time():.2f} fps")
                self.progress_bar.update_stats_panel()

        self.logger.info(f"Completed processing {frame_count} frames")

    def finish(self) -> None:
        """Realize resources used by the pipeline."""
        self.logger.info("Realizing pipeline resources...")
        if self.progress_bar:
            self.progress_bar.stop()
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