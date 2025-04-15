"""Frame processor manager for handling individual and batch frame processing.

This module provides a class for managing frame processing operations,
which can handle single frames and process remaining frames when input is None.
"""

from typing import List, Optional, Sequence
import logging

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.processor import FrameProcessor
from src.core.video.frames.processors.processor_info import ProcessorInfo

class FrameProcessorManager:
    """Manages frame processing operations for video pipeline.

    This class handles the logic for processing individual frames and
    processing remaining frames when a frame is None. It simplifies the
    video pipeline implementation by encapsulating the frame cutting and
    processing logic.

    Attributes:
        processors: The processor pipeline to apply to frames
    """

    def __init__(
        self,
        processors: Sequence[FrameProcessor],
    ):
        """Initialize the frame processor manager.

        Args:
            processors: Sequence of frame processors to apply
        """
        self.processors = processors or []
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Initialize the frame processor manager."""
        for processor in self.processors:
            processor.initialize()

    def reset(self) -> None:
        """Reset the frame processor manager."""
        for processor in self.processors:
            processor.reset()

    @property
    def is_finished(self) -> bool:
        """Check if the frame processor manager is finished."""
        return all(processor.is_finished for processor in self.processors)

    def _run_frames(
        self, frames: List[ProcessedFrame], processor_id: int, do_finish: bool = False
    ) -> List[ProcessedFrame]:
        # If all processors have been run, return the frames
        if processor_id >= len(self.processors):
            return frames

        processor = self.processors[processor_id]
        # If we need to finish the processor, get the remaining frames
        results = []
        # Run the next processor
        self.logger.debug(
            f"Running processor {processor_id} with {len(frames)} frames {do_finish}"
        )
        for frame in frames:
            result = processor(frame)
            if result.ready:
                processor_results = self._run_frames(
                    result.frames, processor_id + 1, False
                )
                # If last processor, return the frames we will extend
                results.extend(processor_results)
        if do_finish:
            last_result = processor.finish()
            if last_result.ready:
                remaining_frames = last_result.frames
            else:
                remaining_frames = []
            # We need to run the next processor with the remaining frames also if no remaining frames
            remaining_results = self._run_frames(
                remaining_frames, processor_id + 1, True
            )
            results.extend(remaining_results)
        return results

    def process_frame(
        self, frame: Optional[ProcessedFrame]
    ) -> Optional[List[ProcessedFrame]]:
        """Process a single frame.

        Args:
            frame: The frame to process, or None to finish the pipeline

        Returns:
            The processed frames if available, otherwise None
        """
        if len(self.processors) == 0:
            return [frame]
        if frame is None:
            frames = []
            do_finish = True
        else:
            frames = [frame]
            do_finish = False
        results = self._run_frames(frames, 0, do_finish)
        return results

    def __call__(
        self, frame: Optional[ProcessedFrame]
    ) -> Optional[List[ProcessedFrame]]:
        return self.process_frame(frame)

    def get_processor_info(self) -> ProcessorInfo:
        """Get the processor info."""
        processor_info = ProcessorInfo()
        for processor in self.processors:
            processor_info = processor.update_processor_info(processor_info)
        return processor_info
