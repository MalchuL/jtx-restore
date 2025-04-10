"""Frame processor manager for handling individual and batch frame processing.

This module provides a class for managing frame processing operations,
which can handle single frames and process remaining frames when input is None.
"""

from typing import List, Optional, Sequence
import logging

from src.core.video.utils.frame_cutter import FrameCutter, CutterWindow
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.pipeline import ProcessorPipeline
from src.core.video.processors.processor import FrameProcessor


class FrameProcessorManager:
    """Manages frame processing operations for video pipeline.

    This class handles the logic for processing individual frames and
    processing remaining frames when a frame is None. It simplifies the
    video pipeline implementation by encapsulating the frame cutting and
    processing logic.

    Attributes:
        processors: The processor pipeline to apply to frames
        frame_cutter: The frame cutter to use for batch processing
        batch_size: Number of frames to process in each batch
        frame_count: Number of frames processed so far
        non_processed_frames: Number of frames that haven't been fully processed yet
    """

    def __init__(
        self,
        processors: Sequence[FrameProcessor],
        batch_size: int = 1,
    ) -> None:
        """Initialize the frame processor manager.

        Args:
            processors: Sequence of frame processors to apply
            batch_size: Number of frames to process in each batch
        """
        self.processors = ProcessorPipeline(processors or [])
        if batch_size < 1:
            raise ValueError("Batch size must be greater than 0")
        self.frame_cutter = FrameCutter(window_size=batch_size)
        self.non_processed_frames = 0
        self.is_reader_finished = False
        self.logger = logging.getLogger(__name__)

    def process_frame(
        self, frame: Optional[ProcessedFrame]
    ) -> Optional[List[ProcessedFrame]]:
        if self.is_finished():
            return None
        if frame is None:
            windows = self.frame_cutter.get_remaining_windows()
            self.is_reader_finished = True
        else:
            if not isinstance(frame, ProcessedFrame):
                raise ValueError("Frame must be a ProcessedFrame")
            self.non_processed_frames += 1
            window = self.frame_cutter(frame)
            if not window.ready:
                return None
            windows = [window]
        output_frames = []
        for frame_window in windows:
            processing_frames = frame_window.frames
            processed_frames = self.processors.process_batch(processing_frames)
            processed_frames = processed_frames[:self.non_processed_frames]
            self.non_processed_frames -= len(processed_frames)
            output_frames.extend(processed_frames)
        return output_frames
    
    def is_finished(self) -> bool:
        return self.is_reader_finished and self.non_processed_frames <= 0

    def __call__(self, frame: Optional[ProcessedFrame]) -> Optional[List[ProcessedFrame]]:
        return self.process_frame(frame)
