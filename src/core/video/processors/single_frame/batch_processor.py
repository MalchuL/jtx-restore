"""
Frame interpolation for increasing video frame rate.

This module provides a frame interpolator class that can increase video frame rate
by generating intermediate frames between existing frames using various interpolation methods.
"""

from abc import abstractmethod
from typing import List, Optional, Sequence
import logging

import numpy as np

from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.processor import FrameProcessor
from src.core.video.processors.processor_result import ProcessorResult
from src.core.video.utils.frame_cutter import FrameCutter


class BatchProcessor(FrameProcessor):

    def __init__(self, batch_size: Optional[int] = None):
        super().__init__()
        self.batch_size = batch_size
        self._num_frames = 0
        self.cutter = self._create_cutter()
        self.logger = logging.getLogger(__name__)

    def _create_cutter(self) -> FrameCutter[ProcessedFrame]:
        return FrameCutter[ProcessedFrame](
            window_size=self.batch_size,
            non_overlap_size=self.batch_size,
            begin_non_overlap=0,
        )

    def _cut_frames(self, frames: List[ProcessedFrame]) -> List[ProcessedFrame]:
        return frames[: self._num_frames]

    @abstractmethod
    def _process_single_window(
        self, window: Sequence[ProcessedFrame]
    ) -> List[ProcessedFrame]:
        pass

    def _process_frame(self, frame: ProcessedFrame) -> ProcessorResult:
        """Process a single frame.

        Args:
            frame: The input frame to process in RGB format

        Returns:
            The processed frame in RGB format, or None if processing failed
        """
        self._num_frames += 1
        window = self.cutter(frame)
        if not window.ready:
            return ProcessorResult(frames=[], ready=False)
        if len(window.frames) == 0:
            raise ValueError("Window is empty")
        processed_frames = self._process_single_window(window.frames)
        processed_frames = self._cut_frames(processed_frames)
        if len(processed_frames) == 0:
            raise RuntimeError("Output frames are empty")
        self._num_frames = 0
        return ProcessorResult(frames=processed_frames, ready=True)

    def _do_finish(self) -> ProcessorResult:
        remaining_frames = self.cutter.get_remaining_windows()
        if len(remaining_frames) > 1:
            raise RuntimeError("More than one remaining window")
        results = []
        for window in remaining_frames:
            processed_frames = self._process_single_window(window.frames)
            processed_frames = self._cut_frames(processed_frames)
            if len(processed_frames) == 0:
                raise RuntimeError("Output frames are empty")
            results.extend(processed_frames)
        self._num_frames = 0
        return ProcessorResult(frames=results, ready=len(results) > 0)

    def reset(self) -> None:
        super().reset()
        self._num_frames = 0
        self.cutter.reset()
