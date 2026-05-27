"""
Frame interpolation for increasing video frame rate.

This module provides a frame interpolator class that can increase video frame rate
by generating intermediate frames between existing frames using various interpolation methods.
"""

from abc import abstractmethod
from typing import List, Optional, Sequence
import logging

import numpy as np

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.processor import FrameProcessor
from src.core.video.frames.processors.processor_result import ProcessorResult
from src.core.video.frames.utils.frame_cutter import FrameCutter


class BatchProcessor(FrameProcessor):

    def __init__(self, batch_size: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self._num_frames = 0
        self._cutter = self._create_cutter()
        self.logger = logging.getLogger(__name__)

    def _create_cutter(self) -> FrameCutter[ProcessedFrame]:
        return FrameCutter[ProcessedFrame](
            window_size=self.batch_size,
            non_overlap_size=self.batch_size,
            begin_non_overlap=0,
        )

    def _cut_frames(self, frames: List[ProcessedFrame], cut_processed_frames: bool = True) -> List[ProcessedFrame]:
        """Cut the frames to get output frame sequence. Used to process after interpolation to match sequence ordering.

        Args:
            frames: List of frames to cut
            cut_processed_frames: Whether to cut the processed frames or batch size

        Returns:
            List of frames
        """
        slice_end = self._num_frames if cut_processed_frames else self.batch_size
        return frames[0: slice_end]
    @abstractmethod
    def _process_single_batch(
        self, batch: Sequence[ProcessedFrame]
    ) -> List[ProcessedFrame]:
        """Process a single batch of frames.

        Args:
            batch: The batch of frames to process in the same order as the input frames and always the same size as the batch size

        Returns:
            The processed frames in the same order as the input frames
        """

    def _process_frame(self, frame: ProcessedFrame) -> ProcessorResult:
        """Process a single frame.

        Args:
            frame: The input frame to process in RGB format

        Returns:
            The processed frame in RGB format, or None if processing failed
        """
        self._num_frames += 1
        batch = self._cutter(frame)
        if not batch.ready:
            return ProcessorResult(frames=[], ready=False)
        if len(batch.frames) == 0:
            raise ValueError("Batch is empty")
        if len(batch.frames) != self.batch_size:
            raise RuntimeError(
                f"Batch frames are not equal to batch size {len(batch.frames)} != {self.batch_size}"
            )
        processed_frames = self._process_single_batch(batch.frames)
        if len(processed_frames) != self.batch_size:
            raise RuntimeError(
                f"Processed frames are not equal to batch size {len(processed_frames)} != {self.batch_size}"
            )
        processed_frames = self._cut_frames(processed_frames)
        if len(processed_frames) == 0:
            raise RuntimeError("Output frames are empty")
        self._num_frames = max(0, self._num_frames - self.batch_size)
        return ProcessorResult(frames=processed_frames, ready=True)

    def _do_finish(self) -> ProcessorResult:
        remaining_frames = self._cutter.get_remaining_windows()
        results = []
        for i, batch in enumerate(remaining_frames):
            is_last = i == len(remaining_frames) - 1
            processed_frames = self._process_single_batch(batch.frames)
            if len(processed_frames) != self.batch_size:
                raise RuntimeError(
                    f"Processed frames are not equal to batch size {len(processed_frames)} != {self.batch_size}"
                )
            processed_frames = self._cut_frames(processed_frames, is_last)
            self._num_frames = max(0, self._num_frames - self.batch_size)
            if len(processed_frames) == 0:
                raise RuntimeError("Output frames are empty")
            results.extend(processed_frames)
        self._num_frames = 0
        return ProcessorResult(frames=results, ready=len(results) > 0)

    def reset(self) -> None:
        super().reset()
        self._num_frames = 0
        self._cutter.reset()
