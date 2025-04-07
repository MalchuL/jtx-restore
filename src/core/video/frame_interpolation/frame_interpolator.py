"""
Frame interpolation for increasing video frame rate.

This module provides a frame interpolator class that can increase video frame rate
by generating intermediate frames between existing frames using optical flow.
"""

from abc import abstractmethod
from typing import List, Optional, Sequence
import logging

from src.core.video.utils.frame_cutter import FrameCutter
from src.core.video.frame_interpolation.interpolated_frame import InterpolatedFrame


class StreamingFrameInterpolator:
    """
    A streaming processor that increases frame rate by interpolating between frames.

    This class is designed to process frames one at a time and buffer them internally.
    It will return interpolated frames only when they are ready to be consumed,
    making it suitable for streaming applications where frames arrive one at a time.

    Attributes:
        interpolator (FrameInterpolator): The underlying frame interpolator
        buffer (List[InterpolatedFrame]): Buffer for storing generated frames
        frame_count (int): Counter for input frames
        output_count (int): Counter for output frames
    """

    def __init__(self, factor: int = 2):
        """
        Initialize the streaming frame interpolator.

        Args:
            factor: The frame rate increase factor (e.g., 2 doubles the frame rate)
            method: Interpolation method ('linear', 'flow', or 'blend')
            use_gpu: Whether to use GPU acceleration if available
        """
        self.cutter = self._create_cutter()
        self.factor = factor
        self._used_frames_count = 0
        self.finished = False
        self.logger = logging.getLogger(__name__)

    def _create_cutter(self) -> FrameCutter[InterpolatedFrame]:
        return FrameCutter[InterpolatedFrame](
            window_size=2, non_overlap_size=1, begin_non_overlap=0
        )

    @abstractmethod
    def _interpolate_window(
        self, window: Sequence[InterpolatedFrame]
    ) -> List[InterpolatedFrame]:
        pass

    def _cut_frames(self, frames: List[InterpolatedFrame]) -> List[InterpolatedFrame]:
        """
        Cut the frames to get output frame sequence. Used to process after interpolation to match sequence ordering.
        By default we need to cut frames more that factor times.
        For example for frames [1, 2, 3, 4, 5] and factor 2 we need to return
        We will have two frames [1, 2] and [2, 3].
        Interpolated can be [1, 1.5, 2] and [2, 2.5, 3] for factor 2.
        Or [1, 1.33, 1.66, 2] or [2, 2.33, 2.66, 3] for factor 3.
        We need to cut First frames:
        [1, 1.5] or [1, 1.33, 1.66] and [2, 2.5] or [2, 2.33, 2.66]
        And then merge them.
        """
        return frames[: self.factor]

    def _process_single_window(
        self, window: Sequence[InterpolatedFrame]
    ) -> List[InterpolatedFrame]:
        """
        Process a single window of frames.
        """

        # Interpolate frames. Output will be in the same order as input
        # Ids will be float numbers to interpolate between
        frames_to_interpolate = self._interpolate_window(window)
        # Cut frames to match sequence ordering
        frames_to_interpolate = self._cut_frames(frames_to_interpolate)
        return frames_to_interpolate

    def process_frame(
        self, frame: Optional[InterpolatedFrame]
    ) -> Optional[List[InterpolatedFrame]]:
        """
        Process a single frame and return a processed frame if available.

        Args:
            frame: The input frame to process

        Returns:
            Optional[InterpolatedFrame]: A interpolated frame if available, None otherwise
        """
        if frame is None:
            windows = self.cutter.get_remaining_windows()
            self.finished = True
        else:
            # Cut frames to buffer
            window = self.cutter(frame)
            self._used_frames_count += 1
            if not window.ready:
                return None
            windows = [window]
        output_frames = []
        for interpolation_window in windows:
            # We have frames to interpolate
            frames_to_interpolate = interpolation_window.frames
            frames_to_interpolate = self._process_single_window(frames_to_interpolate)
            output_frames.extend(frames_to_interpolate)

        return frames_to_interpolate

    def __call__(
        self, frame: Optional[InterpolatedFrame]
    ) -> Optional[List[InterpolatedFrame]]:
        """
        Process a frame (callable interface).

        Args:
            frame: The input frame to process, or None to finish

        Returns:
            Optional[List[InterpolatedFrame]]: A list of interpolated frames if available, None otherwise
        """
        return self.process_frame(frame)
