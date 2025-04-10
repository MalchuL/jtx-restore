"""
Frame interpolation for increasing video frame rate.

This module provides a frame interpolator class that can increase video frame rate
by generating intermediate frames between existing frames using various interpolation methods.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, TypeVar, Generic
import logging

import numpy as np

from src.core.video.utils.frame_cutter import FrameCutter
from src.core.video.frame_interpolation.interpolated_frame import InterpolatedFrame

T = TypeVar('T', bound=InterpolatedFrame)


class StreamingFrameInterpolator(ABC, Generic[T]):
    """
    A streaming processor that increases frame rate by interpolating between frames.

    This abstract base class provides the framework for processing frames one at a time
    and buffering them internally. Concrete implementations should provide specific
    interpolation methods by overriding the _interpolate_window method.
    
    It will return interpolated frames only when they are ready to be consumed,
    making it suitable for streaming applications where frames arrive one at a time.

    Attributes:
        cutter (FrameCutter[T]): Frame cutter for managing frame windows
        factor (float): The frame rate increase factor (e.g., 2 doubles the frame rate)
        _used_frames_count (int): Counter for processed input frames
        finished (bool): Flag indicating whether processing is complete
        logger: Logger instance for this class
    """

    def __init__(self, factor: float = 2):
        """
        Initialize the streaming frame interpolator.

        Args:
            factor: The frame rate increase factor (e.g., 2 doubles the frame rate)
        """
        self.cutter = self._create_cutter()
        self._factor = factor
        self._used_frames_count = 0
        self.finished = False
        self.logger = logging.getLogger(__name__)
        
    @property
    def factor(self) -> float:
        """
        Get the frame rate increase factor.
        
        Returns:
            float: The frame rate increase factor
        """
        return self._factor
    
    @property
    def _interpolated_frame_count(self) -> int:
        """
        Get the number of frames added to the buffer.
        
        Returns:
            int: The number of frames added to the buffer
        """
        return round(self._factor)

    def _create_cutter(self) -> FrameCutter[T]:
        """
        Create a frame cutter instance for managing frame windows.
        
        The default implementation creates a cutter with a window size of 2,
        which is appropriate for interpolation between consecutive frames.
        
        Returns:
            FrameCutter[T]: Configured frame cutter instance
        """
        return FrameCutter[T](
            window_size=2, non_overlap_size=1, begin_non_overlap=0
        )

    @abstractmethod
    def _interpolate_window(
        self, window: Sequence[T]
    ) -> List[T]:
        """
        Interpolate frames within a window to generate intermediate frames.
        
        This abstract method must be implemented by concrete subclasses to provide
        specific interpolation algorithms (e.g., linear, optical flow, etc.).
        
        Args:
            window: Sequence of frames to interpolate between
            
        Returns:
            List[T]: List of interpolated frames
        """
        pass

    def _cut_frames(self, frames: List[T]) -> List[T]:
        """
        Cut the frames to get output frame sequence. Used to process after interpolation to match sequence ordering.
        
        This method extracts the appropriate subset of interpolated frames to maintain
        the correct sequence when multiple interpolation windows are processed.
        
        For example, with factor=2:
        - For frames [1, 2, 3, 4, 5], we interpolate between pairs [1, 2], [2, 3], etc.
        - The interpolator might produce [1, 1.5, 2] and [2, 2.5, 3]
        - We need to extract [1, 1.5] and [2, 2.5] to maintain the sequence
        
        Args:
            frames: List of interpolated frames
            
        Returns:
            List[T]: Subset of frames that maintain the correct sequence
        """
        return frames[: self._interpolated_frame_count]

    def _process_single_window(
        self, window: Sequence[T]
    ) -> List[T]:
        """
        Process a single window of frames.
        
        This method applies interpolation to a window of frames and extracts
        the appropriate subset to maintain the correct sequence.
        
        Args:
            window: Sequence of frames to process
            
        Returns:
            List[T]: Processed frames
        """
        # Interpolate frames. Output will be in the same order as input
        # Ids will be float numbers to interpolate between
        frames_to_interpolate = self._interpolate_window(window)
        # Cut frames to match sequence ordering
        frames_to_interpolate = self._cut_frames(frames_to_interpolate)
        return frames_to_interpolate

    def process_frame(
        self, frame: Optional[T]
    ) -> Optional[List[T]]:
        """
        Process a single frame and return interpolated frames if available.
        
        This method adds the frame to an internal buffer and generates
        interpolated frames when enough frames are available.
        
        Args:
            frame: The input frame to process, or None to process remaining frames
            
        Returns:
            Optional[List[T]]: List of interpolated frames if available, None otherwise
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
            processed_frames = self._process_single_window(frames_to_interpolate)
            for frame in processed_frames:
                if processed_frames[0].height != frame.height or processed_frames[0].width != frame.width:
                    raise ValueError(f"Interpolated frame shape is different from original frame shape, got {frame.shape} and {processed_frames[0].shape}")
                if frame.data is None:
                    raise ValueError("Interpolated frame data is None")
            output_frames.extend(processed_frames)

        return output_frames if output_frames else None

    def __call__(
        self, frame: Optional[T]
    ) -> Optional[List[T]]:
        """
        Process a frame (callable interface).
        
        This method allows the class to be used as a callable function.
        
        Args:
            frame: The input frame to process, or None to process remaining frames
            
        Returns:
            Optional[List[T]]: List of interpolated frames if available, None otherwise
        """
        return self.process_frame(frame)
