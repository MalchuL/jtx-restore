#!/usr/bin/env python
"""
Abstract base frame processor.

This module provides the abstract base class for all video frame processors.
All processors work with frames in RGB format, regardless of the input source.
"""

import abc
from typing import List, Optional, Sequence

from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.processor_info import ProcessorInfo
from src.core.video.processors.processor_result import ProcessorResult


class FrameProcessor(abc.ABC):
    """Abstract base class for video frame processors.

    This class defines the interface for processors that enhance or modify
    video frames. Implementations can be stateless (each frame processed
    independently) or stateful (requiring context from previous/future frames).

    All frame processors in this framework expect and return frames in RGB format.
    If a processor needs to work with other color spaces (like BGR for OpenCV
    operations), it should handle the conversion internally.
    """

    def __init__(self):
        """Initialize the frame processor."""
        self._is_initialized = False
        self._finished = False

    @property
    def is_finished(self) -> bool:
        """Check if the processor has finished processing."""
        return self._finished


    def initialize(self) -> None:
        """Initialize the processor if needed.

        This method should be called before processing any frames. It can be used to load models,
        allocate GPU memory, etc.
        """
        self._is_initialized = True

    @abc.abstractmethod
    def _process_frame(self, frame: ProcessedFrame) -> ProcessorResult:
        """Process a single frame.

        Args:
            frame: The input frame to process in RGB format

        Returns:
            The processed frame in RGB format, or None if processing failed
        """
        pass

    def __call__(self, frame: ProcessedFrame) -> ProcessorResult:
        """Process a single frame (callable interface).

        Args:
            frame (ProcessorFrame): The input frame to process in RGB format.

        Returns:
            Optional[List[ProcessorFrame]]: The processed frames in RGB format.
        """
        if not self._is_initialized:
            self.initialize()

        if self.is_finished:
            return ProcessorResult(frames=[], ready=False)

        if frame is None:
            raise ValueError("Frame is None")
        if not isinstance(frame, ProcessedFrame):
            raise ValueError(f"Frame is not a ProcessedFrame, got {type(frame)}")
        result = self._process_frame(frame)
        return result

    def finish(self) -> ProcessorResult:
        """Finish the processing.

        Returns:
            ProcessorResult: The result of the processing. Used to process the remaining frames.
        """
        self._finished = True
        return self._do_finish()

    def _do_finish(self) -> ProcessorResult:
        return ProcessorResult(frames=[], ready=False)

    def update_processor_info(self, processor_info: ProcessorInfo) -> ProcessorInfo:
        return processor_info.copy()
    
    def reset(self) -> None:
        """Reset the processor."""
        self._is_initialized = False
        self._finished = False
