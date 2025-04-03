#!/usr/bin/env python
"""
Parallel frame processor.

This module provides the ParallelProcessor class for processing video frames
in parallel using multiple CPU cores. All processors work with frames in RGB format.
"""

from typing import List, Optional, Sequence, Callable, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import functools

from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.processor import FrameProcessor


class ParallelProcessor(FrameProcessor):
    """Frame processor that supports parallel processing using multiprocessing.

    This class provides a framework for processing frames in parallel using
    multiple processes, which is suitable for CPU-bound operations on
    stateless processors. Works with frames in RGB format.
    """

    def __init__(self, num_workers: Optional[int] = None):
        """Initialize the parallel processor.

        Args:
            num_workers: Number of worker processes to use. If None or 0,
            pipeline will process frames sequentially. If positive, uses the
            specified number of worker processes. If negative, uses CPU count - 1.
        """
        super().__init__()
        if num_workers is None:
            num_workers = 0
        elif num_workers < 0:
            num_workers = max(1, mp.cpu_count() - 1)
        self.num_workers = num_workers
        self._executor = None

    def _get_executor(self) -> ProcessPoolExecutor:
        """Get or create a process pool executor.

        Returns:
            ProcessPoolExecutor: The executor instance
        """
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.num_workers)
        return self._executor

    def process_frame(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Process a single frame.

        Args:
            frame: The frame to process in RGB format

        Returns:
            The processed frame in RGB format, or None if processing failed
        """
        # For single frames, use sequential processing
        return super().process_frame(frame)

    def _process_sequential(
        self, frames: Sequence[ProcessedFrame]
    ) -> List[ProcessedFrame]:
        """Process a batch of frames sequentially.

        Args:
            frames: List of input frames to process in RGB format

        Returns:
            List of processed frames in RGB format
        """
        return super().process_batch(frames)

    def _get_parallel_kwargs(self, frame: ProcessedFrame) -> Dict[str, Any]:
        """Generate kwargs for parallel processing.
        
        This method should be overridden by subclasses to provide their specific
        parameters for parallel processing. The default implementation returns
        an empty dictionary.
        
        Args:
            frame: The frame to process
            
        Returns:
            Dictionary of keyword arguments for parallel processing
        """
        return {}

    @staticmethod
    def _process_frame_parallel(frame: ProcessedFrame, **kwargs) -> ProcessedFrame:
        """Process a single frame in a worker process.
        
        This static method is the entry point for parallel processing.
        It should be overridden by subclasses to implement their specific
        processing logic.
        
        Args:
            frame: The frame to process
            **kwargs: Additional parameters needed for processing
            
        Returns:
            The processed frame
        """
        raise NotImplementedError(
            "Subclasses must implement _process_frame_parallel"
        )

    def process_batch(self, frames: Sequence[ProcessedFrame]) -> List[ProcessedFrame]:
        """Process a batch of frames in parallel.

        Args:
            frames: List of input frames to process in RGB format

        Returns:
            List of processed frames in RGB format
        """
        if not frames:
            return []

        # For small batches or single worker, use sequential processing
        if 0 <= self.num_workers <= 1 or len(frames) < 3:
            return self._process_sequential(frames)

        # Process frames in parallel
        executor = self._get_executor()
        
        # Submit all frames for parallel processing with their kwargs
        futures = [
            executor.submit(
                self._process_frame_parallel,
                frame,
                **self._get_parallel_kwargs(frame)
            )
            for frame in frames
        ]

        # Gather results
        results = []
        for future in futures:
            result = future.result()
            if result is None:
                raise ValueError(f"{self.__class__.__name__} returned None.")


        return results

    def __del__(self):
        """Clean up the process pool executor."""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=True)
