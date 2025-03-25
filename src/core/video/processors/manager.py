#!/usr/bin/env python
"""
Batch processing manager for efficient frame processing.

This module provides utilities for batch processing of video frames,
with optional threading support for improved performance.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Any, Dict, Callable

from src.core.video.processors.base import FrameProcessor
from src.core.video.processors.frame import ProcessedFrame



class BatchProcessingManager:
    """Manager for efficient batch processing of video frames.
    
    This class handles the logistics of processing frames in batches,
    with optional threading support for improved performance on multi-core systems.
    """
    
    def __init__(
        self,
        processor: FrameProcessor,
        batch_size: int = 1,
        max_workers: int = 4,
        use_threading: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the batch processing manager.
        
        Args:
            processor (FrameProcessor): The frame processor to apply to frames
            batch_size (int, optional): Size of frame batches. Defaults to 1.
            max_workers (int, optional): Maximum number of worker threads. Defaults to 4.
            use_threading (bool, optional): Whether to use threading. Defaults to True.
            logger (Optional[logging.Logger], optional): Logger for messages. Defaults to None.
        """
        self.processor = processor
        self.batch_size = max(1, batch_size)
        self.max_workers = max(1, max_workers)
        self.use_threading = use_threading
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize processor if needed
        if self.processor.requires_initialization and not self.processor.is_initialized:
            self.processor.initialize()
        
    def process_frames(self, frames: List[ProcessedFrame]) -> List[ProcessedFrame]:
        """Process a batch of frames using the configured processor.
        
        Args:
            frames (List[ProcessorFrame]): List of frames to process
            
        Returns:
            List[ProcessorFrame]: List of processed frames
        """
        if not frames:
            return []
            
        if len(frames) == 1 or not self.use_threading:
            # Process sequentially for single frame or if threading is disabled
            return self._process_frames_sequential(frames)
        else:
            # Process in parallel with threading
            return self._process_frames_threaded(frames)
    
    def _process_frames_sequential(self, frames: List[ProcessedFrame]) -> List[ProcessedFrame]:
        """Process frames sequentially without threading.
        
        Args:
            frames (List[ProcessorFrame]): List of frames to process
            
        Returns:
            List[ProcessorFrame]: List of processed frames
        """
        processed_frames = []
        
        for frame in frames:
            # Apply processor to each frame
            processed_frame = self.processor(frame)
            processed_frames.append(processed_frame)
            
        return processed_frames
    
    def _process_frames_threaded(self, frames: List[ProcessedFrame]) -> List[ProcessedFrame]:
        """Process frames in parallel using threading.
        
        Args:
            frames (List[ProcessorFrame]): List of frames to process
            
        Returns:
            List[ProcessorFrame]: List of processed frames
        """
        # Cannot use threading for stateful processors since state might be shared
        if self.processor.is_stateful:
            self.logger.warning("Cannot use threading for stateful processor, falling back to sequential processing")
            return self._process_frames_sequential(frames)
        
        # Determine optimal number of workers based on batch size
        num_workers = min(self.max_workers, len(frames))
        
        # Process frames in parallel using ThreadPoolExecutor
        processed_frames_dict = {}  # Maps frame_id to processed frame
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all processing tasks
            future_to_frame_id = {
                executor.submit(self._process_single_frame, frame): frame.frame_id
                for frame in frames
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_frame_id):
                frame_id = future_to_frame_id[future]
                try:
                    processed_frame = future.result()
                    processed_frames_dict[frame_id] = processed_frame
                except Exception as exc:
                    self.logger.error(f"Processing frame {frame_id} generated an exception: {exc}")
                    # Keep the original frame in case of error
                    processed_frames_dict[frame_id] = next(f for f in frames if f.frame_id == frame_id)
        
        # Ensure frames are returned in the original order
        return [processed_frames_dict[frame.frame_id] for frame in frames]
    
    def _process_single_frame(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Process a single frame with the processor.
        
        This is a helper method for threaded processing.
        
        Args:
            frame (ProcessorFrame): Frame to process
            
        Returns:
            ProcessorFrame: Processed frame
        """
        return self.processor(frame) 