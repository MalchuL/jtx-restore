#!/usr/bin/env python
"""
Processor pipeline.

This module provides the ProcessorPipeline class for chaining multiple
frame processors together. All processors work with frames in RGB format.
"""

from typing import List, Sequence

from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.processor import FrameProcessor


class ProcessorPipeline(FrameProcessor):
    """A pipeline of frame processors that are applied sequentially.
    
    This class allows chaining multiple FrameProcessor instances together,
    where each processor in the chain receives the output of the previous one.
    All processors in the pipeline expect and return frames in RGB format.
    """
    
    def __init__(self, processors: List[FrameProcessor]):
        """Initialize a processor pipeline.
        
        Args:
            processors: List of frame processors to apply in sequence.
            
        Raises:
            ValueError: If processors list is empty.
        """
        super().__init__()
        
        if not processors:
            raise ValueError("Pipeline must contain at least one processor")
            
        self.processors = processors
    
    @property
    def requires_initialization(self) -> bool:
        """Whether any processor in the pipeline requires initialization.
        
        Returns:
            bool: True if any processor requires initialization, False otherwise.
        """
        return any(processor.requires_initialization for processor in self.processors)
    
    def initialize(self) -> None:
        """Initialize all processors in the pipeline that require initialization."""
        for processor in self.processors:
            if processor.requires_initialization:
                processor.initialize()
    
    def reset_state(self) -> None:
        """Reset the state of all stateful processors in the pipeline."""
        for processor in self.processors:
            if processor.is_stateful:
                processor.reset_state()
    
    def process_frame(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Process a single frame through the pipeline.
        
        Args:
            frame: Input frame to process in RGB format
        
        Returns:
            Processed frame after going through all processors in RGB format
        """
        result = frame
        for processor in self.processors:
            result = processor(result)
        return result
    
    def process_batch(self, frames: Sequence[ProcessedFrame]) -> List[ProcessedFrame]:
        """Process a batch of frames through the pipeline.
        
        Args:
            frames: List of input frames to process in RGB format
        
        Returns:
            List of processed frames after going through all processors in RGB format
        """
        if not frames:
            return []
            
        result = list(frames)
        for processor in self.processors:
            result = processor.process_batch(result)
        return result 