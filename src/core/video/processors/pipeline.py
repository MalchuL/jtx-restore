#!/usr/bin/env python
"""
Processor pipeline.

This module provides the ProcessorPipeline class for chaining multiple
frame processors together. All processors work with frames in RGB format.
"""

from typing import Dict, List, Optional, Sequence, Union

from src.core.video.processors.frame import ProcessorFrame
from src.core.video.processors.base import FrameProcessor




class ProcessorPipeline(FrameProcessor):
    """A pipeline of frame processors that are applied sequentially.
    
    This class allows chaining multiple FrameProcessor instances together,
    where each processor in the chain receives the output of the previous one.
    All processors in the pipeline expect and return frames in RGB format.
    """
    
    def __init__(self, 
                processors: List[FrameProcessor],
                name: str = "ProcessorPipeline",
                **kwargs):
        """Initialize a processor pipeline.
        
        Args:
            processors (List[FrameProcessor]): List of frame processors to apply in sequence.
            name (str, optional): Name of the pipeline. Defaults to "ProcessorPipeline".
            **kwargs: Additional configuration parameters to pass to the base class.
        
        Raises:
            ValueError: If processors list is empty.
        """
        super().__init__(name=name, **kwargs)
        
        if not processors:
            raise ValueError("Pipeline must contain at least one processor")
            
        self.processors = processors
        self._processor_names = [p.name for p in processors]
        
    @property
    def is_stateful(self) -> bool:
        """Whether any processor in the pipeline is stateful.
        
        Returns:
            bool: True if any processor in the pipeline is stateful, False otherwise.
        """
        return any(processor.is_stateful for processor in self.processors)
    
    @property
    def requires_initialization(self) -> bool:
        """Whether any processor in the pipeline requires initialization.
        
        Returns:
            bool: True if any processor in the pipeline requires initialization,
                 False otherwise.
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
    
    def process_frame(self, frame: ProcessorFrame) -> ProcessorFrame:
        """Process a single frame through the pipeline.
        
        Args:
            frame (ProcessorFrame): Input frame to process in RGB format
        
        Returns:
            ProcessorFrame: Processed frame after going through all processors in RGB format
        """
        result = frame
        for processor in self.processors:
            result = processor.process_frame(result)
        return result
    
    def process_batch(self, frames: Sequence[ProcessorFrame]) -> List[ProcessorFrame]:
        """Process a batch of frames through the pipeline.
        
        Args:
            frames (Sequence[ProcessorFrame]): List of input frames to process in RGB format
        
        Returns:
            List[ProcessorFrame]: List of processed frames after going through all processors in RGB format
        """
        result = list(frames)
        for processor in self.processors:
            result = processor.process_batch(result)
        return result 