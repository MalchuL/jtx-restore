#!/usr/bin/env python
"""
AI-based frame processors.

This module provides the base AIProcessor class for deep learning-based
video frame processing. It defines the interface and common functionality
for all AI-based processors.
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.processor import FrameProcessor


class AIProcessor(FrameProcessor):
    """Base class for AI-based frame processors.
    
    This processor provides a framework for using deep learning models
    to process video frames. It handles model loading, device management,
    and basic preprocessing/postprocessing.
    
    Subclasses should implement the specific model loading and inference logic.
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 1,
        fill_batch: bool = False
    ):
        """Initialize AI processor.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to run the model on (implementation specific)
            batch_size: Number of frames to process in each batch
            fill_batch: Whether to pad incomplete batches to batch_size
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, batch_size)
        self.fill_batch = fill_batch
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self._is_initialized = False

    @property
    def requires_initialization(self) -> bool:
        """Whether the processor requires explicit initialization.
        
        Returns:
            bool: True since AI processors always need initialization
        """
        return True

    def initialize(self) -> None:
        """Initialize the AI model and processor.
        
        This method should be overridden by subclasses to implement
        specific model loading logic.
        """
        if not self._is_initialized:
            self._load_model()
            self._is_initialized = True

    def _load_model(self) -> None:
        """Load the AI model and processor.
        
        This method should be overridden by subclasses to implement
        specific model loading logic.
        """
        raise NotImplementedError(
            "Subclasses must implement _load_model"
        )

    def _preprocess(self, frame: ProcessedFrame) -> Any:
        """Preprocess a frame for model input.
        
        This method should be overridden by subclasses to implement
        specific preprocessing logic.
        
        Args:
            frame: Input frame to preprocess
            
        Returns:
            Preprocessed data ready for model input
        """
        raise NotImplementedError(
            "Subclasses must implement _preprocess"
        )

    def _postprocess(self, model_output: Any) -> np.ndarray:
        """Postprocess model output into a frame.
        
        This method should be overridden by subclasses to implement
        specific postprocessing logic.
        
        Args:
            model_output: Raw model output to postprocess
            
        Returns:
            Processed frame data as numpy array
        """
        raise NotImplementedError(
            "Subclasses must implement _postprocess"
        )

    def _infer_model(self, inputs: List[Any], *args, **kwargs) -> List[Any]:
        """Run model inference on a batch of inputs.
        
        This method should be overridden by subclasses to implement
        specific model inference logic.
        
        Args:
            inputs: List of preprocessed inputs
            
        Returns:
            List of model outputs
        """
        return self.model(inputs, *args, **kwargs)

    def _process_mini_batch(self, frames: List[ProcessedFrame]) -> List[ProcessedFrame]:
        """Process a mini-batch of frames.
        
        Args:
            frames: List of frames to process
            
        Returns:
            List of processed frames
        """
        # Preprocess all frames in the batch
        inputs = [self._preprocess(frame) for frame in frames]
        
        # Run inference
        outputs = self._infer_model(inputs)
        
        # Postprocess outputs
        results = []
        for i, output in enumerate(outputs):
            result_data = self._postprocess(output)
            results.append(ProcessedFrame(
                data=result_data,
                frame_id=frames[i].frame_id,
                metadata=frames[i].metadata
            ))
        
        return results

    def process_frame(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Process a single frame.
        
        Args:
            frame: The frame to process
            
        Returns:
            The processed frame
        """
        if self.requires_initialization and not self._is_initialized:
            self.initialize()
            
        # Process single frame as a batch of size 1
        result = self._process_mini_batch([frame])[0]
        return result

    def process_batch(self, frames: Sequence[ProcessedFrame]) -> List[ProcessedFrame]:
        """Process a batch of frames.
        
        This method handles batching of frames and processes them in mini-batches
        of size batch_size. If fill_batch is True, it will pad incomplete batches
        by repeating frames from the batch.
        
        Args:
            frames: List of frames to process
            
        Returns:
            List of processed frames
        """
        if not frames:
            return []
            
        if self.requires_initialization and not self._is_initialized:
            self.initialize()
            
        # Convert to list for easier manipulation
        frames_list = list(frames)
        results = []
        
        # Process frames in mini-batches
        for i in range(0, len(frames_list), self.batch_size):
            # Get current mini-batch
            mini_batch = frames_list[i:i + self.batch_size]
            
            # Pad mini-batch if needed
            if self.fill_batch and len(mini_batch) < self.batch_size:
                # Repeat frames from the batch to fill it
                while len(mini_batch) < self.batch_size:
                    mini_batch.extend(mini_batch[:self.batch_size - len(mini_batch)])
            
            # Process mini-batch
            processed_frames = self._process_mini_batch(mini_batch)
            if len(processed_frames) != len(mini_batch):
                raise ValueError(
                    f"Expected {len(mini_batch)} processed frames, got {len(processed_frames)}, data: {processed_frames}"
                )
            
            # Keep only the frames we actually processed
            results.extend(processed_frames[:len(frames_list[i:i + self.batch_size])])
        
        return results 