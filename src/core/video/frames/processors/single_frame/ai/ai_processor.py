#!/usr/bin/env python
"""
AI-based frame processors.

This module provides the base AIProcessor class for deep learning-based
video frame processing. It defines the interface and common functionality
for all AI-based processors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.single_frame.batch_processor import BatchProcessor


class AIProcessor(BatchProcessor, ABC):
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
    ):
        """Initialize AI processor.

        Args:
            model_name: Name or path of the model to load
            device: Device to run the model on (implementation specific)
            batch_size: Number of frames to process in each batch
        """
        super().__init__(batch_size=batch_size)

        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, batch_size)

        # Initialize model and processor
        self._is_model_loaded = False


    def initialize(self) -> None:
        """Initialize the AI model and processor.

        This method should be overridden by subclasses to implement
        specific model loading logic.
        """
        if not self._is_model_loaded:
            self._load_model()
            self._is_model_loaded = True
        super().initialize()

    @abstractmethod
    def _load_model(self) -> None:
        """Load the AI model and processor.

        This method should be overridden by subclasses to implement
        specific model loading logic.
        """

    @abstractmethod
    def _preprocess(self, frame: ProcessedFrame) -> Any:
        """Preprocess a frame for model input.

        This method should be overridden by subclasses to implement
        specific preprocessing logic.

        Args:
            frame: Input frame to preprocess

        Returns:
            Preprocessed data ready for model input
        """

    @abstractmethod
    def _postprocess(self, model_output: Any) -> np.ndarray:
        """Postprocess model output into a frame.

        This method should be overridden by subclasses to implement
        specific postprocessing logic.

        Args:
            model_output: Raw model output to postprocess

        Returns:
            Processed frame data as numpy array
        """

    @abstractmethod
    def _infer_model(self, inputs: List[Any], *args, **kwargs) -> List[Any]:
        """Run model inference on a batch of inputs.

        This method should be overridden by subclasses to implement
        specific model inference logic.

        Args:
            inputs: List of preprocessed inputs

        Returns:
            List of model outputs
        """

    def _process_single_batch(
        self, batch: Sequence[ProcessedFrame]
    ) -> List[ProcessedFrame]:
        """Process a single batch of frames.

        Args:
            batch: The batch of frames to process

        Returns:
            The processed frames in the same order as the input frames
        """

        if len(batch) != self.batch_size:
            raise RuntimeError(
                f"Batch frames are not equal to batch size {len(batch)} != {self.batch_size}"
            )
        # Preprocess all frames in the batch
        inputs = [self._preprocess(frame) for frame in batch]

        # Run inference
        outputs = self._infer_model(inputs)

        # Postprocess outputs
        results = []
        for i, output in enumerate(outputs):
            result_data = self._postprocess(output)
            results.append(
                ProcessedFrame(
                    data=result_data,
                    frame_id=batch[i].frame_id,
                    metadata=batch[i].metadata,
                )
            )

        return results
