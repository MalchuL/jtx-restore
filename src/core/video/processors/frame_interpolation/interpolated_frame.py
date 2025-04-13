"""
Interpolated frame container class for frame interpolation.

This module defines the InterpolatedFrame class which encapsulates a video frame
along with its metadata and position information for frame interpolation.
"""

from typing import Dict, Any, Optional
import numpy as np

from src.core.video.processors.frame import ProcessedFrame


class InterpolatedFrame(ProcessedFrame):
    """
    Container for video frames with associated metadata for interpolation.

    This class provides a container for frame data along with its
    frame_id and any additional metadata specifically tailored for
    frame interpolation operations.

    Note:
        Frame data is stored in RGB format, while OpenCV functions typically
        use BGR format. Implementations should handle the conversion when needed.

    Attributes:
        data (np.ndarray): The frame data as a numpy array
        frame_id (float): The position/index of the frame, can be fractional
        metadata (Dict[str, Any]): Additional frame information
    """

    def __init__(
        self,
        data: np.ndarray,
        frame_id: int,
        dt: float = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an InterpolatedFrame object.

        Args:
            data: The frame data as a numpy array in RGB format
            frame_id: The position/index of the frame in the video sequence
                     (can be fractional for interpolated frames)
            dt: The time difference between the current and previous frame
            metadata: Optional dictionary containing additional frame information
        """
        super().__init__(data, frame_id, metadata)
        assert 0 <= dt < 1, f"dt must be in range [0, 1), got {dt}"
        metadata = metadata or {}
        metadata["dt"] = dt
        self.metadata = metadata

    @property
    def dt(self) -> float:
        return self.metadata["dt"]

    def __repr__(self):
        return f"InterpolatedFrame(frame_id={self.frame_id}, shape={self.shape}, dtype={self.dtype}, dt={self.dt})"
