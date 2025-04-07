"""
Frame container class for video interpolation.

This module defines the InterpolatedFrame class which encapsulates a video frame
along with its metadata and position information.
"""

from typing import Dict, Any, Optional
import numpy as np


class InterpolatedFrame:
    """
    Container for video frames with associated metadata.
    
    This class provides a simple container for frame data along with its 
    frame_id and any additional metadata. It explicitly exposes the data
    as a property to ensure transparent understanding for users.
    
    Note:
        Frame data is stored in RGB format, while OpenCV functions typically
        use BGR format. Processor implementations should handle the conversion
        when needed.
    """
    
    def __init__(
        self, 
        data: np.ndarray, 
        frame_id: int,
        dt: float = 0, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a InterpolatedFrame object.
        
        Args:
            data: The frame data as a numpy array in RGB format
            frame_id: The position/index of the frame in the video sequence, 
                can be float number for interpolated frames
            metadata: Optional dictionary containing additional frame information
        """
        self.data = data
        self.frame_id = frame_id
        self.dt = dt
        assert 0 <= dt < 1, f"dt must be in range [0, 1), got {dt}"
        self.metadata = metadata or {}
    
    @property
    def shape(self) -> tuple:
        """Return the shape of the underlying frame data in [H, W, C] format."""
        return self.data.shape
    
    @property
    def height(self) -> int:
        """Return the height of the underlying frame data."""
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        """Return the width of the underlying frame data."""
        return self.data.shape[1]
    
    @property   
    def channels(self) -> int:
        """Return the number of channels in the underlying frame data."""
        return self.data.shape[2]
    
    @property
    def dtype(self):
        """Return the data type of the underlying frame data."""
        return self.data.dtype 
    
    def __repr__(self):
        return f"InterpolatedFrame(frame_id={self.frame_id}, shape={self.shape}, dtype={self.dtype})"
    
    def __str__(self):
        return self.__repr__()
    
    def __eq__(self, other):
        return self.frame_id == other.frame_id and np.array_equal(self.data, other.data)
