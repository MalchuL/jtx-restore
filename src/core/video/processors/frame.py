"""
Frame container class for video processors.

This module defines the ProcessorFrame class which encapsulates a video frame
along with its metadata and position information.
"""

from typing import Dict, Any, Optional
import numpy as np


class ProcessorFrame:
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
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a ProcessorFrame object.
        
        Args:
            data: The frame data as a numpy array in RGB format
            frame_id: The position/index of the frame in the video sequence
            metadata: Optional dictionary containing additional frame information
        """
        self.data = data
        self.frame_id = frame_id
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