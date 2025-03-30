import cv2
import os
from pathlib import Path
from typing import Optional, Tuple, Union

from src.core.video.types import FrameType
from src.core.video.writers.video_writer import VideoWriter


class OpenCVVideoWriter(VideoWriter):
    """Video writer implementation using OpenCV.

    This class provides a concrete implementation of the VideoWriter interface
    using OpenCV's backend for encoding video files.
    """

    # Default codecs for common container formats
    DEFAULT_CODECS = {
        ".mp4": "mp4v",
        ".avi": "XVID",
        ".mov": "mp4v",
        ".wmv": "WMV2",
        ".mkv": "mp4v",
        ".webm": "VP90",  # WebM format typically uses VP8/VP9 codec
    }

    def __init__(self,
                 output_path: Union[str, Path],
                 fps: float,
                 frame_size: Optional[Tuple[int, int]],
                 codec: Optional[str] = None,
                 resize_frames: bool = False):
        """Initialize the OpenCV video writer.

        Args:
            output_path (Union[str, Path]): Path where the video will be saved
            fps (float): Frames per second
            frame_size (Optional[Tuple[int, int]]): Size of video frames as (width, height).
                Must be provided at initialization.
            codec (Optional[str], optional): Four character codec code (e.g., 'mp4v', 'avc1', 'XVID').
                Defaults to None, which will use default codec based on file extension.
            resize_frames (bool, optional): Whether to resize frames to the specified frame_size.
                Defaults to True.
        Raises:
            ValueError: If frame_size is not provided or if the specified codec is not available
            IOError: If the video file cannot be created
        """
        super().__init__(output_path, fps, frame_size, codec)
        
        if self.frame_size is None:
            raise ValueError("frame_size must be provided at initialization")
            
        # Ensure output directory exists
        os.makedirs(self.output_path.parent, exist_ok=True)
        
        # Determine codec if not provided
        if self._codec is None:
            self._codec = self._select_codec()
        else:
            # Verify if the requested codec is available
            if not self._is_codec_available(self._codec):
                raise ValueError(f"Codec '{self._codec}' is not available on this system")
        
        self.resize_frames = resize_frames
        # Initialize the video writer
        self._initialize_writer()
        
    def _is_codec_available(self, codec: str) -> bool:
        """Check if a codec is available on the current system.

        Args:
            codec: The codec fourcc code to check

        Returns:
            bool: True if the codec is available, False otherwise
        """
        try:
            # Try to initialize a writer with this codec
            fourcc = cv2.VideoWriter_fourcc(*codec)
            temp_file = Path(os.path.join(os.path.dirname(str(self.output_path)), f"_codec_test_{codec}.mp4"))
            test_writer = cv2.VideoWriter(
                str(temp_file),
                fourcc,
                30.0,
                (320, 240),
                True  # Always use color for testing
            )

            is_available = test_writer.isOpened()
            test_writer.release()

            # Clean up temp file if it was created
            if temp_file.exists():
                os.unlink(temp_file)

            return is_available
        except:
            return False

    def _select_codec(self) -> str:
        """Select an appropriate codec based on file extension.

        Returns:
            str: The selected codec fourcc code
        """
        extension = self.output_path.suffix.lower()
        if extension in self.DEFAULT_CODECS:
            return self.DEFAULT_CODECS[extension]
        else:
            raise ValueError(f"Automatic codec selection failed. "
                             f"Please specify a valid codec for {extension} extension manually.")

    def _initialize_writer(self) -> None:
        """Initialize the OpenCV VideoWriter.
        
        Raises:
            IOError: If the writer cannot be initialized
        """
        try:
            # Ensure even dimensions (required by some codecs)
            width, height = self.frame_size
            if width % 2 == 1:
                width -= 1
            if height % 2 == 1:
                height -= 1
            self.frame_size = (width, height)
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.frame_size,
                True  # Always use color
            )

            if not self._writer.isOpened():
                raise IOError(f"Failed to open VideoWriter for {self.output_path} with codec {self.codec}")

        except Exception as e:
            raise IOError(f"Error initializing VideoWriter: {e}")

    def open(self) -> None:
        """Open the writer and prepare for writing frames.
        
        This method is a no-op since the writer is initialized in __init__.
        """
        if not self._is_open:
            self._initialize_writer()
            self._is_open = True

    def close(self) -> None:
        """Close the writer and finalize the video file."""
        if not self.is_open:
            return

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        # Verify the video was created properly
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            raise IOError(f"Failed to create valid video file at {self.output_path}")

        self._is_open = False

    def write_frame(self, frame: FrameType) -> None:
        """Write a single frame to the video.

        This method handles RGB to BGR conversion for OpenCV compatibility.

        Args:
            frame (FrameType): The frame to write in RGB format (numpy array)

        Raises:
            ValueError: If frame dimensions don't match the initialized frame_size
        """
        if not self.is_open:
            raise IOError("Writer is not open")
        if not os.path.exists(self.output_path):
            raise IOError(f"Video file does not exist: {self.output_path}")

        if not isinstance(frame, FrameType):
            raise ValueError(f"Frame is not a FrameType, got: {type(frame)}")
        # Check if frame needs resizing
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            self.logger.warning(f"Resizing frame from {frame.shape[1]}x{frame.shape[0]} to {self.frame_size}")
            if self.resize_frames:
                frame = cv2.resize(frame, self.frame_size)
            else:
                raise ValueError("Frame dimensions do not match the initialized frame_size. "
                                 "If you want to resize the frames, set resize_frames to True.")

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write the frame
        self._writer.write(frame_bgr)

    @property
    def codec(self) -> str:
        """Get the codec that was actually used for encoding.

        Returns:
            str: The codec used for encoding
        """
        return self._codec
