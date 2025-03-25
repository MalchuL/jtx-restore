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
                 frame_size: Optional[Tuple[int, int]] = None,
                 codec: Optional[str] = None,
                 is_color: bool = True):
        """Initialize the OpenCV video writer.

        Args:
            output_path (Union[str, Path]): Path where the video will be saved
            fps (float): Frames per second
            frame_size (Optional[Tuple[int, int]], optional): Size of video frames as (width, height),
                or None to determine from first frame. Defaults to None.
            codec (Optional[str], optional): Four character codec code (e.g., 'mp4v', 'avc1', 'XVID').
                Defaults to None, which will use default codec based on file extension.
            is_color (bool, optional): Whether the video contains color frames. Defaults to True.

        Raises:
            ValueError: If the specified codec is not available
        """
        super().__init__(output_path, fps, frame_size, codec, is_color)
        self._writer = None

        # Determine codec if not provided
        if self.codec is None:
            self._actual_codec = self._select_codec()
        else:
            # Verify if the requested codec is available
            if self._is_codec_available(self.codec):
                self._actual_codec = self.codec
            else:
                raise ValueError(f"Codec '{self.codec}' is not available on this system")

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
                True
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

    def _ensure_directory_exists(self) -> None:
        """Ensure the output directory exists."""
        os.makedirs(self.output_path.parent, exist_ok=True)

    def open(self) -> None:
        """Open the writer and prepare for writing frames.

        This method initializes the OpenCV VideoWriter with the specified parameters.
        If frame_size was not provided at initialization, the first frame written will
        determine the size.

        Raises:
            IOError: If the writer cannot be opened with the specified codec
        """
        if self.is_open:
            return

        self._ensure_directory_exists()

        if self.frame_size is None:
            self._is_open = True
            return

        try:
            fourcc = cv2.VideoWriter_fourcc(*self._actual_codec)
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.frame_size,
                self.is_color
            )

            if not self._writer.isOpened():
                raise IOError(f"Failed to open VideoWriter for {self.output_path} with codec {self._actual_codec}")

            self._is_open = True

        except Exception as e:
            raise IOError(f"Error opening VideoWriter: {e}")

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
        If the writer hasn't been initialized yet (when frame_size was None),
        it will initialize the writer with the size of the first frame.

        Args:
            frame (FrameType): The frame to write in RGB format (numpy array)

        Raises:
            IOError: If the writer cannot be opened with the specified codec
        """
        if not self.is_open:
            self.open()

        if frame is None:
            return

        # Initialize writer with first frame if not already done
        if self._writer is None:
            height, width = frame.shape[:2]

            # Ensure even dimensions (required by some codecs)
            if width % 2 == 1:
                width -= 1
            if height % 2 == 1:
                height -= 1

            self.frame_size = (width, height)

            fourcc = cv2.VideoWriter_fourcc(*self._actual_codec)
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.frame_size,
                self.is_color
            )

            if not self._writer.isOpened():
                raise IOError(f"Failed to open VideoWriter for {self.output_path} with codec {self._actual_codec}")

        # Check if frame needs resizing
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write the frame
        self._writer.write(frame_bgr)

    @property
    def output_codec(self) -> str:
        """Get the codec that was actually used for encoding.

        Returns:
            str: The codec used for encoding
        """
        return self._actual_codec
