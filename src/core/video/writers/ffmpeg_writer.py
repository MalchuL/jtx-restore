#!/usr/bin/env python
"""
FFmpeg video writer that uses ImageWriter to write frames to a temporary directory
and then converts them to video using FFmpeg.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple, Union, Dict

import numpy as np

from src.core.video.errors import FFmpegNotInstalledError
from src.core.video.writers.image_writer import ImageWriter
from src.core.video.writers.video_writer import VideoWriter
from src.core.video.types import FrameType


class FFmpegVideoWriter(VideoWriter[FrameType]):
    """Writer that uses FFmpeg to create videos from image sequences.

    This writer first saves frames as images using ImageWriter, then converts
    them to video using FFmpeg. This approach is useful when you need high-quality
    video output with specific FFmpeg settings.
    """

    # Default codecs for common container formats
    DEFAULT_CODECS = {
        ".mp4": "libx264",
        ".avi": "libx264",
        ".mov": "libx264",
        ".mkv": "libx264",
        ".webm": "libvpx-vp9",
    }

    # Codec presets with their default settings
    CODEC_PRESETS = {
        "libx264": {
            "preset": "medium",
            "crf": "23",
            "pix_fmt": "yuv420p",
        },
        "libx265": {
            "preset": "medium",
            "crf": "28",
            "pix_fmt": "yuv420p",
        },
        "libvpx-vp9": {
            "cpu-used": "4",
            "crf": "31",
            "b:v": "0",
        },
        "libvpx": {
            "cpu-used": "4",
            "crf": "31",
            "b:v": "0",
        },
        "mpeg4": {
            "q:v": "5",
        },
        "msmpeg4v2": {
            "q:v": "5",
        },
    }

    # Installation instructions for different operating systems
    INSTALLATION_GUIDE = {
        "linux": """
FFmpeg is not installed. To install FFmpeg on Linux:

Ubuntu/Debian:
    sudo apt update
    sudo apt install ffmpeg

Fedora:
    sudo dnf install ffmpeg

Arch Linux:
    sudo pacman -S ffmpeg

For other distributions, check your package manager or visit:
https://ffmpeg.org/download.html
""",
        "darwin": """
FFmpeg is not installed. To install FFmpeg on macOS:

Using Homebrew:
    brew install ffmpeg

Using MacPorts:
    sudo port install ffmpeg

Or download from:
https://ffmpeg.org/download.html
""",
        "win32": """
FFmpeg is not installed. To install FFmpeg on Windows:

1. Download FFmpeg from: https://ffmpeg.org/download.html
2. Extract the downloaded zip file
3. Add FFmpeg's bin directory to your system's PATH environment variable
4. Restart your terminal/IDE

Alternatively, using Chocolatey:
    choco install ffmpeg

Or using Scoop:
    scoop install ffmpeg
""",
    }

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: Optional[str] = None,
        ffmpeg_args: Optional[list] = None,
        temp_dir: Optional[Union[str, Path]] = None,
        image_format: str = "png",
    ):
        """Initialize the FFmpeg video writer.

        Args:
            output_path: Path where the video will be saved
            fps: Frames per second
            frame_size: Size of video frames as (width, height)
            codec: FFmpeg codec to use (default: None, auto-select based on extension)
            ffmpeg_args: Additional FFmpeg arguments (default: None)
            temp_dir: Directory for temporary image files (default: None, creates temp dir)
            image_format: Format of image files (default: "png")
        Raises:
            FFmpegNotInstalledError: If FFmpeg is not installed on the system
        """
        super().__init__(
            output_path=output_path, fps=fps, frame_size=frame_size, codec=codec
        )

        # Check if FFmpeg is installed
        if not self._is_ffmpeg_installed():
            raise FFmpegNotInstalledError(self._get_installation_guide())

        self.ffmpeg_args = ffmpeg_args or []
        self._temp_dir = Path(temp_dir) if temp_dir else None
        self._image_writer = None
        
        self._image_format = image_format

        # Determine codec if not provided
        if self._codec is None:
            self._codec = self._select_codec()
        else:
            # Verify if the requested codec is available
            if not self._is_codec_available(self._codec):
                raise ValueError(
                    f"Codec '{self._codec}' is not available on this system"
                )

    def _is_ffmpeg_installed(self) -> bool:
        """Check if FFmpeg is installed on the system.

        Returns:
            bool: True if FFmpeg is installed, False otherwise
        """
        return shutil.which("ffmpeg") is not None

    def _get_installation_guide(self) -> str:
        """Get installation instructions for the current operating system.

        Returns:
            str: Installation instructions
        """
        platform = sys.platform
        if platform.startswith("linux"):
            platform = "linux"
        elif platform.startswith("darwin"):
            platform = "darwin"
        elif platform.startswith("win"):
            platform = "win32"
        else:
            platform = "linux"  # Default to Linux instructions

        return self.INSTALLATION_GUIDE[platform]

    @staticmethod
    def _is_codec_available(codec: str) -> bool:
        """Check if a codec is available on the current system.

        Args:
            codec: The codec to check

        Returns:
            bool: True if the codec is available, False otherwise
        """
        try:
            # Run ffmpeg -codecs to get list of available codecs
            result = subprocess.run(
                ["ffmpeg", "-codecs"], capture_output=True, text=True, check=True
            )
            return codec in result.stdout
        except subprocess.CalledProcessError:
            return False

    def _select_codec(self) -> str:
        """Select an appropriate codec based on file extension.

        Returns:
            str: The selected codec

        Raises:
            ValueError: If no suitable codec is found
        """
        extension = self.output_path.suffix.lower()
        if extension in self.DEFAULT_CODECS:
            codec = self.DEFAULT_CODECS[extension]
            if self._is_codec_available(codec):
                return codec

        # Try common codecs as fallback
        for codec in ["libx264", "libx265", "mpeg4"]:
            if self._is_codec_available(codec):
                return codec

        raise ValueError(f"No suitable codec found for {extension} extension")

    def _get_codec_args(self) -> list:
        """Get FFmpeg arguments for the selected codec.

        Returns:
            list: FFmpeg arguments for the codec
        """
        args = ["-c:v", self.codec]

        # Add codec-specific arguments from presets
        if self.codec in self.CODEC_PRESETS:
            preset = self.CODEC_PRESETS[self.codec]
            for key, value in preset.items():
                args.extend([f"-{key}", str(value)])

        return args

    def _initialize(self) -> None:
        """Initialize the writer and create temporary directory if needed."""

        if self._temp_dir is None:
            self._temp_dir = TemporaryDirectory()
            self._temp_dir_path = Path(self._temp_dir.name)
        else:
            self._temp_dir_path = self._temp_dir

        self.logger.info(f"Initializing image writer with output path: {self._temp_dir_path}")
        # Create image writer
        self._image_writer = ImageWriter(
            output_path=self._temp_dir_path,
            fps=self.fps,
            frame_size=self.frame_size,
            format=self._image_format,  # Use PNG for best quality
            frame_name_template="frame_{:08d}.{ext}",
            saving_freq=1000,  # Save metadata more frequently
        )

    def open(self) -> None:
        """Open the writer and prepare for writing frames."""
        if not self._is_open:
            self._initialize()

        self._image_writer.open()
        self._is_open = True

    def close(self) -> None:
        """Close the writer and convert images to video."""
        if not self.is_open:
            return

        # Close image writer
        self._image_writer.close()

        # Convert images to video using FFmpeg
        self._convert_to_video()

        # Clean up temporary directory
        if isinstance(self._temp_dir, TemporaryDirectory):
            self._temp_dir.cleanup()

        self._is_open = False

    def _convert_to_video(self) -> None:
        """Convert image sequence to video using FFmpeg."""
        # Build FFmpeg command
        input_pattern = str(self._temp_dir_path / "images" / "frame_%08d.png")
        output_path = str(self.output_path)

        # Base command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if exists
            "-framerate",
            str(self.fps),
            "-i",
            input_pattern,
        ]

        # Add codec-specific arguments
        cmd.extend(self._get_codec_args())

        # Add additional FFmpeg arguments
        cmd.extend(self.ffmpeg_args)

        # Add output path
        cmd.append(output_path)

        # Run FFmpeg
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e.stderr}")

    def write_frame(self, frame: FrameType) -> None:
        """Write a frame to the video.

        Args:
            frame: Frame to write (in RGB format)
        """
        if not self.is_open:
            self.open()

        if frame is None:
            raise ValueError("Frame is None")

        # Write frame using image writer
        self._image_writer.write_frame(frame)

    @property
    def codec(self) -> str:
        """Get the codec that was actually used for encoding.

        Returns:
            str: The codec used for encoding
        """
        return self._codec
