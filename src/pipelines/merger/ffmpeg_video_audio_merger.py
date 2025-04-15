import logging
import os
import subprocess
from pathlib import Path
import tempfile
import random
import shutil
from typing import Optional, Dict, Any, List, Union
import json

from src.pipelines.merger.video_audio_merger import VideoAudioMerger


class FFMPEGVideoAudioMerger(VideoAudioMerger):
    """FFMPEG implementation of video and audio merger.
    
    This class uses ffmpeg to combine video and audio streams into a single output file.
    """
    
    def __init__(
        self,
        video_path: Union[Path, str],
        audio_path: Union[Path, str],
        output_path: Union[Path, str],
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        extra_args: Optional[List[str]] = None
    ):
        """Initialize the FFMPEG video audio merger.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file
            output_path: Path where the merged file will be saved
            ffmpeg_path: Path to the ffmpeg executable
            ffprobe_path: Path to the ffprobe executable
            extra_args: Additional ffmpeg arguments to use during merging
        """
        super().__init__(output_path)
        self.video_path = Path(video_path)
        self.audio_path = Path(audio_path)
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.extra_args = extra_args or []
        self.logger = logging.getLogger(__name__)
        
        
        # Check if FFMPEG can create videos with current configuration
        self.check_ffmpeg_capability()
    
    def _build_test_video_cmd(self, output_path: Path) -> List[str]:
        """Build command to create a test video file.
        
        Args:
            output_path: Path where the test video will be saved
            
        Returns:
            List[str]: Command arguments for ffmpeg
        """
        return [
            self.ffmpeg_path,
            "-f", "lavfi",           # Use libavfilter
            "-i", "color=black:s=320x240:r=30:d=1",  # 1 second black video
            "-c:v", "libx264",       # H.264 codec
            "-tune", "stillimage",
            "-pix_fmt", "yuv420p",   # Compatible pixel format
            str(output_path)
        ]
    
    def _build_test_audio_cmd(self, output_path: Path) -> List[str]:
        """Build command to create a test audio file.
        
        Args:
            output_path: Path where the test audio will be saved
            
        Returns:
            List[str]: Command arguments for ffmpeg
        """
        return [
            self.ffmpeg_path,
            "-f", "lavfi",           # Use libavfilter
            "-i", "anullsrc=r=44100:cl=stereo",  # 1 second silence
            "-t", "1",
            str(output_path)
        ]
    
    def _build_merge_cmd(self, video_path: Path, audio_path: Path, output_path: Path) -> List[str]:
        """Build command to merge video and audio files.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file
            output_path: Path where the merged file will be saved
            
        Returns:
            List[str]: Command arguments for ffmpeg
        """
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]
        
        # Add any extra arguments
        cmd.extend(self.extra_args)        
        # Add output path
        cmd.append(str(output_path))
        
        return cmd
    
    def _run_ffmpeg_cmd(self, cmd: List[str], description: str = "FFMPEG command") -> None:
        """Run an FFMPEG command and handle errors.
        
        Args:
            cmd: Command arguments for ffmpeg
            description: Description of the command for logging
            
        Raises:
            RuntimeError: If the command fails
        """
        self.logger.debug(f"Running {description}: {' '.join(cmd)}")
        
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return process
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else "Unknown error"
            self.logger.error(f"{description} failed: {error_msg}")
            raise RuntimeError(f"{description} failed: {error_msg}")
    
    def check_ffmpeg_capability(self) -> bool:
        """Check if FFMPEG can create videos with the current configuration.
        
        This method creates temporary test files and tries to merge them using
        the configured FFMPEG settings to verify that video creation works.
        
        Returns:
            bool: True if FFMPEG can create videos, False otherwise
            
        Raises:
            RuntimeError: If FFMPEG is not capable of creating videos
        """
        self.logger.info("Testing FFMPEG capability with test files...")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Create test video file (1 second, black screen)
            video_ext = self.video_path.suffix.lower()
            test_video_path = temp_dir_path / f"test_video.{video_ext}"
            video_cmd = self._build_test_video_cmd(test_video_path)
            
            # Create test audio file (1 second, silent)
            audio_ext = self.audio_path.suffix.lower()
            test_audio_path = temp_dir_path / f"test_audio.{audio_ext}"
            audio_cmd = self._build_test_audio_cmd(test_audio_path)
            
            # Test output path
            output_ext = self.output_path.suffix.lower()
            test_output_path = temp_dir_path / f"test_output.{output_ext}"
            
            try:
                # Create test video
                self._run_ffmpeg_cmd(video_cmd, "Test video creation")
                
                # Create test audio
                self._run_ffmpeg_cmd(audio_cmd, "Test audio creation")
                
                # Verify files were created
                if not test_video_path.exists() or not test_audio_path.exists():
                    raise RuntimeError("Failed to create test files")
                
                # Try to merge them
                merge_cmd = self._build_merge_cmd(test_video_path, test_audio_path, test_output_path)
                self._run_ffmpeg_cmd(merge_cmd, "Test merge")
                
                # Verify merged file was created
                if not test_output_path.exists():
                    raise RuntimeError("Failed to create test merged file")
                
                self.logger.info("FFMPEG capability check successful")
                return True
                
            except Exception as e:
                self.logger.error(f"FFMPEG capability check failed: {str(e)}")
                raise RuntimeError(f"FFMPEG capability check failed: {str(e)}")
    
    def validate_inputs(self, video_path: Union[Path, str], audio_path: Union[Path, str]) -> bool:
        """Validate that the input files exist and are of supported formats.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file
            
        Returns:
            bool: True if inputs are valid, False otherwise
            
        Raises:
            FileNotFoundError: If video or audio file doesn't exist
            ValueError: If provided files are invalid or unsupported formats
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        
        # Check if files exist
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        
        return True
    
    def merge(self) -> Path:
        """Merge video and audio files into a single output file using FFMPEG.
        
        Returns:
            Path: Path to the merged output file
            
        Raises:
            FileNotFoundError: If source files don't exist
            ValueError: If provided files are invalid or incompatible
            RuntimeError: If ffmpeg process fails
        """
        # Validate inputs when initializing
        self.validate_inputs(self.video_path, self.audio_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build and run the merge command
        cmd = self._build_merge_cmd(self.video_path, self.audio_path, self._output_path)
        self.logger.info(f"Running ffmpeg merge command: {' '.join(cmd)}")
        
        # Execute ffmpeg command
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr if stderr else "Unknown error"
                self.logger.error(f"FFMPEG error: {error_msg}")
                raise RuntimeError(f"FFMPEG merging failed: {error_msg}")
            
            self.logger.info(f"Successfully merged video and audio to {self._output_path}")
            return self._output_path
            
        except Exception as e:
            self.logger.exception("Error during FFMPEG processing")
            raise RuntimeError(f"Error during FFMPEG processing: {str(e)}")
    
    def probe_media_info(self, file_path: Union[Path, str]) -> Dict[str, Any]:
        """Get media file information using ffprobe.
        
        Args:
            file_path: Path to the media file
            
        Returns:
            Dict: Information about the media file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If ffprobe process fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path)
        ]
        
        try:
            process = self._run_ffmpeg_cmd(cmd, "FFprobe")
            return json.loads(process.stdout)
        except Exception as e:
            raise RuntimeError(f"Error during FFprobe processing: {str(e)}") 