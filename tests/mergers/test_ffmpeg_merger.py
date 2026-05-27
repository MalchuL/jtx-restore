import pytest
import os
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess

from src.pipelines.merger.ffmpeg_video_audio_merger import FFMPEGVideoAudioMerger

class TestFFMPEGVideoAudioMerger:
    """Tests for the FFMPEGVideoAudioMerger class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample video and audio files for testing."""
        video_path = temp_dir / "test_video.mp4"
        audio_path = temp_dir / "test_audio.wav"
        output_path = temp_dir / "test_output.mp4"
        
        # Create empty files
        video_path.touch()
        audio_path.touch()
        
        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "output_path": output_path,
            "temp_dir": temp_dir
        }
    
    @pytest.fixture
    def mock_subprocess_run(self):
        """Mock the subprocess.run function."""
        with patch("subprocess.run") as mock_run:
            # Create a return value that has the expected attributes
            process_mock = MagicMock()
            process_mock.returncode = 0
            process_mock.stdout = json.dumps({"streams": [], "format": {}})
            mock_run.return_value = process_mock
            yield mock_run
    
    @pytest.fixture
    def mock_subprocess_popen(self):
        """Mock the subprocess.Popen function."""
        with patch("subprocess.Popen") as mock_popen:
            # Create mock process
            process_mock = MagicMock()
            process_mock.returncode = 0
            process_mock.communicate.return_value = ("", "")
            mock_popen.return_value = process_mock
            yield mock_popen
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_init_with_valid_inputs(self, mock_check, sample_files, mock_subprocess_run):
        """Test initialization with valid inputs."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        
        # Act
        merger = FFMPEGVideoAudioMerger(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path
        )
        
        # Assert
        assert merger.video_path == video_path
        assert merger.audio_path == audio_path
        assert merger.output_path == output_path
        assert merger.ffmpeg_path == "ffmpeg"
        assert merger.ffprobe_path == "ffprobe"
        assert merger.extra_args == []
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_init_with_nonexistent_video(self, mock_check, sample_files):
        """Test initialization with nonexistent video file."""
        # Arrange
        video_path = Path("/nonexistent/video.mp4")
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        
        # Act & Assert
        with pytest.raises(FileNotFoundError, match=f"Video file not found: {video_path}"):
            FFMPEGVideoAudioMerger(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path
            ).merge()
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_init_with_nonexistent_audio(self, mock_check, sample_files):
        """Test initialization with nonexistent audio file."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = Path("/nonexistent/audio.wav")
        output_path = sample_files["output_path"]
        
        # Act & Assert
        with pytest.raises(FileNotFoundError, match=f"Audio file not found: {audio_path}"):
            FFMPEGVideoAudioMerger(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path
            ).merge()
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_init_with_extra_args(self, mock_check, sample_files):
        """Test initialization with extra arguments."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        extra_args = ["-shortest", "-vf", "scale=1280:720"]
        
        # Act
        merger = FFMPEGVideoAudioMerger(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            extra_args=extra_args
        )
        
        # Assert
        assert merger.extra_args == extra_args
    
    def test_check_ffmpeg_capability_success(self, sample_files, mock_subprocess_run):
        """Test successful FFMPEG capability check."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        
        # Make the run calls succeed
        mock_subprocess_run.return_value.returncode = 0
        
        with patch("pathlib.Path.exists") as mock_exists:
            # Make file existence checks succeed
            mock_exists.return_value = True
            
            # Act
            with patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.validate_inputs"):
                merger = FFMPEGVideoAudioMerger(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_path=output_path
                )
                result = merger.check_ffmpeg_capability()
            
            # Assert
            assert result is True
            assert mock_subprocess_run.call_count >= 3  # At least 3 FFMPEG commands should be run
    
    def test_check_ffmpeg_capability_failure(self, sample_files, mock_subprocess_run):
        """Test FFMPEG capability check failure."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        
        # Make the run calls fail
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr="FFMPEG error")
        
        # Act & Assert
        with patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.validate_inputs"):
            with pytest.raises(RuntimeError, match="FFMPEG capability check failed"):
                FFMPEGVideoAudioMerger(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_path=output_path
                )
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_merge_success(self, mock_check, sample_files, mock_subprocess_popen):
        """Test successful merge operation."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        
        with patch("pathlib.Path.mkdir"):
            # Act
            merger = FFMPEGVideoAudioMerger(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path
            )
            result = merger.merge()
        
        # Assert
        assert result == output_path
        mock_subprocess_popen.assert_called_once()
        assert "-c:v" in mock_subprocess_popen.call_args[0][0]
        assert "-c:a" in mock_subprocess_popen.call_args[0][0]
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_merge_failure(self, mock_check, sample_files, mock_subprocess_popen):
        """Test merge operation failure."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        
        # Make the subprocess fail
        mock_subprocess_popen.return_value.returncode = 1
        mock_subprocess_popen.return_value.communicate.return_value = ("", "FFMPEG error")
        
        with patch("pathlib.Path.mkdir"):
            # Act & Assert
            merger = FFMPEGVideoAudioMerger(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path
            )
            with pytest.raises(RuntimeError, match="FFMPEG merging failed"):
                merger.merge()
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_probe_media_info_success(self, mock_check, sample_files, mock_subprocess_run):
        """Test successful media info probing."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        
        # Set up return JSON
        mock_info = {
            "streams": [
                {"codec_type": "video", "width": 1280, "height": 720},
                {"codec_type": "audio", "channels": 2, "sample_rate": "44100"}
            ],
            "format": {"duration": "60.0", "size": "10485760"}
        }
        mock_subprocess_run.return_value.stdout = json.dumps(mock_info)
        
        # Act
        merger = FFMPEGVideoAudioMerger(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path
        )
        result = merger.probe_media_info(video_path)
        
        # Assert
        assert result == mock_info
        assert mock_subprocess_run.call_count >= 1
        assert "-show_format" in mock_subprocess_run.call_args[0][0]
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_probe_media_info_nonexistent_file(self, mock_check, sample_files):
        """Test probing a nonexistent file."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        nonexistent_path = Path("/nonexistent/file.mp4")
        
        # Act & Assert
        merger = FFMPEGVideoAudioMerger(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path
        )
        with pytest.raises(FileNotFoundError, match=f"File not found: {nonexistent_path}"):
            merger.probe_media_info(nonexistent_path)
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_build_merge_cmd(self, mock_check, sample_files):
        """Test building merge command."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        
        # Act
        merger = FFMPEGVideoAudioMerger(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path
        )
        cmd = merger._build_merge_cmd(video_path, audio_path, output_path)
        
        # Assert
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert str(video_path) in cmd
        assert str(audio_path) in cmd
        assert "-c:v" in cmd
        assert "-c:a" in cmd
        assert str(output_path) in cmd
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_run_ffmpeg_cmd_success(self, mock_check, sample_files, mock_subprocess_run):
        """Test running FFMPEG command successfully."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        
        # Act
        merger = FFMPEGVideoAudioMerger(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path
        )
        result = merger._run_ffmpeg_cmd(["echo", "test"], "Test command")
        
        # Assert
        assert result == mock_subprocess_run.return_value
        mock_subprocess_run.assert_called_once()
    
    @patch("src.pipelines.merger.ffmpeg_video_audio_merger.FFMPEGVideoAudioMerger.check_ffmpeg_capability")
    def test_run_ffmpeg_cmd_failure(self, mock_check, sample_files, mock_subprocess_run):
        """Test running FFMPEG command with failure."""
        # Arrange
        video_path = sample_files["video_path"]
        audio_path = sample_files["audio_path"]
        output_path = sample_files["output_path"]
        
        # Make the command fail
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr="Command failed")
        
        # Act & Assert
        merger = FFMPEGVideoAudioMerger(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path
        )
        with pytest.raises(RuntimeError, match="Test command failed"):
            merger._run_ffmpeg_cmd(["ffmpeg", "--invalid-flag"], "Test command")