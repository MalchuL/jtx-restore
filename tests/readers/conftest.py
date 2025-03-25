import pytest
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
import shutil


@pytest.fixture(scope="module")
def temp_video_path():
    """Create a temporary video file for testing.
    
    This fixture creates a test video with 10 frames, where each frame has
    an increasing blue channel value (which will be red in RGB format).
    """
    temp_dir = tempfile.mkdtemp()
    temp_video = Path(temp_dir) / "test_video.mp4"
    
    # Create a small test video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_video), fourcc, 30.0, (640, 480))
    
    # Create 10 frames with increasing color
    for i in range(10):
        # Create a colored frame (BGR format, but will be converted to RGB in reader)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 20  # B - this will be R in RGB format
        frame[:, :, 1] = 100     # G
        frame[:, :, 2] = 200     # R - this will be B in RGB format
        out.write(frame)
    
    out.release()
    
    yield temp_video
    
    # # Cleanup
    # if temp_video.exists():
    #     os.unlink(temp_video)
    # os.rmdir(temp_dir)


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory.
    
    This fixture provides a temporary directory that will be cleaned up
    after the test.
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir) 