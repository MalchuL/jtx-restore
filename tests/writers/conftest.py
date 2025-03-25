import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="function")
def temp_output_dir():
    """Create a temporary directory for output videos.
    
    This fixture provides a temporary directory that will be cleaned up
    after the test.
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def sample_frame():
    """Create a sample frame for testing.
    
    This fixture creates an RGB gradient frame where:
    - Red increases horizontally (0-255 from left to right)
    - Green increases vertically (0-255 from top to bottom)
    - Blue is constant at 128
    """
    # Create a simple gradient frame in RGB format
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # RGB gradient
    for y in range(480):
        for x in range(640):
            frame[y, x, 0] = int(x * 255 / 640)  # R
            frame[y, x, 1] = int(y * 255 / 480)  # G
            frame[y, x, 2] = 128  # B
    return frame


@pytest.fixture(scope="function")
def sample_frames():
    """Create a sequence of sample frames for testing.
    
    This fixture creates 10 RGB frames where:
    - Red increases with each frame
    - Green increases vertically
    - Blue increases horizontally
    """
    # Create 10 frames with different patterns
    frames = []
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Make red value dependent on frame index
        r_value = i * 25  # 0, 25, 50, ..., 225
        for y in range(480):
            for x in range(640):
                frame[y, x, 0] = r_value  # R increases with each frame
                frame[y, x, 1] = int(y * 255 / 480)  # G increases vertically
                frame[y, x, 2] = int(x * 255 / 640)  # B increases horizontally
        frames.append(frame)
    return frames 