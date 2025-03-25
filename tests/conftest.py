"""Common fixtures for all tests."""
import pytest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
@pytest.fixture(scope="session", autouse=True)
def add_project_root_to_path():
    """Add the project root directory to the Python path."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(os.path.dirname(__file__)) / "data"

@pytest.fixture(scope="session")
def sample_video_path(test_data_dir):
    """Return the path to a sample video file for testing."""
    video_path = test_data_dir / "sample.mp4"
    
    # Skip tests if the video file doesn't exist
    if not video_path.exists():
        pytest.skip(f"Test video not found at {video_path}")
    
    return video_path

@pytest.fixture(scope="session")
def corrupt_video_path(test_data_dir):
    """Return the path to a corrupt or non-existent video file for testing error handling."""
    return test_data_dir / "non_existent.mp4" 