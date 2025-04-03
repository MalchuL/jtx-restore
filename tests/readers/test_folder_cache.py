import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil
import os

from src.core.video.readers.opencv_reader import OpenCVVideoReader
from src.core.video.readers.folder_cache import FolderCacheReader


@pytest.fixture
def temp_video_path():
    """Create a temporary video file for testing."""
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
    
    # Cleanup
    if temp_video.exists():
        os.unlink(temp_video)
    os.rmdir(temp_dir)


@pytest.fixture
def base_reader(temp_video_path):
    """Create an OpenCVVideoReader for testing."""
    reader = OpenCVVideoReader(temp_video_path)
    yield reader
    if reader.is_open:
        reader.close()


@pytest.fixture
def cache_dir():
    """Create a temporary directory for the cache."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def cache_reader(base_reader, cache_dir):
    """Create a FolderCacheReader for testing."""
    reader = FolderCacheReader(
        base_reader=base_reader,
        cache_folder=cache_dir,
        image_format="png"
    )
    yield reader
    if reader.is_open:
        reader.close()


class TestFolderCacheReader:
    """Tests for the FolderCacheReader class."""
    
    def test_initialization(self, base_reader, cache_dir):
        """Test initializing the reader."""
        reader = FolderCacheReader(
            base_reader=base_reader,
            cache_folder=cache_dir,
            image_format="png"
        )
        
        assert reader.base_reader == base_reader
        assert reader.cache_folder == cache_dir
        assert reader.image_format == "png"
        assert reader.frame_name_template == "frame_{:08d}.{ext}"
        assert not reader.is_open
        assert not reader.is_finished
        assert cache_dir.exists()  # Cache folder should be created
    
    def test_custom_params(self, base_reader, cache_dir):
        """Test initializing with custom parameters."""
        reader = FolderCacheReader(
            base_reader=base_reader,
            cache_folder=cache_dir,
            image_format="jpg",
            frame_name_template="custom_{:04d}.{ext}"
        )
        
        assert reader.image_format == "jpg"
        assert reader.frame_name_template == "custom_{:04d}.{ext}"
    
    def test_metadata_passthrough(self, cache_reader, base_reader):
        """Test that metadata is correctly passed through from the base reader."""
        assert cache_reader.metadata == base_reader.metadata
        assert cache_reader.frame_count == base_reader.frame_count
        assert cache_reader.fps == base_reader.fps
        assert cache_reader.duration == base_reader.duration
        assert cache_reader.resolution == base_reader.resolution
    
    def test_open_close(self, cache_reader):
        """Test opening and closing the reader."""
        assert not cache_reader.is_open
        assert not cache_reader.is_finished
        
        cache_reader.open()
        assert cache_reader.is_open
        assert not cache_reader.is_finished
        assert cache_reader.base_reader.is_open
        
        cache_reader.close()
        assert not cache_reader.is_open
        assert cache_reader.is_finished
        assert not cache_reader.base_reader.is_open
    
    def test_is_finished_property(self, cache_reader):
        """Test the is_finished property behavior."""
        # Initially not finished
        assert not cache_reader.is_finished
        
        # Open the reader
        cache_reader.open()
        assert not cache_reader.is_finished
        
        # Read all frames
        while True:
            frame = cache_reader.read_frame()
            if frame is None:
                break
        
        # Should be finished after reading all frames
        assert cache_reader.is_finished
        
        # Reset should set is_finished to False
        cache_reader.reset()
        assert not cache_reader.is_finished
        
        # Set frame index should set is_finished to False
        cache_reader.set_frame_index(5)
        assert not cache_reader.is_finished
        
        # Close should set is_finished to True
        cache_reader.close()
        assert cache_reader.is_finished
    
    def test_frame_caching(self, cache_reader, cache_dir):
        """Test that frames are cached correctly."""
        # Read the first frame
        frame = cache_reader.read_frame()
        
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        
        # Check that the frame was cached
        cached_files = list(cache_dir.glob("*.png"))
        assert len(cached_files) == 1
        
        # Get the cached frame path
        cached_frame_path = cache_reader.get_frame_path(0)
        assert cached_frame_path.exists()
        
        # Check that frame_is_cached works
        assert cache_reader.frame_is_cached(0)
        assert not cache_reader.frame_is_cached(100)
        
        # Test loading from cache
        cached_frame = cache_reader.load_frame_from_cache(0)
        assert cached_frame is not None
        assert cached_frame.shape == (480, 640, 3)
        # Compare with original frame
        assert np.array_equal(frame, cached_frame)
    
    def test_read_frame_from_cache(self, cache_reader, cache_dir):
        """Test reading a frame that's already cached."""
        # Read a frame to cache it
        first_frame = cache_reader.read_frame()
        
        # Close and reopen to ensure we're reading from cache
        cache_reader.close()
        cache_reader.open()
        
        # Read the same frame again
        second_frame = cache_reader.read_frame()
        
        # Frames should be identical
        assert np.array_equal(first_frame, second_frame)
        
        # Only one file should exist in the cache
        cached_files = list(cache_dir.glob("*.png"))
        assert len(cached_files) == 1
    
    def test_read_frames(self, cache_reader, cache_dir):
        """Test reading multiple frames."""
        # Read 5 frames
        frames = cache_reader.read_frames(5)
        
        assert len(frames) == 5
        
        # Check cache contents
        cached_files = list(cache_dir.glob("*.png"))
        assert len(cached_files) == 5
        
        # Check that the frames are cached
        for i in range(5):
            assert cache_reader.frame_is_cached(i)
    
    def test_get_frame_at_index(self, cache_reader, cache_dir):
        """Test getting a frame at a specific index."""
        # Get frame at index 5 (should cache it)
        frame = cache_reader.get_frame_at_index(5)
        
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        
        # Check that only this frame was cached
        cached_files = list(cache_dir.glob("*.png"))
        assert len(cached_files) == 1
        assert cache_reader.frame_is_cached(5)
        
        # Get the same frame again (should read from cache)
        frame_again = cache_reader.get_frame_at_index(5)
        
        # Frames should be identical
        assert np.array_equal(frame, frame_again)
        
        # Still only one file in cache
        cached_files = list(cache_dir.glob("*.png"))
        assert len(cached_files) == 1
    
    def test_get_frame_at_timestamp(self, cache_reader):
        """Test getting a frame at a specific timestamp."""
        # Get frame at timestamp 0.2 seconds (should be around frame 6)
        frame = cache_reader.get_frame_at_timestamp(0.2)
        
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        
        # Should have cached a frame
        assert cache_reader.frame_is_cached(int(0.2 * 30))
    
    def test_yield_frames(self, cache_reader, cache_dir):
        """Test yielding frames in chunks."""
        # Yield frames in chunks of 3
        chunks = list(cache_reader.yield_frames(3))
        
        # Should yield 4 chunks: 3, 3, 3, 1
        assert len(chunks) == 4
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3
        assert len(chunks[3]) == 1
        
        # All frames should be cached
        cached_files = list(cache_dir.glob("*.png"))
        assert len(cached_files) == 10
        
        # Reader should be finished after yielding all frames
        assert cache_reader.is_finished
    
    def test_clear_cache(self, cache_reader, cache_dir):
        """Test clearing the cache."""
        # Read all frames to cache them
        _ = list(cache_reader.yield_frames(5))
        
        # Check that frames were cached
        cached_files = list(cache_dir.glob("*.png"))
        assert len(cached_files) == 10
        
        # Clear the cache
        cache_reader.clear_cache()
        
        # Check that cache is empty
        cached_files = list(cache_dir.glob("*.png"))
        assert len(cached_files) == 0
    
    def test_set_frame_index(self, cache_reader, cache_dir):
        """Test setting the current position to a specific frame index."""
        # First, check that setting a valid frame index works
        success = cache_reader.set_frame_index(5)
        assert success
        
        # Read the frame at the current position (which should now be index 5)
        frame = cache_reader.read_frame()
        assert frame is not None
        
        # Convert to int before comparison
        r_value = int(frame[0, 0, 0])
        b_value = int(frame[0, 0, 2])
        
        # At frame index 5, the blue channel (now red in RGB) should be around 100
        assert 195 <= r_value <= 205
        assert 95 <= b_value <= 105
        
        # Check that the frame was cached
        assert cache_reader.frame_is_cached(5)
        
        # Now read the next frame (which should be index 6)
        frame = cache_reader.read_frame()
        assert frame is not None
        
        # Convert to int before comparison
        b_value = int(frame[0, 0, 2])
        # At frame index 6, the blue channel (now red in RGB) should be around 120
        assert 115 <= b_value <= 125
        
        # Test setting an invalid frame index (negative)
        success = cache_reader.set_frame_index(-1)
        assert not success
        
        # Test setting an invalid frame index (too large)
        success = cache_reader.set_frame_index(100)
        assert not success
    
    def test_set_frame_timestamp(self, cache_reader, cache_dir):
        """Test setting the current position to a specific timestamp."""
        # Set position to 0.2 seconds (around frame 6 with 30 fps)
        success = cache_reader.set_frame_timestamp(0.2)
        assert success
        
        # Read the frame at the current position
        frame = cache_reader.read_frame()
        assert frame is not None
        
        # Convert to int before comparison
        b_value = int(frame[0, 0, 2])
        
        # Around 0.2 seconds with 30fps should be close to frame 6
        # Frame 6 should have blue channel (now red in RGB) around 120
        assert 110 <= b_value <= 130
        
        # Check that the frame was cached
        frame_idx = int(0.2 * 30)  # Calculate expected frame index
        assert cache_reader.frame_is_cached(frame_idx)
        
        # Test setting an invalid timestamp (negative)
        success = cache_reader.set_frame_timestamp(-1.0)
        assert not success
        
        # Test setting an invalid timestamp (too large)
        success = cache_reader.set_frame_timestamp(10.0)
        assert not success
    
    def test_current_index(self, cache_reader):
        """Test getting the current frame index."""
        # At the beginning, index should be 0
        assert cache_reader.current_index == 0
        
        # After reading one frame, index should be 1
        frame = cache_reader.read_frame()
        assert cache_reader.current_index == 1
        
        # After setting frame index to 5, index should be 5
        cache_reader.set_frame_index(5)
        assert cache_reader.current_index == 5
        
        # After reading another frame, index should be 6
        frame = cache_reader.read_frame()
        assert cache_reader.current_index == 6
    
    def test_reset(self, cache_reader, cache_dir):
        """Test resetting the reader position."""
        # Read a few frames to advance the position and cache them
        for _ in range(5):
            cache_reader.read_frame()
        
        # Position should be at frame 5
        assert cache_reader.current_index == 5
        
        # Reset should return true and set position to 0
        success = cache_reader.reset()
        assert success
        assert cache_reader.current_index == 0
        assert not cache_reader.is_finished
        
        # After reset, we should be able to read all frames again
        frames = []
        while True:
            frame = cache_reader.read_frame()
            if frame is None:
                break
            frames.append(frame)
        
        # Should have read all frames
        assert len(frames) >= 9  # Allow for MP4 compression differences
        
        # And position should be at the end
        assert cache_reader.current_index >= 9
        assert cache_reader.is_finished
        
        # Check that all frames were cached
        cached_files = list(cache_dir.glob("*.png"))
        assert len(cached_files) >= 9  # At least 9 frames should be cached
    
    def test_custom_frame_template(self, base_reader, cache_dir):
        """Test using a custom frame_name_template."""
        # Create reader with custom frame name template
        custom_template = "video_frame_{:04d}.{ext}"
        reader = FolderCacheReader(
            base_reader=base_reader,
            cache_folder=cache_dir,
            image_format="jpg",  # Also test different format
            frame_name_template=custom_template
        )
        
        # Read a few frames to cache them
        frames = []
        for _ in range(3):
            frame = reader.read_frame()
            frames.append(frame)
        
        # Check that frames were cached with the correct naming pattern
        cached_files = list(cache_dir.glob("*.jpg"))
        assert len(cached_files) == 3
        
        # Verify the specific filenames match our template
        for i in range(3):
            expected_filename = f"video_frame_{i:04d}.jpg"
            assert (cache_dir / expected_filename).exists()
            
            # Verify frame can be loaded from cache
            cached_frame = reader.load_frame_from_cache(i)
            assert cached_frame is not None
            assert np.allclose(frames[i], cached_frame, rtol=0.01, atol=5)
        
        # Test that clear_cache works with the custom template
        reader.clear_cache()
        
        # Check that cache is empty
        cached_files = list(cache_dir.glob("*.jpg"))
        assert len(cached_files) == 0
        
        # Close reader
        reader.close() 