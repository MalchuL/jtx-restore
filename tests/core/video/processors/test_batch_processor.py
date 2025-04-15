import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from typing import List, Sequence

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.single_frame.batch_processor import BatchProcessor
from src.core.video.frames.processors.single_frame.parallel_processor import ParallelProcessor


class MockBatchProcessor(BatchProcessor):
    """Concrete implementation for testing BatchProcessor."""
    
    def _process_single_batch(self, batch: Sequence[ProcessedFrame]) -> List[ProcessedFrame]:
        return list(batch)


class MockParallelProcessor(ParallelProcessor):
    """Concrete implementation for testing ParallelProcessor."""
    
    @staticmethod
    def _process_frame_parallel(frame: ProcessedFrame, **kwargs) -> ProcessedFrame:
        return frame


@pytest.fixture
def mock_frame():
    """Create a mock frame for testing."""
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    return ProcessedFrame(data=data, frame_id=0)


@pytest.fixture
def batch_processor():
    """Create a BatchProcessor instance for testing."""
    return MockBatchProcessor(batch_size=3)


@pytest.fixture
def parallel_processor():
    """Create a ParallelProcessor instance for testing."""
    return MockParallelProcessor(num_workers=2)


def test_batch_processor_initialization(batch_processor):
    """Test BatchProcessor initialization."""
    assert batch_processor.batch_size == 3
    assert batch_processor._num_frames == 0
    assert not batch_processor.is_finished


def test_batch_processor_process_frame(batch_processor, mock_frame):
    """Test processing a single frame with BatchProcessor."""
    result = batch_processor(mock_frame)
    assert not result.ready
    assert len(result.frames) == 0


def test_batch_processor_process_multiple_frames(batch_processor, mock_frame):
    """Test processing multiple frames with BatchProcessor."""
    frames = [ProcessedFrame(np.zeros((100, 100, 3), dtype=np.uint8), i) for i in range(5)]
    
    results = []
    for frame in frames:
        result = batch_processor(frame)
        if result.ready:
            results.extend(result.frames)
    assert len(results) == 3
    results.extend(batch_processor.finish().frames)
    assert len(results) == 5
    for i, frame in enumerate(frames):
        assert results[i].frame_id == frame.frame_id


def test_batch_processor_finish(batch_processor, mock_frame):
    """Test finish method of BatchProcessor."""
    # Process some frames
    batch_processor(mock_frame)
    batch_processor(mock_frame)
    
    # Call finish
    result = batch_processor.finish()
    assert batch_processor.is_finished
    assert len(result.frames) == 2
    assert result.ready
    

def test_batch_processor_finish_without_remaining_frames(batch_processor, mock_frame):
    """Test finish method of BatchProcessor."""
    # Process some frames
    batch_processor(mock_frame)
    batch_processor(mock_frame)
    batch_processor(mock_frame)
    
    # Call finish
    result = batch_processor.finish()
    assert len(result.frames) == 0
    assert not result.ready
    assert batch_processor.is_finished


def test_parallel_processor_initialization(parallel_processor):
    """Test ParallelProcessor initialization."""
    assert parallel_processor.num_workers == 2
    assert parallel_processor.batch_size == 2
    assert not parallel_processor.is_finished


@patch('concurrent.futures.ProcessPoolExecutor')
def test_parallel_processor_process_frame(mock_executor, parallel_processor, mock_frame):
    """Test processing a single frame with ParallelProcessor."""
    # Mock the executor and its submit method
    mock_future = MagicMock()
    mock_future.result.return_value = mock_frame
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
    
    result = parallel_processor(mock_frame)
    assert not result.ready
    assert len(result.frames) == 0

    result = parallel_processor(mock_frame)
    assert result.ready
    assert len(result.frames) == 2
    assert result.frames[0] == mock_frame


@patch('concurrent.futures.ProcessPoolExecutor')
def test_parallel_processor_process_multiple_frames(mock_executor, parallel_processor):
    """Test processing multiple frames with ParallelProcessor."""
    # Create test frames
    frames = [ProcessedFrame(np.zeros((100, 100, 3), dtype=np.uint8), i) for i in range(4)]
    
    # Mock the executor and its submit method
    mock_future = MagicMock()
    mock_future.result.side_effect = frames
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
    
    results = []
    for frame in frames:
        result = parallel_processor(frame)
        if result.ready:
            results.extend(result.frames)
    
    assert len(results) == 4
    for i, frame in enumerate(frames):
        assert results[i].frame_id == frame.frame_id


def test_parallel_processor_cleanup(parallel_processor):
    """Test cleanup of ParallelProcessor resources."""
    # Initialize the processor
    parallel_processor._get_executor()
    assert parallel_processor._executor is not None
    
    # Trigger cleanup
    del parallel_processor
    
    # The executor should be shut down
    # Note: This is a bit tricky to test directly, but we can verify the cleanup method was called
    # In a real scenario, you might want to use a context manager or explicit cleanup method


def test_parallel_processor_sequential_processing(parallel_processor, mock_frame):
    """Test sequential processing when workers <= 1."""
    # Create a processor with 1 worker
    processor = MockParallelProcessor(num_workers=1)
    
    result = processor(mock_frame)
    assert result.ready
    assert len(result.frames) == 1
    assert result.frames[0] == mock_frame


def test_parallel_processor_small_batch_processing(parallel_processor):
    """Test processing when number of frames is small."""
    frames = [ProcessedFrame(np.zeros((100, 100, 3), dtype=np.uint8), i) for i in range(2)]
    
    results = []
    for frame in frames:
        result = parallel_processor(frame)
        if result.ready:
            results.extend(result.frames)
    
    assert len(results) == 2
    for i, frame in enumerate(frames):
        assert results[i].frame_id == frame.frame_id 
        
def test_parallel_processor_with_five_workers(parallel_processor):
    """Test parallel processing with 5 workers."""
    # Create a processor with 5 workers
    processor = MockParallelProcessor(num_workers=5)
    
    # Create test frames
    frames = [ProcessedFrame(np.zeros((100, 100, 3), dtype=np.uint8), i) for i in range(10)]
    
    results = []
    for frame in frames:
        result = processor(frame)
        if result.ready:
            results.extend(result.frames)
    
    # Process any remaining frames
    final_result = processor.finish()
    if final_result.ready:
        results.extend(final_result.frames)
    
    # Verify results
    assert len(results) == 10
    for i, frame in enumerate(frames):
        assert results[i].frame_id == frame.frame_id

def test_parallel_processor_process_and_finish(parallel_processor, mock_frame):
    """Test processing a frame and then calling finish multiple times."""
    # Process one frame
    result = parallel_processor(mock_frame)
    assert not result.ready
    
    # First call to finish - should return empty result since all frames were already processed
    final_result = parallel_processor.finish()
    assert final_result.ready
    assert len(final_result.frames) == 1
    assert parallel_processor.is_finished

    # Second call to finish - should still return empty result
    second_final_result = parallel_processor.finish()
    assert not second_final_result.ready 
    assert len(second_final_result.frames) == 0
    assert parallel_processor.is_finished




