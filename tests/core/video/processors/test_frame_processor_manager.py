import itertools
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from typing import List, Optional, Sequence

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.frame_processor_manager import FrameProcessorManager
from src.core.video.frames.processors.processor import FrameProcessor
from src.core.video.frames.processors.processor_result import ProcessorResult
from src.core.video.frames.processors.single_frame.batch_processor import BatchProcessor
from src.core.video.frames.processors.single_frame.parallel_processor import ParallelProcessor


class MockProcessor(FrameProcessor):
    """Mock processor that passes frames through with a tag."""
    
    def __init__(self, processor_id: str):
        super().__init__()
        self.processor_id = processor_id
        self.processed_frames = []
        self.finish_called = False
    
    def _process_frame(self, frame: ProcessedFrame) -> ProcessorResult:
        """Add a tag to the frame metadata and return it."""
        # Clone the frame to avoid modifying the original
        new_frame = ProcessedFrame(
            data=frame.data.copy(),
            frame_id=frame.frame_id,
            metadata=frame.metadata.copy() if frame.metadata else {}
        )
        # Add processor_id to metadata
        new_frame.metadata["processors"] = new_frame.metadata.get("processors", []) + [self.processor_id]
        self.processed_frames.append(new_frame)
        return ProcessorResult(frames=[new_frame], ready=True)
    
    def finish(self) -> ProcessorResult:
        """Return an empty result when finish is called."""
        self.finish_called = True
        result = super().finish()
        return ProcessorResult(frames=[], ready=False)


class MockBatchProcessor(BatchProcessor):
    """Mock batch processor that processes frames in batches."""
    
    def __init__(self, batch_size: int, processor_id: str):
        super().__init__(batch_size=batch_size)
        self.processor_id = processor_id
        self.processed_batches = []
    
    def _process_single_batch(self, batch: Sequence[ProcessedFrame]) -> List[ProcessedFrame]:
        """Process a batch of frames."""
        self.processed_batches.append(batch)
        result_frames = []
        for frame in batch:
            # Clone the frame to avoid modifying the original
            new_frame = ProcessedFrame(
                data=frame.data.copy(),
                frame_id=frame.frame_id,
                metadata=frame.metadata.copy() if frame.metadata else {}
            )
            # Add processor_id to metadata
            new_frame.metadata["processors"] = new_frame.metadata.get("processors", []) + [self.processor_id]
            result_frames.append(new_frame)
        return result_frames


class MockParallelProcessor(ParallelProcessor):
    """Mock parallel processor that processes frames in parallel."""
    
    def __init__(self, num_workers: int, processor_id: str):
        super().__init__(num_workers=num_workers)
        self.processor_id = processor_id
        self.processed_frames = []
    
    @staticmethod
    def _process_frame_parallel(frame: ProcessedFrame, **kwargs) -> ProcessedFrame:
        """Process a single frame in parallel."""
        processor_id = kwargs.get("processor_id", "unknown")
        # Clone the frame to avoid modifying the original
        new_frame = ProcessedFrame(
            data=frame.data.copy(),
            frame_id=frame.frame_id,
            metadata=frame.metadata.copy() if frame.metadata else {}
        )
        # Add processor_id to metadata
        new_frame.metadata["processors"] = new_frame.metadata.get("processors", []) + [processor_id]
        return new_frame
    
    def _get_parallel_kwargs(self, frame: ProcessedFrame) -> dict:
        """Get kwargs for parallel processing."""
        return {"processor_id": self.processor_id}


@pytest.fixture
def mock_frame():
    """Create a mock frame for testing."""
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    return ProcessedFrame(data=data, frame_id=0)


@pytest.fixture
def processor_manager():
    """Create a processor manager with mock processors."""
    processors = [
        MockProcessor("p1"),
        MockProcessor("p2"),
        MockProcessor("p3")
    ]
    return FrameProcessorManager(processors=processors)


def test_frame_processor_manager_init():
    """Test initializing a frame processor manager."""
    processors = [MockProcessor("p1"), MockProcessor("p2")]
    manager = FrameProcessorManager(processors=processors)
    assert len(manager.processors) == 2


def test_frame_processor_manager_initialize(processor_manager):
    """Test initializing all processors in the manager."""
    # Mock the initialize method on all processors
    for processor in processor_manager.processors:
        processor.initialize = MagicMock()
    
    # Call initialize on the manager
    processor_manager.initialize()
    
    # Verify initialize was called on all processors
    for processor in processor_manager.processors:
        processor.initialize.assert_called_once()


def test_frame_processor_manager_reset(processor_manager):
    """Test resetting all processors in the manager."""
    # Mock the reset method on all processors
    for processor in processor_manager.processors:
        processor.reset = MagicMock()
    
    # Call reset on the manager
    processor_manager.reset()
    
    # Verify reset was called on all processors
    for processor in processor_manager.processors:
        processor.reset.assert_called_once()


def test_frame_processor_manager_process_frame(processor_manager, mock_frame):
    """Test processing a frame through all processors."""
    # Process a frame
    result = processor_manager.process_frame(mock_frame)
    
    # Verify the result
    assert len(result) == 1
    assert result[0].frame_id == mock_frame.frame_id
    
    # Verify the frame was processed by all processors
    assert "processors" in result[0].metadata
    assert result[0].metadata["processors"] == ["p1", "p2", "p3"]
    
    # Verify the frame_id was preserved
    assert result[0].frame_id == 0


def test_frame_processor_manager_process_none_frame(processor_manager):
    """Test processing a None frame (finish case)."""
    # Process a None frame
    result = processor_manager.process_frame(None)
    assert len(result) == 0


def test_frame_processor_manager_call(processor_manager, mock_frame):
    """Test the call method of the manager."""
    # Mock the process_frame method
    processor_manager.process_frame = MagicMock(return_value=[mock_frame])
    
    # Call the manager
    result = processor_manager(mock_frame)
    
    # Verify process_frame was called with the frame
    processor_manager.process_frame.assert_called_once_with(mock_frame)
    assert result == [mock_frame]


def test_frame_processor_manager_with_batch_processors(mock_frame):
    """Test processor manager with batch processors of different batch sizes."""
    # Create processors with different batch sizes
    processors = [
        MockBatchProcessor(batch_size=2, processor_id="batch2"),
        MockBatchProcessor(batch_size=3, processor_id="batch3"),
        MockProcessor("normal")
    ]
    manager = FrameProcessorManager(processors=processors)
    
    # Create test frames
    frames = [
        ProcessedFrame(np.zeros((100, 100, 3), dtype=np.uint8), i) 
        for i in range(6)
    ]
    
    # Process frames
    results = []
    for frame in frames:
        result = manager.process_frame(frame)
        if result:
            results.extend(result)
    
    # Get remaining frames
    final_result = manager.process_frame(None)
    if final_result:
        results.extend(final_result)
    
    # Verify all frames were processed
    assert len(results) == 6
    
    # Verify frame_ids are preserved and in correct order
    frame_ids = [frame.frame_id for frame in results]
    expected_ids = list(range(6))
    assert frame_ids == expected_ids, f"Expected frame_ids {expected_ids}, got {frame_ids}"
    
    # Create a mapping of frame_id to result frame for additional checks
    results_by_id = {result.frame_id: result for result in results}
    
    # Verify each frame was processed by the appropriate processors
    for frame_id in range(6):
        result = results_by_id[frame_id]
        assert "processors" in result.metadata
        # The first batch processor will process in batches of 2
        # The second batch processor will process in batches of 3
        # The normal processor will process each frame individually
        assert "batch2" in result.metadata["processors"]
        assert "batch3" in result.metadata["processors"]
        assert "normal" in result.metadata["processors"]


def test_frame_processor_manager_with_parallel_processors(mock_frame):
    """Test processor manager with parallel processors of different worker counts."""
    # Create processors with different worker counts
    processors = [
        MockParallelProcessor(num_workers=2, processor_id="parallel2"),
        MockParallelProcessor(num_workers=3, processor_id="parallel3"),
        MockProcessor("normal")
    ]
    manager = FrameProcessorManager(processors=processors)
    
    # Create test frames
    frames = [
        ProcessedFrame(np.zeros((100, 100, 3), dtype=np.uint8), i) 
        for i in range(6)
    ]
    
    # Process frames
    results = []
    for frame in frames:
        result = manager.process_frame(frame)
        if result:
            results.extend(result)
    
    # Get remaining frames
    final_result = manager.process_frame(None)
    if final_result:
        results.extend(final_result)
    
    # Verify all frames were processed
    assert len(results) == 6
    
    # Verify frame_ids are preserved and in correct order
    frame_ids = [frame.frame_id for frame in results]
    expected_ids = list(range(6))
    assert frame_ids == expected_ids, f"Expected frame_ids {expected_ids}, got {frame_ids}"
    
    # Create a mapping of frame_id to result frame for additional checks
    results_by_id = {result.frame_id: result for result in results}
    
    # Verify each frame was processed by the appropriate processors
    for frame_id in range(6):
        result = results_by_id[frame_id]
        assert "processors" in result.metadata
        # Both parallel processors and the normal processor should have processed each frame
        assert "parallel2" in result.metadata["processors"]
        assert "parallel3" in result.metadata["processors"]
        assert "normal" in result.metadata["processors"]


@pytest.mark.parametrize("num_frames", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("parallel_size", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5])
def test_auto_processor_pipeline(num_frames, parallel_size, batch_size):
    processors = [
        MockProcessor("normal"),
        MockBatchProcessor(batch_size=batch_size, processor_id=f"batch{batch_size}"),
        MockParallelProcessor(num_workers=parallel_size, processor_id=f"1parallel{parallel_size}"),
        MockParallelProcessor(num_workers=parallel_size, processor_id=f"2parallel{parallel_size}"),
    ]
    frames = [
        ProcessedFrame(np.zeros((100, 100, 3), dtype=np.uint8), i) 
        for i in range(num_frames)
    ]
    for permutations in itertools.permutations(processors):
        for processor in permutations:
            processor.reset()
        manager = FrameProcessorManager(processors=permutations)
        # Process frames
        results = []
        for frame in frames:
            result = manager.process_frame(frame)
            if result:
                results.extend(result)
        
        # Get remaining frames
        final_result = manager.process_frame(None)
        if final_result:
            results.extend(final_result)
        assert len(results) == num_frames, f"Expected {num_frames} frames, got {len(results)}, processors: {permutations}"
        frame_ids = [frame.frame_id for frame in results]
        expected_ids = list(range(num_frames))
        assert frame_ids == expected_ids, f"Expected frame_ids {expected_ids}, got {frame_ids}, processors: {permutations}"
    


def test_mixed_processor_pipeline(mock_frame):
    """Test a pipeline with mixed processor types (batch, parallel, normal)."""
    # Create a mixed pipeline of processors
    processors = [
        MockProcessor("normal"),
        MockBatchProcessor(batch_size=4, processor_id="batch4"),
        MockParallelProcessor(num_workers=3, processor_id="parallel3"),
    ]
    manager = FrameProcessorManager(processors=processors)
    
    # Create test frames
    frames = [
        ProcessedFrame(np.zeros((100, 100, 3), dtype=np.uint8), i) 
        for i in range(7)
    ]
    
    # Process frames
    results = []
    for frame in frames:
        result = manager.process_frame(frame)
        if result:
            results.extend(result)
    
    # Get remaining frames
    final_result = manager.process_frame(None)
    if final_result:
        results.extend(final_result)
    
    # Verify all frames were processed
    assert len(results) == 7
    print(results)
    
    # Verify frame_ids are preserved and in correct order
    frame_ids = [frame.frame_id for frame in results]
    expected_ids = list(range(7))
    assert frame_ids == expected_ids, f"Expected frame_ids {expected_ids}, got {frame_ids}"
    
    # Create a mapping of frame_id to result frame for additional checks
    results_by_id = {result.frame_id: result for result in results}
    
    # Verify each frame was processed by all processors
    for frame_id in range(7):
        result = results_by_id[frame_id]
        assert result.frame_id == frame_id
        assert "processors" in result.metadata
        processors_in_metadata = result.metadata["processors"]
        assert "batch4" in processors_in_metadata, f"Frame {frame_id} missing batch4 processing"
        assert "parallel3" in processors_in_metadata, f"Frame {frame_id} missing parallel3 processing"
        assert "normal" in processors_in_metadata, f"Frame {frame_id} missing normal processing"
        
    # Ensure processor order was maintained (processors should be applied in order)
    for frame_id in range(7):
        processors_applied = results_by_id[frame_id].metadata["processors"]
        # Check that batch4 comes before parallel3, and parallel3 comes before normal
        batch4_idx = processors_applied.index("batch4")
        parallel3_idx = processors_applied.index("parallel3")
        normal_idx = processors_applied.index("normal")
        
        # Our mock implementations add processor IDs in order of processing
        # So indices should be in ascending order
        assert normal_idx < batch4_idx < parallel3_idx, \
            f"Frame {frame_id} processors applied in wrong order: {processors_applied}" 