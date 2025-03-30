import pytest
import numpy as np
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.frame_interpolation.frame_cutter import FrameCutter


def create_test_frame(frame_id: int) -> ProcessedFrame:
    """Helper function to create test frames with dummy data."""
    return ProcessedFrame(
        data=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_id=frame_id
    )


def create_num_frame(frame_id: int) -> ProcessedFrame:
    """Helper function to create test frames with numeric data."""
    return ProcessedFrame(np.ones((1,)) * frame_id, frame_id=frame_id)


class TestFrameCutter:
    """Test suite for the FrameCutter class."""

    def test_initialization(self):
        """Test proper initialization of FrameCutter with valid parameters."""
        cutter = FrameCutter(non_overlap_size=1, window_size=3)
        assert cutter.non_overlap_size == 1
        assert cutter.window_size == 3
        assert len(cutter._frames) == 0

    def test_invalid_initialization(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="window must be at least 1"):
            FrameCutter(window_size=0)
        
        with pytest.raises(ValueError, match="non_overlap must be greater than 0"):
            FrameCutter(non_overlap_size=0, window_size=3)
        
        with pytest.raises(ValueError, match="non_overlap must be less or equal than window"):
            FrameCutter(non_overlap_size=4, window_size=3)
        
        with pytest.raises(ValueError, match="begin_non_overlap must be less than window - non_overlap"):
            FrameCutter(non_overlap_size=1, window_size=3, begin_non_overlap=3)

    @pytest.mark.parametrize("range_size", [6, 7, 8, 9, 10])
    @pytest.mark.parametrize("window_size", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("non_overlap_size", [None, 1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("begin_non_overlap", [None, 0, 1,  2, 3, 4, 5, 6])
    def test_automatic_cutter(self, range_size, window_size, non_overlap_size, begin_non_overlap):
        # 2. All frames without overlap at center
        if non_overlap_size is not None and non_overlap_size > window_size:
            return
        if begin_non_overlap is not None and non_overlap_size is not None and begin_non_overlap > window_size - non_overlap_size:
            return
        if begin_non_overlap not in [None, 0] and non_overlap_size is None:
            return
        windows = self.infer_cutter(range_size=range_size, window_size=window_size, non_overlap_size=non_overlap_size, begin_non_overlap=begin_non_overlap)
        # Assign concrete values
        if non_overlap_size is None:
            non_overlap_size = window_size
        if begin_non_overlap is None:
            begin_non_overlap = (window_size - non_overlap_size) // 2
        
        checker = 0
        # Task of cutter is to make overlap at all positions without intersection, but in window size they can
        for window in windows:
            assert len(window) == window_size, f"Window size is not equal to window_size, window: {window}, window_size: {window_size}, non_overlap_size: {non_overlap_size}, begin_non_overlap: {begin_non_overlap}"
            for i in range(non_overlap_size):
                assert window[begin_non_overlap + i] == checker, windows
                checker += 1
                checker = min(checker, range_size-1)
        assert checker == range_size - 1

    def test_frame_accumulation(self):
        """Test that frames are properly accumulated in the buffer."""
        ws = 4
        cutter = FrameCutter[ProcessedFrame](window_size=ws, non_overlap_size=2, begin_non_overlap=1)
        
        frames = [create_num_frame(i) for i in range(10)]
        windows = []
        
        for frame in frames:
            out = cutter(frame)
            if out is not None:
                assert len(out) == ws
                windows.append(out)
        
        # Test final window
        final_window = cutter(None)
        assert final_window is not None
        assert len(final_window) == ws
        windows.append(final_window)
        
        # Verify window contents
        assert len(windows) > 0
        for window in windows:
            assert all(isinstance(f, ProcessedFrame) for f in window)
            assert len(window) == ws

    def test_overlap_behavior(self):
        """Test that overlapping frames are correctly maintained between windows."""
        ws = 4
        non_overlap = 2
        cutter = FrameCutter[ProcessedFrame](window_size=ws, non_overlap_size=non_overlap)
        
        frames = [create_num_frame(i) for i in range(6)]
        windows = []
        
        for frame in frames:
            out = cutter(frame)
            if out is not None:
                windows.append(out)
        
        final_window = cutter(None)
        if final_window is not None:
            windows.append(final_window)
        
        # Verify overlap between consecutive windows
        for i in range(len(windows) - 1):
            current_window = windows[i]
            next_window = windows[i + 1]
            assert current_window[-non_overlap:] == next_window[:non_overlap]

    def test_begin_non_overlap(self):
        """Test that begin_non_overlap parameter correctly pads the start."""
        ws = 4
        non_overlap = 2
        begin_non_overlap = 1
        cutter = FrameCutter[ProcessedFrame](
            window_size=ws,
            non_overlap_size=non_overlap,
            begin_non_overlap=begin_non_overlap
        )
        
        frames = [create_num_frame(i) for i in range(5)]
        windows = []
        
        for frame in frames:
            out = cutter(frame)
            if out is not None:
                windows.append(out)
        
        final_window = cutter(None)
        if final_window is not None:
            windows.append(final_window)
        
        # Verify first window has correct padding
        if windows:
            first_window = windows[0]
            assert len(first_window) == ws
            # First frame should be repeated begin_non_overlap times
            assert all(first_window[i].frame_id == frames[0].frame_id 
                      for i in range(begin_non_overlap))

    def test_finish_behavior(self):
        """Test that the cutter properly handles the end of processing."""
        cutter = FrameCutter[ProcessedFrame](window_size=3, non_overlap_size=1)
        
        # Process some frames
        frames = [create_num_frame(i) for i in range(4)]
        for frame in frames:
            out = cutter(frame)
            if out is not None:
                print([frame.frame_id for frame in out])
        
        # Signal end of processing
        final_window = cutter(None)
        assert len(final_window) == 3

    def test_empty_sequence(self):
        """Test behavior with an empty sequence of frames."""
        cutter = FrameCutter[ProcessedFrame](window_size=3, non_overlap_size=1)
        
        # Immediately signal end of processing
        final_window = cutter(None)
        assert final_window is None

    def infer_cutter(self, range_size=10, window_size=4, non_overlap_size=None, begin_non_overlap=None):
        cutter = FrameCutter[int](window_size=window_size, non_overlap_size=non_overlap_size, begin_non_overlap=begin_non_overlap)
        frames = [i for i in range(range_size)]
        windows = []
        for frame in frames:
            out = cutter(frame)
            if out is not None:
                windows.append(out)
        for _ in range(window_size):
            out = cutter(None)
            if out is not None:
                windows.append(out)
            else:
                break
        else:
            raise RuntimeError("Frame cutter is finished")
        return windows

    def test_cutter_cases(self):
        # 1. All frames without overlap
        # 2. All frames with overlap
        # 3. All frames with begin overlap
        # 4. All frames with end overlap
        # 5. All frames with begin and end overlap
        # 6. All frames with begin and end overlap and overlap size is not equal to window size
        
        # 1. All frames without overlap
        windows = self.infer_cutter(range_size=10, window_size=4)
        assert windows == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 9, 9]]

        # 2. All frames without overlap at center
        begin_non_overlap = 1
        non_overlap_size = 2
        windows = self.infer_cutter(range_size=6, window_size=4, non_overlap_size=non_overlap_size, begin_non_overlap=begin_non_overlap)
        assert windows == [[0, 0, 1, 2], 
                           [1, 2, 3, 4], 
                           [3, 4, 5, 5]]
        checker = 0
        for window in windows:
            for i in range(non_overlap_size):
                assert window[begin_non_overlap + i] == checker
                checker += 1

        # Test no overlap that returns all frames without overlap
        begin_non_overlap = 0
        non_overlap_size = 3
        windows = self.infer_cutter(range_size=6, window_size=non_overlap_size, non_overlap_size=non_overlap_size, begin_non_overlap=begin_non_overlap)
        assert windows == [[0, 1, 2], 
                           [3, 4, 5]]
        checker = 0
        for window in windows:
            for i in range(non_overlap_size):
                assert window[begin_non_overlap + i] == checker
                checker += 1       
                
    def test_single_frame(self):
        windows = self.infer_cutter(range_size=6, window_size=3, non_overlap_size=2, begin_non_overlap=1)
        print(windows)