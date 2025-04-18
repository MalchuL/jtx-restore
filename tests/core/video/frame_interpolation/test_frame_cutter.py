import pytest
import numpy as np
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.utils.frame_cutter import FrameCutter, CutterWindow


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

    def test_reset_after_processing(self):
        """Test that reset properly clears state after processing frames."""
        cutter = FrameCutter[ProcessedFrame](window_size=3, non_overlap_size=1)
        
        # Process some frames
        frames = [create_num_frame(i) for i in range(4)]
        for frame in frames:
            window = cutter(frame)
            if window.ready:
                assert len(window.frames) == 3
        
        # Verify state is not empty
        assert len(cutter._frames) > 0
        assert cutter._processed_frames > 0
        assert cutter._padded
        
        # Reset the cutter
        cutter.reset()
        
        # Verify all state is cleared
        assert len(cutter._frames) == 0
        assert cutter._processed_frames == 0
        assert not cutter._padded
        assert cutter._remaining_frames == 0
        assert not cutter._is_finish

    def test_reset_and_reuse(self):
        """Test that cutter can be reused after reset."""
        cutter = FrameCutter[ProcessedFrame](window_size=3, non_overlap_size=1)
        
        # First processing sequence
        frames1 = [create_num_frame(i) for i in range(4)]
        windows1 = []
        for frame in frames1:
            window = cutter(frame)
            if window.ready:
                windows1.append(window.frames)
        remaining_windows = cutter.get_remaining_windows()
        windows1.extend([w.frames for w in remaining_windows])
        
        # Reset and process new sequence
        cutter.reset()
        frames2 = [create_num_frame(i) for i in range(4)]
        windows2 = []
        for frame in frames2:
            window = cutter(frame)
            if window.ready:
                windows2.append(window.frames)
        remaining_windows = cutter.get_remaining_windows()
        windows2.extend([w.frames for w in remaining_windows])
        
        # Verify both sequences produced same number of windows
        assert len(windows1) == len(windows2)
        
        # Verify window contents are correct for second sequence
        for i, (window1, window2) in enumerate(zip(windows1, windows2)):
            assert len(window1) == len(window2) == 3
            assert all(w1.frame_id == w2.frame_id for w1, w2 in zip(window1, window2))


    def test_reset_with_begin_non_overlap(self):
        """Test reset with non-zero begin_non_overlap."""
        cutter = FrameCutter[ProcessedFrame](
            window_size=4,
            non_overlap_size=2,
            begin_non_overlap=2
        )
        
        # Process some frames
        frames = [create_num_frame(i) for i in range(5)]
        windows = []
        for frame in frames:
            window = cutter(frame)
            if window.ready:
                assert len(window.frames) == 4
                windows.append(window.frames)
        remaining_windows = cutter.get_remaining_windows()
        windows.extend([w.frames for w in remaining_windows])
        
        # Reset and verify initial padding is cleared
        cutter.reset()
        assert not cutter._padded
        assert len(cutter._frames) == 0
        assert cutter._processed_frames == 0
        assert not cutter._padded
        assert cutter._remaining_frames == 0
        assert not cutter._is_finish
        
        # Process new frames and verify initial padding is reapplied
        new_frame = create_num_frame(0)
        window = cutter(new_frame)
        assert not window.ready  # First window not complete yet
        assert cutter._padded
        assert len(cutter._frames) == 3  # Because we pad frames to make initial non_overlap

    @pytest.mark.parametrize("range_size", [4, 5, 6, 7, 8])
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
            window = cutter(frame)
            if window.ready:
                assert len(window.frames) == ws
                windows.append(window.frames)
        
        # Get remaining windows
        remaining_windows = cutter.get_remaining_windows()
        windows.extend([w.frames for w in remaining_windows])
        
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
            window = cutter(frame)
            if window.ready:
                windows.append(window.frames)
        
        # Get remaining windows
        remaining_windows = cutter.get_remaining_windows()
        windows.extend([w.frames for w in remaining_windows])
        
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
            window = cutter(frame)
            if window.ready:
                windows.append(window.frames)
        
        # Get remaining windows
        remaining_windows = cutter.get_remaining_windows()
        windows.extend([w.frames for w in remaining_windows])
        
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
        windows = []
        for frame in frames:
            window = cutter(frame)
            if window.ready:
                windows.append(window.frames)
        
        # Get remaining windows
        remaining_windows = cutter.get_remaining_windows()
        windows.extend([w.frames for w in remaining_windows])
        
        # Verify final window
        assert len(windows[-1]) == 3

    def test_empty_sequence(self):
        """Test behavior with an empty sequence of frames."""
        cutter = FrameCutter[ProcessedFrame](window_size=3, non_overlap_size=1)
        
        # Get remaining windows (should be empty)
        remaining_windows = cutter.get_remaining_windows()
        assert len(remaining_windows) == 0

    def infer_cutter(self, range_size=10, window_size=4, non_overlap_size=None, begin_non_overlap=None):
        cutter = FrameCutter[int](window_size=window_size, non_overlap_size=non_overlap_size, begin_non_overlap=begin_non_overlap)
        frames = [i for i in range(range_size)]
        windows = []
        for frame in frames:
            window = cutter(frame)
            if window.ready:
                windows.append(window.frames)
        remaining_windows = cutter.get_remaining_windows()
        windows.extend([w.frames for w in remaining_windows])
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
        
    def test_remaining_frames(self):
        cutter = FrameCutter[ProcessedFrame](window_size=3, non_overlap_size=3)
        frames = [create_num_frame(i) for i in range(6)]
        windows = []
        for frame in frames:
            window = cutter(frame)
            if window.ready:
                windows.append(window.frames)
                
        remaining_windows = cutter.get_remaining_windows()
        assert len(windows + [w.frames for w in remaining_windows]) == 2
        assert len(remaining_windows) == 0
        out = cutter(None)
        assert out.finished
        
    def test_no_remaining_frames(self):
        """Test behavior when there are no remaining frames to process."""
        cutter = FrameCutter[ProcessedFrame](window_size=3, non_overlap_size=3)
        frames = [create_num_frame(i) for i in range(6)]
        windows = []
        for frame in frames:
            window = cutter(frame)
            if window.ready:
                windows.append(window.frames)
                
        # Signal end of processing
        window = cutter(None)
        assert not window.ready
        assert window.finished
        
        # Get remaining windows
        remaining_windows = cutter.get_remaining_windows()
        assert len(windows + [w.frames for w in remaining_windows]) == 2
        assert len(remaining_windows) == 0
        
    def test_not_ready_state(self):
        """Test that not ready state is properly indicated."""
        cutter = FrameCutter[ProcessedFrame](window_size=3, non_overlap_size=1)
        
        # Process first frame
        window = cutter(create_num_frame(0))
        assert not window.ready
        assert window.remaining_frames == 1  # We pad left first
        
        # Process second frame
        window = cutter(create_num_frame(1))
        assert window.ready
        assert window.remaining_frames == 0
        
        # Process third frame
        window = cutter(create_num_frame(2))
        assert window.ready  # Because non_overlap_size is 1, we always return full window
        assert window.remaining_frames == 0
        assert len(window.frames) == 3
        
    def test_end_of_sequence(self):
        """Test that end of sequence is properly indicated."""
        cutter = FrameCutter[ProcessedFrame](window_size=3, non_overlap_size=1)
        
        # Process frames
        frames = [create_num_frame(i) for i in range(4)]
        for frame in frames:
            window = cutter(frame)
            if window.ready:
                assert not window.finished
        
        # Signal end of processing
        window = cutter(None)
        assert not window.finished  # We still have frames to process
        
        # Get remaining windows
        remaining_windows = cutter.get_remaining_windows()
        assert len(remaining_windows) == 0
        
    def test_reset_after_finish(self):
        """Test that reset works after finish signal."""
        cutter = FrameCutter[ProcessedFrame](window_size=3, non_overlap_size=1)
        
        # Process frames and signal finish
        frames = [create_num_frame(i) for i in range(4)]
        windows = []
        for frame in frames:
            window = cutter(frame)
            if window.ready:
                windows.append(window.frames)
        remaining_windows = cutter.get_remaining_windows()
        windows.extend([w.frames for w in remaining_windows])
        
        print(remaining_windows)
        # Verify finish state
        # We must not add empty windows to remaining windows
        assert not remaining_windows[-1].finished
        assert len(remaining_windows[-1].frames) != 0
        
        # Reset and verify state
        cutter.reset()
        assert not cutter._is_finish
        assert cutter._remaining_frames == 0
        
        # Verify can process new frames
        new_frame = create_num_frame(0)
        window = cutter(new_frame)
        assert not window.ready  # First window not complete yet
