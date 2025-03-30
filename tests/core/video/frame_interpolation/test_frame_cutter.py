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
    return ProcessedFrame(np.ones((1,)) * frame_id, frame_id=frame_id)


class TestFrameCutter:
    """Test suite for the FrameCutter class."""

    def test_initialization(self):
        """Test proper initialization of FrameCutter with valid parameters."""
        cutter = FrameCutter(overlap_size=1, window_size=3)
        assert cutter.overlap_size == 1
        assert cutter.window_size == 3
        assert len(cutter._frames) == 0

    def test_invalid_initialization(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="window must be at least 1"):
            FrameCutter(window_size=0)



    def test_frame_accumulation(self):
        """Test that frames are properly accumulated in the buffer."""
        ws = 4
        cutter = FrameCutter[ProcessedFrame](window_size=ws, overlap_size=2, begin_overlap=1)
        
        frames = [create_num_frame(i) for i in range(10)]
        for frame in frames:
            out = cutter(frame)
            if out is not None:
                assert len(out) == ws
                print(out)
        out = cutter(None)
        print("last",out)
        print(cutter._frames)
       # assert cutter._frames == frames
    
    