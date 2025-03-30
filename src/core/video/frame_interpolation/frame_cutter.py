from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar
from src.core.video.processors.frame import ProcessedFrame


T = TypeVar("T")


class FrameCutter(Generic[T]):
    def __init__(
        self,
        overlap_size: int = 1,
        window_size: int = 2,
        begin_overlap: Optional[int] = None,
    ):
        self.overlap_size = overlap_size

        self.window_size = window_size
        if self.window_size < 1:
            raise ValueError("window must be at least 1")
        if self.overlap_size > self.window_size:
            raise ValueError("overlap must be less than window - 1")
        if self.overlap_size <= 0:
            raise ValueError("overlap must be greater than 0")
        self.begin_overlap = begin_overlap
        if self.begin_overlap is None:
            self.begin_overlap = (self.window_size - self.overlap_size) // 2
        elif self.begin_overlap > self.window_size - self.overlap_size:
            raise ValueError("begin overlap must be less than window - overlap")

        self._processed_frames = 0
        self._frames: Deque[T] = deque()
        self._padded = False  # Is inintial padding applied. Used for begin overlap.

        self._is_finish = False

        """
        In example we'll have 12 frames, we already process
        This class splits frames to make overlap to be at all positions without intersection.
        00000000000000  # 12 frames
        00001111000000  # 12 frames, 4 window size
        00002211000000  # 12 frames, 4 window size, 2 overlap
        00001122211100  # 12 frames, 8 window size, 3 overlap, 2 begin overlap (position start from 0)
        """

    def _pad(self, frame: T, left_padding: int = 0, right_padding: int = 0) -> T:
        if left_padding > 0:
            for _ in range(left_padding):
                self._frames.appendleft(frame)
                if len(self._frames) > self.window_size:
                    self._frames.pop()
        if right_padding > 0:
            for _ in range(right_padding):
                self._frames.append(frame)
                if len(self._frames) > self.window_size:
                    self._frames.popleft()

    def _add_frame(self, frame: T) -> List[T]:
        self._pad(
            frame, right_padding=1
        )  # Add one frame to the end of the buffer similar to padding

    def process_frame(self, frame: Optional[T]) -> Optional[List[T]]:
        if not self._padded:
            self._pad(frame, self.begin_overlap)
            self._processed_frames += self.begin_overlap
            self._padded = True
            # Pads frames to make initial overlap

        # If frame is None, it means that we have reached the end of the video
        # We need to pad the last frame to make it the same size as the window
        # And return frames that we doesnt see in buffer
        if frame is None:
            self._pad(
                self._frames[-1],
                right_padding=self.window_size - self._processed_frames,
            )
            self._processed_frames = self.window_size
            self._is_finish = True
            return list(self._frames)

        if self._is_finish:
            raise RuntimeError("Frame cutter is finished")

        self._add_frame(frame)
        self._processed_frames += 1

        if (
            self._processed_frames >= self.window_size
            and len(self._frames) >= self.window_size
        ):
            self._processed_frames = (
                # Count of frames that we have for new window
                # This value doesn't depend on begin overlap
                # Because we only need to offset at the beggining of processing
                # After that we can just take offset
                # Because we always offsets on window size - overlap size
                self.window_size
                - self.overlap_size
            )  # Reset processed frames to begin overlap

            frames = list(self._frames)
            return list(frames)

    def __call__(self, frame: T) -> Optional[List[T]]:
        return self.process_frame(frame)
