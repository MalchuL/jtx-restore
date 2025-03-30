from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar
from src.core.video.processors.frame import ProcessedFrame


T = TypeVar("T")


class FrameCutter(Generic[T]):
    """A class that splits a sequence of frames into overlapping windows.

    This class is designed to process video frames by creating overlapping windows
    of frames, which is useful for frame interpolation and other temporal processing tasks.
    It maintains a sliding window with configurable overlap and window sizes.

    Attributes:
        window_size (int): Total number of frames in each window.
        non_overlap_size (int): Number of frames that don't overlap between consecutive windows.
        begin_non_overlap (int): Number of frames at the start that are not part of the overlap.

    Example:
        >>> cutter = FrameCutter(window_size=4, non_overlap_size=2)
        >>> frames = [frame1, frame2, frame3, frame4, frame5]
        >>> for frame in frames:
        ...     window = cutter(frame)
        ...     if window:
        ...         process_window(window)
        >>> final_window = cutter(None)  # Process remaining frames

    Args:
        non_overlap_size (int): Number of frames that don't overlap between windows. Must be > 0 and < window_size.
        window_size (int): Size of each window. Must be >= 1.
        begin_non_overlap (Optional[int]): Number of frames at the start that are not part of the overlap. If None,
            defaults to (window_size - non_overlap_size) // 2.

    Raises:
        ValueError: If window_size < 1, non_overlap_size <= 0, non_overlap_size > window_size,
            or begin_non_overlap > window_size - non_overlap_size.
    """

    def __init__(
        self,
        window_size: int = 2,
        non_overlap_size: Optional[int] = None,
        begin_non_overlap: Optional[int] = None,
    ):
        self.window_size = window_size
        self.non_overlap_size = non_overlap_size
        if self.non_overlap_size is None:
            self.non_overlap_size = self.window_size
        if self.window_size < 1:
            raise ValueError("window must be at least 1")
        if self.non_overlap_size > self.window_size:
            raise ValueError("non_overlap must be less or equal than window")
        if self.non_overlap_size <= 0:
            raise ValueError("non_overlap must be greater than 0")
        self.begin_non_overlap = begin_non_overlap
        if self.begin_non_overlap is None:
            self.begin_non_overlap = (self.window_size - self.non_overlap_size) // 2
        elif self.begin_non_overlap > self.window_size - self.non_overlap_size:
            raise ValueError("begin_non_overlap must be less than window - non_overlap")

        self._processed_frames = 0
        self._frames: Deque[T] = deque()
        self._padded = False  # Is initial padding applied. Used for begin_non_overlap.
        
        # When we have None frame, we need to check if we finished processing
        # We need to check if we have enough frames to return
        # Because we can have situation when we have less frames than window size
        # And we need to pad the last frame
        self._remaining_frames = 0
        self._is_finish = False

        """
        In example we'll have 12 frames, we already process
        This class splits frames to make overlap to be at all positions without intersection.
        00000000000000  # 12 frames
        00001111000000  # 12 frames, 4 window size
        00002211000000  # 12 frames, 4 window size, 2 non_overlap
        00001122211100  # 12 frames, 8 window size, 3 non_overlap, 2 begin_non_overlap (position start from 0)
        """

    def _pad(self, frame: T, left_padding: int = 0, right_padding: int = 0) -> T:
        """Add padding frames to the buffer.

        Args:
            frame (T): The frame to use for padding.
            left_padding (int): Number of frames to pad on the left.
            right_padding (int): Number of frames to pad on the right.

        Note:
            The buffer size is maintained at window_size by removing excess frames.
        """
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
        """Add a new frame to the buffer.

        Args:
            frame (T): The frame to add.

        Note:
            This method adds one frame to the end of the buffer and maintains
            the window size by removing excess frames.
        """
        self._pad(frame, right_padding=1)

    def process_frame(self, frame: Optional[T]) -> Optional[List[T]]:
        """Process a single frame and return a window if available.

        Args:
            frame (Optional[T]): The frame to process. If None, indicates end of processing.

        Returns:
            Optional[List[T]]: A list of frames forming a window if available, None otherwise.

        Raises:
            RuntimeError: If called after processing is finished.
        """
        if not self._padded:
            self._pad(frame, self.begin_non_overlap)
            self._processed_frames += self.begin_non_overlap
            self._padded = True
            # Pads frames to make initial non_overlap

        if frame is not None:
            self._add_frame(frame)
            self._processed_frames += 1
        else:
            # If frame is None, it means that we have reached the end of the video
            # We need to pad the last frame to make it the same size as the window
            # And return frames that we doesnt see in buffer
            
            # If _processed_frames is less than window size, we need to pad the last frame
            # But if _processed_frames is at start of processing, we already return all frames
            if not self._is_finish:
                # Number of frames that we add but window size is not reached
                remaining_frames = self._processed_frames - (self.window_size - self.non_overlap_size)
                # Number of frames that remains at right side of window that we must process at the end
                not_processed_frames = self.window_size - self.non_overlap_size - self.begin_non_overlap
                self._remaining_frames = remaining_frames + not_processed_frames
                self._is_finish = True

            if self._remaining_frames > 0:
                self._pad(self._frames[-1], right_padding=self.window_size - self._processed_frames)
                self._processed_frames = self.window_size
                self._remaining_frames -= self.non_overlap_size
            else:
                return None

        if (
            self._processed_frames >= self.window_size
            and len(self._frames) >= self.window_size
        ):
            # Count of frames that we have for new window
            # This value doesn't depend on begin_non_overlap
            # Because we only need to offset at the beggining of processing
            # After that we can just take offset
            # Because we always offsets on window size - non_overlap_size

            self._processed_frames = self.window_size - self.non_overlap_size

            frames = list(self._frames)
            return list(frames)

    def __call__(self, frame: T) -> Optional[List[T]]:
        """Make the FrameCutter callable.

        Args:
            frame (T): The frame to process.

        Returns:
            Optional[List[T]]: A window of frames if available, None otherwise.
        """
        return self.process_frame(frame)
