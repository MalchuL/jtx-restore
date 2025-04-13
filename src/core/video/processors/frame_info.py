from dataclasses import dataclass


@dataclass(frozen=True)
class FrameInfo:
    fps: float
    frame_width: int
    frame_height: int

    def copy(self) -> "FrameInfo":
        return FrameInfo(self.fps, self.frame_width, self.frame_height)
