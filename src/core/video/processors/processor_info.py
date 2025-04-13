from dataclasses import dataclass


@dataclass
class ProcessorInfo:
    fps_scale: float = 1.0
    frame_width_scale: float = 1.0
    frame_height_scale: float = 1.0

    def copy(self) -> "ProcessorInfo":
        return ProcessorInfo(
            self.fps_scale, self.frame_width_scale, self.frame_height_scale
        )

    def set_fps_scale(self, fps_scale: float) -> "ProcessorInfo":
        info = self.copy()
        info.fps_scale = fps_scale
        return info

    def set_frame_width_scale(self, frame_width_scale: float) -> "ProcessorInfo":
        info = self.copy()
        info.frame_width_scale = frame_width_scale
        return info

    def set_frame_height_scale(self, frame_height_scale: float) -> "ProcessorInfo":
        info = self.copy()
        info.frame_height_scale = frame_height_scale
        return info
