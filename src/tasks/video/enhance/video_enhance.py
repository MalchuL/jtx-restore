import os
from pathlib import Path
from typing import List

from src.core.video.frames.processors import FrameProcessor
from src.pipelines.merger.ffmpeg_video_audio_merger import (
    FFMPEGVideoAudioMerger,
    VideoAudioMerger,
)
from src.pipelines.video import OpenCVFFmpegPipeline, DefaultVideoPipeline
from src.utils.path_utils import iterate_files_with_creating_structure


class VideoEnhanceTask:
    def __init__(
        self,
        processors: List[FrameProcessor],
        video_folder: str,
        output_folder: str,
        supported_extensions: List[str] = (
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".flv",
            ".wmv",
            ".webm",
            ".m4v",
            ".m4a",
            ".m4b",
            ".m4p",
            ".m4v"
        ),
        quality: int = 80,
    ):
        self.processors = processors
        self._video_folder = Path(video_folder)
        self._output_folder = Path(output_folder)
        self._supported_extensions = supported_extensions
        self.quality = quality

    def _create_video_pipeline(
        self, video_file: str, output_file: str
    ) -> DefaultVideoPipeline:
        pipeline = OpenCVFFmpegPipeline(
            input_path=video_file,
            output_path=output_file,
            processors=self.processors,
            quality=self.quality,
        )
        return pipeline

    def _create_merger(self, video_file: str, audio_file: str, output_file: str) -> VideoAudioMerger:
        return FFMPEGVideoAudioMerger(
            video_path=video_file, audio_path=audio_file, output_path=output_file
        )

    def enhance(self):

        if os.path.isdir(self._video_folder):
            iterator = iterate_files_with_creating_structure(
                self._video_folder,
                self._output_folder,
                supported_extensions=self._supported_extensions,
                use_natsort=True,
                use_tqdm=True,
            )
        else:
            if os.path.isdir(self._output_folder):
                raise ValueError(
                    f"Output {self._output_folder} is a directory but video folder is a file"
                )
            # Avoid empty path
            dir_path = os.path.dirname(self._output_folder)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            iterator = [(self._video_folder, self._output_folder)]

        for video_file, output_file in iterator:
            # Create tmp file without audio
            ext = os.path.splitext(output_file)[1]
            replace_to = "_tmp" + ext
            tmp_output_file = Path(str(output_file).replace(ext, replace_to))
            # Create merger before to check if it can merge
            merger = self._create_merger(tmp_output_file, video_file, output_file)

            
            video_pipeline = self._create_video_pipeline(video_file, tmp_output_file)
            video_pipeline.run()

            # Rewrite video file with audio
            merger.merge()

            # Remove tmp file
            os.remove(tmp_output_file)
