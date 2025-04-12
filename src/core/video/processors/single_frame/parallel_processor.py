from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence
from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.single_frame.batch_processor import (
    BatchProcessor,
)


class ParallelProcessor(BatchProcessor):
    def __init__(self, num_workers: Optional[int] = None):
        super().__init__(batch_size=num_workers)
        self._executor = None

    @property   
    def num_workers(self) -> int:
        return self.batch_size

    def _get_executor(self) -> ProcessPoolExecutor:
        """Get or create a process pool executor.

        Returns:
            ProcessPoolExecutor: The executor instance
        """
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.num_workers)
        return self._executor

    def _get_parallel_kwargs(self, frame: ProcessedFrame) -> Dict[str, Any]:
        return {}

    @staticmethod
    @abstractmethod
    def _process_frame_parallel(frame: ProcessedFrame, **kwargs) -> ProcessedFrame:
        pass

    def _process_single_window(
        self, window: Sequence[ProcessedFrame]
    ) -> List[ProcessedFrame]:
        
        frames = window
        # If there is only one worker or the number of frames, process sequentially
        if 0 <= self.num_workers <= 1 or len(frames) < 3:
            results = []
            for frame in frames:
                processed = self._process_frame_parallel(
                    frame, **self._get_parallel_kwargs(frame)
                )
                results.append(processed)
            return results

        # If there are multiple workers, process in parallel
        executor = self._get_executor()
        futures = [
            executor.submit(
                self._process_frame_parallel, frame, **self._get_parallel_kwargs(frame)
            )
            for frame in frames
        ]

        results = []
        for future in futures:
            result = future.result()
            if result is None:
                raise ValueError(f"{self.__class__.__name__} returned None.")
            results.append(result)

        return results

    def reset(self) -> None:
        super().reset()
        self._executor = None
    
    def __del__(self):
        """Clean up the process pool executor."""
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown(wait=True)
