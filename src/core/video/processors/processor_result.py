from dataclasses import dataclass, field
from typing import List
from src.core.video.processors.frame import ProcessedFrame


@dataclass
class ProcessorResult:
    frames: List[ProcessedFrame] = field(default_factory=list)
    ready: bool = True
