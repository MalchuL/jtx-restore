
from dataclasses import MISSING, dataclass


@dataclass
class Pipeline:
    task_name: str = MISSING
    exp_name: str = MISSING
