from abc import ABC, abstractmethod
from pathlib import Path


class VideoAudioMerger(ABC):
    """Abstract base class for merging video and audio files.
    
    This class defines the interface for any implementation that combines
    video and audio streams into a single output file.
    """
    
    def __init__(self, output_path: Path | str):
        """Initialize the video audio merger.
        
        Args:
            output_path: Path where the merged file will be saved
        """
        self._output_path = Path(output_path)
    
    @property
    def output_path(self) -> Path:
        """Path where the merged file will be saved.
        
        Returns:
            Path: Output path for the merged file
        """
        return self._output_path
    
    @abstractmethod
    def merge(self) -> Path:
        """Merge video and audio files into a single output file.
        
        Returns:
            Path: Path to the merged output file
            
        Raises:
            FileNotFoundError: If source files don't exist
            ValueError: If provided files are invalid or incompatible
            RuntimeError: If merging process fails
        """
        pass
    