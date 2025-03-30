"""Base class for drawable interfaces.

This module defines the base Drawable class that all rendering interfaces
must inherit from, providing common functionality and type hints.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class Drawable(ABC):
    """Base class for all drawable interfaces.
    
    This class provides the common interface that all rendering implementations
    must follow, ensuring consistent behavior across different backends.
    """

    @abstractmethod
    def render(self) -> str:
        """Render the drawable to a string.

        Returns:
            str: Rendered output
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the drawable content."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the drawable is empty.

        Returns:
            bool: True if empty, False otherwise
        """
        pass

    @abstractmethod
    def get_width(self) -> Optional[int]:
        """Get the width of the drawable.

        Returns:
            Optional[int]: Width in characters, or None if not applicable
        """
        pass

    @abstractmethod
    def get_height(self) -> Optional[int]:
        """Get the height of the drawable.

        Returns:
            Optional[int]: Height in lines, or None if not applicable
        """
        pass 