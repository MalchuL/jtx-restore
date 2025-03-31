"""Progress bar interface.

This module defines the interface for progress bar implementations,
providing a common API for displaying progress in the terminal.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .drawable import Drawable


class ProgressInterface(Drawable):
    """Interface for progress bar implementations."""

    @abstractmethod
    def start(self) -> None:
        """Start the progress bar."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the progress bar."""
        pass

    @abstractmethod
    def update(self, advance: int = 1) -> None:
        """Update the progress.

        Args:
            advance: Number of steps to advance
        """
        pass

    @abstractmethod
    def set_description(self, description: str) -> None:
        """Update the description.

        Args:
            description: New description
        """
        pass

    @abstractmethod
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            float: Elapsed time in seconds
        """
        pass

    @abstractmethod
    def __enter__(self):
        """Start the progress bar display.

        Returns:
            ProgressInterface: The progress bar instance
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the progress bar display."""
        pass

    @abstractmethod
    def render(self) -> str:
        """Render the progress bar to a string.

        Returns:
            str: Rendered progress bar
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the progress bar."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the progress bar is empty.

        Returns:
            bool: True if empty
        """
        pass

    @abstractmethod
    def get_width(self) -> int:
        """Get the width of the progress bar.

        Returns:
            int: Width in characters
        """
        pass

    @abstractmethod
    def get_height(self) -> int:
        """Get the height of the progress bar.

        Returns:
            int: Height in characters
        """
        pass 