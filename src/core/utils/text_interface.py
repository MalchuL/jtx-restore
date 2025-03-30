"""Text interface.

This module defines the interface for text implementations,
providing a common API for displaying styled text in the terminal.
"""

from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum

from src.core.utils.drawable import Drawable


class TextStyle(Enum):
    """Text styles."""
    PLAIN = "plain"
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
    STRIKETHROUGH = "strikethrough"


class TextColor(Enum):
    """Text colors."""
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"
    BRIGHT_BLACK = "bright_black"
    BRIGHT_RED = "bright_red"
    BRIGHT_GREEN = "bright_green"
    BRIGHT_YELLOW = "bright_yellow"
    BRIGHT_BLUE = "bright_blue"
    BRIGHT_MAGENTA = "bright_magenta"
    BRIGHT_CYAN = "bright_cyan"
    BRIGHT_WHITE = "bright_white"


class TextBackground(Enum):
    """Text background colors."""
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"
    BRIGHT_BLACK = "bright_black"
    BRIGHT_RED = "bright_red"
    BRIGHT_GREEN = "bright_green"
    BRIGHT_YELLOW = "bright_yellow"
    BRIGHT_BLUE = "bright_blue"
    BRIGHT_MAGENTA = "bright_magenta"
    BRIGHT_CYAN = "bright_cyan"
    BRIGHT_WHITE = "bright_white"


class TextJustify(Enum):
    """Text justification."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class TextAlign(Enum):
    """Text alignment."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class TextOverflow(Enum):
    """Text overflow behavior."""
    WRAP = "wrap"
    ELLIPSIS = "ellipsis"
    CROP = "crop"


class TextInterface(Drawable):
    """Interface for text implementations."""

    @abstractmethod
    def set_text(self, text: str) -> None:
        """Set the text content.

        Args:
            text: Text content to set
        """
        pass

    @abstractmethod
    def set_style(self, style: TextStyle) -> None:
        """Set the text style.

        Args:
            style: Text style to use
        """
        pass

    @abstractmethod
    def set_color(self, color: Optional[TextColor]) -> None:
        """Set the text color.

        Args:
            color: Text color to use
        """
        pass

    @abstractmethod
    def set_background(self, background: Optional[TextBackground]) -> None:
        """Set the background color.

        Args:
            background: Background color to use
        """
        pass

    @abstractmethod
    def set_justify(self, justify: TextJustify) -> None:
        """Set the text justification.

        Args:
            justify: Text justification to use
        """
        pass

    @abstractmethod
    def set_align(self, align: TextAlign) -> None:
        """Set the text alignment.

        Args:
            align: Text alignment to use
        """
        pass

    @abstractmethod
    def set_overflow(self, overflow: TextOverflow) -> None:
        """Set the text overflow behavior.

        Args:
            overflow: Text overflow behavior to use
        """
        pass

    @abstractmethod
    def set_width(self, width: Optional[int]) -> None:
        """Set the text width.

        Args:
            width: Width in characters, or None for auto
        """
        pass

    @abstractmethod
    def append(self, text: str) -> None:
        """Append text to the current content.

        Args:
            text: Text to append
        """
        pass 