"""Rich text implementation.

This module provides a Rich-based implementation of the text interface.
"""

from typing import Optional
from rich.text import Text as RichText
from rich.console import Console
from rich.align import Align
from rich.panel import Panel

from src.core.utils.text_interface import (
    TextInterface,
    TextStyle,
    TextColor,
    TextBackground,
    TextJustify,
    TextAlign,
    TextOverflow,
)


class RichText(TextInterface):
    """Rich-based text implementation."""

    def __init__(
        self,
        text: str = "",
        style: Optional[TextStyle] = None,
        color: Optional[TextColor] = None,
        background: Optional[TextBackground] = None,
        justify: Optional[TextJustify] = None,
        align: Optional[TextAlign] = None,
        overflow: Optional[TextOverflow] = None,
        width: Optional[int] = None,
    ) -> None:
        """Initialize the text.

        Args:
            text: Text content
            style: Optional text style
            color: Optional text color
            background: Optional background color
            justify: Optional text justification
            align: Optional text alignment
            overflow: Optional text overflow behavior
            width: Optional text width
        """
        self._text = RichText(text)
        self._console = Console()
        self._width = width
        self._style = style
        self._color = color
        self._background = background
        self._justify = justify
        self._align = align
        self._overflow = overflow

        if style:
            self.set_style(style)
        if color:
            self.set_color(color)
        if background:
            self.set_background(background)
        if justify:
            self.set_justify(justify)
        if align:
            self.set_align(align)
        if overflow:
            self.set_overflow(overflow)

    def set_text(self, text: str) -> None:
        """Set the text content.

        Args:
            text: Text content to set
        """
        self._text = RichText(text)

    def set_style(self, style: TextStyle) -> None:
        """Set the text style.

        Args:
            style: Text style to use
        """
        self._style = style
        style_map = {
            TextStyle.PLAIN: "",
            TextStyle.BOLD: "bold",
            TextStyle.ITALIC: "italic",
            TextStyle.UNDERLINE: "underline",
            TextStyle.STRIKETHROUGH: "strikethrough",
        }
        self._text.style = style_map[style]

    def set_color(self, color: Optional[TextColor]) -> None:
        """Set the text color.

        Args:
            color: Text color to use
        """
        self._color = color
        if color:
            self._text.style = f"{self._text.style} {color.value}"
        else:
            self._text.style = self._text.style.replace(" color", "")

    def set_background(self, background: Optional[TextBackground]) -> None:
        """Set the background color.

        Args:
            background: Background color to use
        """
        self._background = background
        if background:
            self._text.style = f"{self._text.style} on_{background.value}"
        else:
            self._text.style = self._text.style.replace(" on_", "")

    def set_justify(self, justify: TextJustify) -> None:
        """Set the text justification.

        Args:
            justify: Text justification to use
        """
        self._justify = justify
        self._text.justify = justify.value

    def set_align(self, align: TextAlign) -> None:
        """Set the text alignment.

        Args:
            align: Text alignment to use
        """
        self._align = align
        self._text.align = align.value

    def set_overflow(self, overflow: TextOverflow) -> None:
        """Set the text overflow behavior.

        Args:
            overflow: Text overflow behavior to use
        """
        self._overflow = overflow
        self._text.overflow = overflow.value

    def set_width(self, width: Optional[int]) -> None:
        """Set the text width.

        Args:
            width: Width in characters, or None for auto
        """
        self._width = width

    def append(self, text: str) -> None:
        """Append text to the current content.

        Args:
            text: Text to append
        """
        self._text.append(text)

    def render(self) -> str:
        """Render the text to a string.

        Returns:
            str: Rendered text
        """
        # Create the text content
        content = self._text

        # Handle overflow
        if self._overflow == TextOverflow.WRAP:
            content = content.wrap(self._width) if self._width else content
        elif self._overflow == TextOverflow.ELLIPSIS:
            content = content.ellipsis(self._width) if self._width else content
        elif self._overflow == TextOverflow.CROP:
            content = content.crop(self._width) if self._width else content

        # Create panel with justification
        panel = Panel(
            content,
            expand=False,
            width=self._width,
            box=None,
            padding=(0, 1),
        )

        # Apply alignment
        if self._align == TextAlign.LEFT:
            return str(panel)
        elif self._align == TextAlign.CENTER:
            return str(Align.center(panel))
        else:  # RIGHT
            return str(Align.right(panel))

    def clear(self) -> None:
        """Clear the text."""
        self._text = RichText()

    def is_empty(self) -> bool:
        """Check if the text is empty.

        Returns:
            bool: True if empty
        """
        return len(self._text) == 0

    def get_width(self) -> int:
        """Get the width of the text.

        Returns:
            int: Width in characters
        """
        if self._width is not None:
            return self._width
        return self._console.width

    def get_height(self) -> int:
        """Get the height of the text.

        Returns:
            int: Height in characters
        """
        if not self._width:
            return len(self._text.split("\n"))
        return len(self._text.wrap(self._width).split("\n")) 