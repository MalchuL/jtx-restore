"""Table interface.

This module defines the interface for table implementations,
providing a common API for displaying tabular data in the terminal.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
from enum import Enum

from src.core.utils.drawable import Drawable


class TableStyle(Enum):
    """Table border styles."""
    ROUNDED = "rounded"
    SIMPLE = "simple"
    DOUBLE = "double"
    SINGLE = "single"
    NONE = "none"


class TableInterface(Drawable):
    """Interface for table implementations."""

    @abstractmethod
    def add_column(
        self,
        name: str,
        style: Optional[str] = None,
        justify: Optional[str] = None,
        width: Optional[int] = None,
        no_wrap: bool = False,
    ) -> None:
        """Add a column to the table.

        Args:
            name: Column name
            style: Optional style for the column
            justify: Optional justification (left, right, center)
            width: Optional column width
            no_wrap: Whether to prevent text wrapping
        """
        pass

    @abstractmethod
    def add_row(self, *cells: Any) -> None:
        """Add a row to the table.

        Args:
            *cells: Cell values for the row
        """
        pass

    @abstractmethod
    def add_rows(self, rows: List[List[Any]]) -> None:
        """Add multiple rows to the table.

        Args:
            rows: List of row cell values
        """
        pass

    @abstractmethod
    def add_section(
        self,
        title: str,
        style: Optional[str] = None,
    ) -> None:
        """Add a section to the table.

        Args:
            title: Section title
            style: Optional style for the section
        """
        pass

    @abstractmethod
    def add_footer(self, *cells: Any) -> None:
        """Add a footer row to the table.

        Args:
            *cells: Cell values for the footer
        """
        pass

    @abstractmethod
    def set_caption(self, caption: str) -> None:
        """Set the table caption.

        Args:
            caption: Table caption
        """
        pass

    @abstractmethod
    def set_title(self, title: str) -> None:
        """Set the table title.

        Args:
            title: Table title
        """
        pass

    @abstractmethod
    def set_style(self, style: TableStyle) -> None:
        """Set the table style.

        Args:
            style: Table style to use
        """
        pass

    @abstractmethod
    def get_row_count(self) -> int:
        """Get the number of rows in the table.

        Returns:
            int: Number of rows
        """
        pass

    @abstractmethod
    def get_column_count(self) -> int:
        """Get the number of columns in the table.

        Returns:
            int: Number of columns
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all rows from the table."""
        pass

    @abstractmethod
    def render(self) -> str:
        """Render the table to a string.

        Returns:
            str: Rendered table
        """
        pass 