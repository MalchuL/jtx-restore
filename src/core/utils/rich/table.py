"""Rich table implementation.

This module provides a Rich-based implementation of the table interface.
"""

from typing import Any, List, Optional
from rich.table import Table as RichTable
from rich.console import Console
from rich import box

from src.core.utils.table_interface import TableInterface, TableStyle


class RichTableImpl(TableInterface):
    """Rich-based table implementation."""

    def __init__(
        self,
        style: TableStyle = TableStyle.ROUNDED,
        title: Optional[str] = None,
        caption: Optional[str] = None,
        show_header: bool = True,
        show_footer: bool = False,
        expand: bool = False,
    ) -> None:
        """Initialize the table.

        Args:
            style: Table style to use
            title: Optional table title
            caption: Optional table caption
            show_header: Whether to show the header
            show_footer: Whether to show the footer
            expand: Whether to expand the table to fill width
        """
        self._style = style
        self._title = title
        self._caption = caption
        self._show_header = show_header
        self._show_footer = show_footer
        self._expand = expand
        self._console = Console()
        self._table = RichTable(
            title=title,
            caption=caption,
            show_header=show_header,
            show_footer=show_footer,
            expand=expand,
        )
        # Set the box style after creating the table
        self._table.box = self._get_box_style(style)

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
        self._table.add_column(
            name,
            style=style,
            justify=justify,
            width=width,
            no_wrap=no_wrap,
        )

    def add_row(self, *cells: Any) -> None:
        """Add a row to the table.

        Args:
            *cells: Cell values for the row
        """
        self._table.add_row(*cells)

    def add_rows(self, rows: List[List[Any]]) -> None:
        """Add multiple rows to the table.

        Args:
            rows: List of row cell values
        """
        for row in rows:
            self._table.add_row(*row)

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
        # Create a section header row that spans all columns
        header_style = style or "bold cyan"
        self._table.add_row(
            f"[{header_style}]{title}[/]",
            style=header_style,
        )

    def add_footer(self, *cells: Any) -> None:
        """Add a footer row to the table.

        Args:
            *cells: Cell values for the footer
        """
        self._table.add_footer(*cells)

    def set_caption(self, caption: str) -> None:
        """Set the table caption.

        Args:
            caption: Table caption
        """
        self._caption = caption
        self._table.caption = caption

    def set_title(self, title: str) -> None:
        """Set the table title.

        Args:
            title: Table title
        """
        self._title = title
        self._table.title = title

    def set_style(self, style: TableStyle) -> None:
        """Set the table style.

        Args:
            style: Table style to use
        """
        self._style = style
        self._table.box = self._get_box_style(style)

    def get_row_count(self) -> int:
        """Get the number of rows in the table.

        Returns:
            int: Number of rows
        """
        return len(self._table.rows)

    def get_column_count(self) -> int:
        """Get the number of columns in the table.

        Returns:
            int: Number of columns
        """
        return len(self._table.columns)

    def clear(self) -> None:
        """Clear all rows from the table."""
        self._table.rows.clear()

    def render(self) -> str:
        """Render the table to a string.

        Returns:
            str: Rendered table
        """
        return str(self._table)

    def is_empty(self) -> bool:
        """Check if the table is empty.

        Returns:
            bool: True if empty
        """
        return len(self._table.rows) == 0

    def get_width(self) -> int:
        """Get the width of the table.

        Returns:
            int: Width in characters
        """
        if not self._table.columns:
            return 0
        return sum(col.width or 0 for col in self._table.columns)

    def get_height(self) -> int:
        """Get the height of the table.

        Returns:
            int: Height in characters
        """
        return len(self._table.rows) + (2 if self._show_header else 0) + (2 if self._show_footer else 0)

    def _get_box_style(self, style: TableStyle) -> box.Box:
        """Get the Rich box style from TableStyle.

        Args:
            style: Table style to convert

        Returns:
            box.Box: Rich box style
        """
        return {
            TableStyle.ROUNDED: box.ROUNDED,
            TableStyle.SIMPLE: box.SIMPLE,
            TableStyle.DOUBLE: box.DOUBLE,
            TableStyle.SINGLE: box.SIMPLE,
            TableStyle.NONE: None,
        }[style] 