"""Rich progress bar implementation.

This module provides a Rich-based implementation of the progress bar interface.
"""

from typing import Optional
from contextlib import contextmanager
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)
from rich.live import Live
from rich.console import Console

from src.core.utils.progress_interface import ProgressInterface


class RichProgressBar(ProgressInterface):
    """Rich-based progress bar implementation."""

    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "Processing",
        show_time: bool = True,
        show_speed: bool = True,
        show_eta: bool = True,
        show_count: bool = True,
        show_percentage: bool = True,
        refresh_per_second: int = 10,
        transient: bool = False,
        expand: bool = False,
    ) -> None:
        """Initialize the progress bar.

        Args:
            total: Total number of steps (None for indeterminate)
            description: Description of the progress bar
            show_time: Whether to show elapsed time
            show_speed: Whether to show processing speed
            show_eta: Whether to show estimated time remaining
            show_count: Whether to show progress count (e.g., "5/10")
            show_percentage: Whether to show percentage
            refresh_per_second: Number of updates per second
            transient: Whether to clear the progress bar when done
            expand: Whether to expand the progress bar to fill width
        """
        self._total = total
        self._description = description
        self._show_time = show_time
        self._show_speed = show_speed
        self._show_eta = show_eta
        self._show_count = show_count
        self._show_percentage = show_percentage
        self._refresh_per_second = refresh_per_second
        self._transient = transient
        self._expand = expand
        self._progress = None
        self._task_id = None
        self._live = None
        self._console = Console()

    def start(self) -> None:
        """Start the progress bar."""
        if self._progress is not None:
            return

        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bright_green"),
        ]

        if self._show_time:
            columns.append(TimeElapsedColumn())
        if self._show_eta:
            columns.append(TimeRemainingColumn())
        if self._show_count:
            columns.append(MofNCompleteColumn())
        if self._show_percentage:
            columns.append(TaskProgressColumn())

        self._progress = Progress(
            *columns,
            refresh_per_second=self._refresh_per_second,
            transient=self._transient,
            expand=self._expand,
            console=self._console,
        )
        self._task_id = self._progress.add_task(
            self._description,
            total=self._total,
        )

    def stop(self) -> None:
        """Stop the progress bar."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def update(self, advance: int = 1) -> None:
        """Update the progress.

        Args:
            advance: Number of steps to advance
        """
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=advance)
            if self._live:
                self._live.refresh()

    def set_description(self, description: str) -> None:
        """Update the description.

        Args:
            description: New description
        """
        self._description = description
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, description=description)

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            float: Elapsed time in seconds
        """
        if self._progress and self._task_id is not None:
            return self._progress.tasks[self._task_id].elapsed
        return 0.0

    def __enter__(self):
        """Start the progress bar display.

        Returns:
            RichProgressBar: The progress bar instance
        """
        if not self._progress:
            self.start()

        self._live = Live(
            self._progress,
            refresh_per_second=self._refresh_per_second,
            transient=self._transient,
            console=self._console,
            auto_refresh=True,
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the progress bar display."""
        if self._live:
            self._live.stop()
            self._live = None

    def render(self) -> str:
        """Render the progress bar to a string.

        Returns:
            str: Rendered progress bar
        """
        if not self._progress:
            return ""
        return str(self._progress)

    def clear(self) -> None:
        """Clear the progress bar."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def is_empty(self) -> bool:
        """Check if the progress bar is empty.

        Returns:
            bool: True if empty
        """
        return not self._progress or self._task_id is None

    def get_width(self) -> int:
        """Get the width of the progress bar.

        Returns:
            int: Width in characters
        """
        if not self._progress:
            return 0
        return self._progress.width

    def get_height(self) -> int:
        """Get the height of the progress bar.

        Returns:
            int: Height in characters
        """
        if not self._progress:
            return 0
        return self._progress.height 