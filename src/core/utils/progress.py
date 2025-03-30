"""Progress bar utility using rich library.

This module provides a flexible progress bar implementation using the rich library
for displaying progress in various scenarios like video processing, file operations,
and batch processing.
"""

from contextlib import contextmanager
from typing import Optional, Union, Dict, Any
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    ProgressColumn,
    Text,
)
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich import box
import time


class ProgressBar:
    """A flexible progress bar implementation using rich library.
    
    This class provides a unified interface for displaying progress in various scenarios:
    - Simple progress bar
    - Progress bar with multiple tasks
    - Progress bar with statistics
    - Progress bar with custom columns
    """

    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "Processing",
        show_time: bool = True,
        show_speed: bool = True,
        show_eta: bool = True,
        show_count: bool = True,
        show_percentage: bool = True,
        console: Optional[Console] = None,
        refresh_per_second: int = 10,
        transient: bool = False,
        expand: bool = False,
    ):
        """Initialize the progress bar.

        Args:
            total: Total number of steps (None for indeterminate)
            description: Description of the progress bar
            show_time: Whether to show elapsed time
            show_speed: Whether to show processing speed
            show_eta: Whether to show estimated time remaining
            show_count: Whether to show progress count (e.g., "5/10")
            show_percentage: Whether to show percentage
            console: Rich console instance (None for default)
            refresh_per_second: Number of updates per second
            transient: Whether to clear the progress bar when done
            expand: Whether to expand the progress bar to fill width
        """
        self.total = total
        self.description = description
        self.console = console or Console()
        self.refresh_per_second = refresh_per_second
        self.transient = transient
        self.expand = expand

        # Create columns based on options
        columns = []
        
        # Add spinner for indeterminate progress
        columns.append(SpinnerColumn())
        
        # Add description
        columns.append(TextColumn("[bold blue]{task.description}"))
        
        # Add progress bar
        columns.append(BarColumn(bar_width=None if expand else 40))
        
        # Add progress count
        if show_count:
            columns.append(MofNCompleteColumn())
        
        # Add percentage
        if show_percentage:
            columns.append(TaskProgressColumn())
        
        # Add time information
        if show_time:
            columns.append(TimeElapsedColumn())
            if show_eta:
                columns.append(TimeRemainingColumn())

        # Create progress instance
        self.progress = Progress(
            *columns,
            console=self.console,
            refresh_per_second=refresh_per_second,
            transient=transient,
            expand=expand,
        )

        # Initialize task
        self.task_id = None
        self._start_time = None
        self._stats: Dict[str, Any] = {}

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def start(self) -> None:
        """Start the progress bar."""
        self._start_time = time.time()
        self.progress.start()
        self.task_id = self.progress.add_task(
            self.description,
            total=self.total,
        )

    def stop(self) -> None:
        """Stop the progress bar."""
        if self.task_id is not None:
            self.progress.remove_task(self.task_id)
        self.progress.stop()

    def update(self, advance: int = 1) -> None:
        """Update the progress.

        Args:
            advance: Number of steps to advance
        """
        if self.task_id is not None:
            self.progress.update(self.task_id, advance=advance)

    def set_description(self, description: str) -> None:
        """Update the description.

        Args:
            description: New description
        """
        if self.task_id is not None:
            self.progress.update(self.task_id, description=description)

    def add_stat(self, name: str, value: Any) -> None:
        """Add or update a statistic.

        Args:
            name: Name of the statistic
            value: Value of the statistic
        """
        self._stats[name] = value

    def get_stats_table(self) -> Table:
        """Get a table of statistics.

        Returns:
            Table: Rich table containing statistics
        """
        table = Table(box=box.ROUNDED)
        table.add_column("Statistic", style="cyan")
        table.add_column("Value", style="green")
        
        for name, value in self._stats.items():
            table.add_row(name, str(value))
            
        return table

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            float: Elapsed time in seconds
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @contextmanager
    def with_stats_panel(self):
        """Context manager for displaying progress with statistics panel.
        
        This context manager creates a layout with the progress bar and a statistics panel.
        The statistics panel is updated in real-time as new stats are added.
        """
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=3),
            Layout(name="stats")
        )

        def make_layout() -> Layout:
            layout["progress"].update(Panel(self.progress))
            layout["stats"].update(
                Panel(
                    Align.center(self.get_stats_table()),
                    title="Statistics",
                    border_style="blue"
                )
            )
            return layout

        with Live(make_layout(), refresh_per_second=self.refresh_per_second) as live:
            self._live = live
            yield self
            self._live = None

    def update_stats_panel(self) -> None:
        """Update the statistics panel if it exists."""
        if hasattr(self, '_live') and self._live is not None:
            self._live.update(make_layout())


class MultiTaskProgress:
    """A progress bar implementation for multiple tasks.
    
    This class provides a unified interface for displaying progress of multiple
    related tasks, such as processing multiple files or stages of a pipeline.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        refresh_per_second: int = 10,
        transient: bool = False,
        expand: bool = False,
    ):
        """Initialize the multi-task progress bar.

        Args:
            console: Rich console instance (None for default)
            refresh_per_second: Number of updates per second
            transient: Whether to clear the progress bar when done
            expand: Whether to expand the progress bar to fill width
        """
        self.console = console or Console()
        self.refresh_per_second = refresh_per_second
        self.transient = transient
        self.expand = expand

        # Create progress instance with columns for multiple tasks
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None if expand else 40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=refresh_per_second,
            transient=transient,
            expand=expand,
        )

        self.tasks: Dict[str, int] = {}

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def start(self) -> None:
        """Start the progress bar."""
        self.progress.start()

    def stop(self) -> None:
        """Stop the progress bar."""
        self.progress.stop()

    def add_task(
        self,
        task_id: str,
        description: str,
        total: Optional[int] = None,
    ) -> None:
        """Add a new task.

        Args:
            task_id: Unique identifier for the task
            description: Description of the task
            total: Total number of steps (None for indeterminate)
        """
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")
        
        task = self.progress.add_task(description, total=total)
        self.tasks[task_id] = task

    def update_task(
        self,
        task_id: str,
        advance: int = 1,
        description: Optional[str] = None,
    ) -> None:
        """Update a task's progress.

        Args:
            task_id: Identifier of the task to update
            advance: Number of steps to advance
            description: New description for the task
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} does not exist")
        
        task = self.tasks[task_id]
        self.progress.update(
            task,
            advance=advance,
            description=description
        )

    def remove_task(self, task_id: str) -> None:
        """Remove a task.

        Args:
            task_id: Identifier of the task to remove
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} does not exist")
        
        task = self.tasks[task_id]
        self.progress.remove_task(task)
        del self.tasks[task_id] 