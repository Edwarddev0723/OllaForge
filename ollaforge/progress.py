"""
Progress tracking and user feedback for OllaForge.

This module provides Rich-based progress tracking, error display, and summary reporting
for the dataset generation process.
"""

import time
from typing import List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.align import Align

from .models import GenerationResult


class ProgressTracker:
    """
    Handles progress tracking and user feedback during dataset generation.
    
    Provides Rich-based progress bars, error messages, and summary displays
    with proper formatting and color coding.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the progress tracker.
        
        Args:
            console: Rich console instance (creates new one if None)
        """
        self.console = console or Console()
        self.progress: Optional[Progress] = None
        self.task_id: Optional[int] = None
        self.start_time: Optional[float] = None
        self.errors: List[str] = []
    
    def start_progress(self, total: int, description: str = "Generating") -> None:
        """
        Start the progress tracking display.
        
        Args:
            total: Total number of items to process
            description: Description text for the progress bar
        """
        self.start_time = time.time()
        self.errors = []
        
        # Create progress bar with multiple columns
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False
        )
        
        self.progress.start()
        self.task_id = self.progress.add_task(description, total=total)
    
    def update_progress(self, advance: int = 1, description: Optional[str] = None) -> None:
        """
        Update the progress bar.
        
        Args:
            advance: Number of items to advance (default: 1)
            description: Optional new description text
        """
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, advance=advance)
            if description:
                self.progress.update(self.task_id, description=description)
    
    def add_error(self, error_message: str) -> None:
        """
        Add an error message to the error list.
        
        Args:
            error_message: Error message to record
        """
        self.errors.append(error_message)
    
    def display_error(self, error_message: str, show_immediately: bool = True) -> None:
        """
        Display a colored error message.
        
        Args:
            error_message: Error message to display
            show_immediately: Whether to display immediately or just record
        """
        self.add_error(error_message)
        
        if show_immediately:
            self.console.print(f"[red]❌ Error: {error_message}[/red]")
    
    def stop_progress(self) -> float:
        """
        Stop the progress tracking and return elapsed time.
        
        Returns:
            float: Elapsed time in seconds
        """
        elapsed_time = 0.0
        
        if self.start_time:
            elapsed_time = time.time() - self.start_time
        
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.task_id = None
        
        return elapsed_time
    
    def display_summary(self, result: GenerationResult) -> None:
        """
        Display a comprehensive summary of the generation process.
        
        Args:
            result: GenerationResult containing operation statistics
        """
        # Create summary table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="bold cyan")
        table.add_column("Value", style="white")
        
        # Add summary rows
        table.add_row("📊 Total Requested:", str(result.total_requested))
        table.add_row("✅ Successfully Generated:", str(result.success_count))
        table.add_row("📈 Success Rate:", f"{result.success_rate:.1f}%")
        table.add_row("⏱️  Total Time:", f"{result.duration:.2f}s")
        table.add_row("📄 Output File:", result.output_file)
        
        if result.errors:
            table.add_row("❌ Errors Encountered:", str(len(result.errors)))
        
        # Display summary in a panel
        summary_panel = Panel(
            Align.center(table),
            title="[bold green]🎉 Generation Complete[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(summary_panel)
        
        # Display errors if any
        if result.errors:
            self.console.print()
            self.console.print("[yellow]⚠️  Errors encountered during generation:[/yellow]")
            for i, error in enumerate(result.errors[:5], 1):  # Show max 5 errors
                self.console.print(f"  {i}. [red]{error}[/red]")
            
            if len(result.errors) > 5:
                remaining = len(result.errors) - 5
                self.console.print(f"  ... and {remaining} more error(s)")
    
    def get_current_progress(self) -> tuple[int, int]:
        """
        Get current progress information.
        
        Returns:
            tuple: (current_count, total_count)
        """
        if self.progress and self.task_id is not None:
            task = self.progress.tasks[self.task_id]
            return int(task.completed), int(task.total or 0)
        return 0, 0
    
    def is_active(self) -> bool:
        """
        Check if progress tracking is currently active.
        
        Returns:
            bool: True if progress tracking is active
        """
        return self.progress is not None and self.task_id is not None