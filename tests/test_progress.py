"""
Tests for OllaForge progress tracking functionality.
"""

import pytest
import time
from io import StringIO
from hypothesis import given, strategies as st
from rich.console import Console

from ollaforge.progress import ProgressTracker
from ollaforge.models import GenerationResult


@given(
    total_count=st.integers(min_value=1, max_value=1000),
    advance_steps=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=50)
)
def test_progress_tracking_updates(total_count, advance_steps):
    """
    **Feature: ollama-cli-generator, Property 12: Progress tracking updates**
    **Validates: Requirements 4.2**
    
    For any generation process, progress indicators should update at appropriate 
    intervals showing current and total counts.
    """
    # Create a string buffer to capture console output
    output_buffer = StringIO()
    console = Console(file=output_buffer, width=80)
    
    # Initialize progress tracker
    tracker = ProgressTracker(console)
    
    # Start progress tracking
    tracker.start_progress(total_count, "Testing progress")
    
    # Verify progress is active
    assert tracker.is_active()
    
    # Track progress updates
    total_advanced = 0
    for advance in advance_steps:
        if total_advanced + advance <= total_count:
            tracker.update_progress(advance)
            total_advanced += advance
            
            # Check current progress
            current, total = tracker.get_current_progress()
            assert current == total_advanced
            assert total == total_count
        else:
            # Don't advance beyond total
            break
    
    # Stop progress tracking
    elapsed_time = tracker.stop_progress()
    
    # Verify progress is no longer active
    assert not tracker.is_active()
    
    # Verify elapsed time is reasonable
    assert elapsed_time >= 0
    
    # Verify output contains progress information
    output = output_buffer.getvalue()
    assert "Testing progress" in output or len(output) > 0  # Progress bar was displayed


@given(
    success_count=st.integers(min_value=0, max_value=1000),
    total_requested=st.integers(min_value=1, max_value=1000),
    duration=st.floats(min_value=0.1, max_value=3600.0),
    output_file=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='.-_'), min_size=1, max_size=50).filter(lambda x: x.strip()),
    error_count=st.integers(min_value=0, max_value=20)
)
def test_summary_information_completeness(success_count, total_requested, duration, output_file, error_count):
    """
    **Feature: ollama-cli-generator, Property 13: Summary information completeness**
    **Validates: Requirements 4.3**
    
    For any completed generation, the summary should include total time, 
    successful entry count, and output file path.
    """
    # Ensure success_count doesn't exceed total_requested
    success_count = min(success_count, total_requested)
    
    # Create a string buffer to capture console output
    output_buffer = StringIO()
    console = Console(file=output_buffer, width=120)
    
    # Initialize progress tracker
    tracker = ProgressTracker(console)
    
    # Create generation result with test data
    errors = [f"Error {i+1}" for i in range(error_count)]
    result = GenerationResult(
        success_count=success_count,
        total_requested=total_requested,
        output_file=output_file.strip(),
        duration=duration,
        errors=errors
    )
    
    # Display summary
    tracker.display_summary(result)
    
    # Capture output
    output = output_buffer.getvalue()
    
    # Verify all required information is present in the summary
    assert str(total_requested) in output  # Total requested count
    assert str(success_count) in output    # Successful entry count
    assert f"{duration:.2f}s" in output    # Duration with proper formatting
    assert output_file.strip() in output   # Output file path
    
    # Verify success rate is displayed
    expected_rate = (success_count / total_requested) * 100 if total_requested > 0 else 0
    assert f"{expected_rate:.1f}%" in output
    
    # If there are errors, verify error count is displayed
    if error_count > 0:
        assert str(error_count) in output or "Error" in output


def test_progress_tracker_initialization():
    """Test progress tracker initialization and basic functionality."""
    # Test with default console
    tracker1 = ProgressTracker()
    assert tracker1.console is not None
    assert not tracker1.is_active()
    
    # Test with custom console
    custom_console = Console()
    tracker2 = ProgressTracker(custom_console)
    assert tracker2.console is custom_console
    assert not tracker2.is_active()


def test_error_handling():
    """Test error message handling and display."""
    output_buffer = StringIO()
    console = Console(file=output_buffer, width=80)
    tracker = ProgressTracker(console)
    
    # Add errors
    tracker.add_error("Test error 1")
    tracker.display_error("Test error 2", show_immediately=False)
    
    # Verify errors are recorded
    assert len(tracker.errors) == 2
    assert "Test error 1" in tracker.errors
    assert "Test error 2" in tracker.errors


def test_progress_lifecycle():
    """Test complete progress tracking lifecycle."""
    output_buffer = StringIO()
    console = Console(file=output_buffer, width=80)
    tracker = ProgressTracker(console)
    
    # Initially not active
    assert not tracker.is_active()
    
    # Start progress
    tracker.start_progress(10, "Test task")
    assert tracker.is_active()
    
    # Update progress
    tracker.update_progress(3)
    current, total = tracker.get_current_progress()
    assert current == 3
    assert total == 10
    
    # Update with description
    tracker.update_progress(2, "Updated description")
    current, total = tracker.get_current_progress()
    assert current == 5
    assert total == 10
    
    # Stop progress
    elapsed = tracker.stop_progress()
    assert not tracker.is_active()
    assert elapsed >= 0