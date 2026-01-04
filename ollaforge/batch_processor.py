"""
Batch processing module for OllaForge document-to-dataset conversion.

This module provides functionality for processing multiple documents in batch,
including directory traversal, pattern filtering, and result aggregation.

Key features:
- Recursive directory traversal for supported file formats
- Glob pattern filtering for file selection
- Result aggregation from multiple files
- Graceful handling of individual file failures

Requirements satisfied:
- 6.1: Accept directory path to process all supported files
- 6.2: Recursively find all supported document files
- 6.3: Filter files by glob pattern
- 6.5: Combine results from all files into single output
- 6.6: Continue processing on individual file failures
"""

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .doc_parser import DocumentParserFactory, UnsupportedFormatError
from .models import BatchProcessingResult, DocProcessingResult


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    recursive: bool = True
    file_pattern: Optional[str] = None
    continue_on_error: bool = True


def collect_supported_files(
    directory: Path, pattern: Optional[str] = None, recursive: bool = True
) -> list[Path]:
    """
    Collect all supported files from a directory.

    This function traverses a directory (optionally recursively) and returns
    all files that match the optional pattern and are supported by the
    document parser factory.

    Args:
        directory: Path to the directory to search
        pattern: Optional glob pattern to filter files (e.g., "*.md", "*.py")
        recursive: Whether to search subdirectories recursively

    Returns:
        List of Path objects for all matching supported files, sorted alphabetically

    Requirements satisfied:
    - 6.1: Accept directory path to process all supported files
    - 6.2: Recursively find all supported document files
    - 6.3: Filter files by glob pattern
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    files: list[Path] = []

    # Get all files based on recursive setting
    if recursive:
        all_files = directory.rglob("*")
    else:
        all_files = directory.glob("*")

    for file_path in all_files:
        # Skip directories
        if not file_path.is_file():
            continue

        # Apply pattern filter if specified
        if pattern and not matches_pattern(file_path.name, pattern):
            continue

        # Check if format is supported by the parser factory
        if DocumentParserFactory.is_supported(str(file_path)):
            files.append(file_path)

    # Return sorted list for consistent ordering
    return sorted(files)


def matches_pattern(filename: str, pattern: str) -> bool:
    """
    Check if a filename matches a glob pattern.

    This function uses fnmatch for glob-style pattern matching.
    Supports wildcards like *, ?, and [seq].

    Args:
        filename: The filename to check (not the full path)
        pattern: The glob pattern to match against

    Returns:
        True if the filename matches the pattern, False otherwise

    Requirements satisfied:
    - 6.3: Filter files by glob pattern

    Examples:
        >>> matches_pattern("readme.md", "*.md")
        True
        >>> matches_pattern("readme.txt", "*.md")
        False
        >>> matches_pattern("test_file.py", "test_*.py")
        True
    """
    return fnmatch.fnmatch(filename, pattern)


def filter_files_by_pattern(files: list[Path], pattern: str) -> list[Path]:
    """
    Filter a list of files by a glob pattern.

    Args:
        files: List of file paths to filter
        pattern: Glob pattern to match against filenames

    Returns:
        List of files that match the pattern

    Requirements satisfied:
    - 6.3: Filter files by glob pattern
    """
    return [f for f in files if matches_pattern(f.name, pattern)]


def aggregate_results(
    file_results: list[DocProcessingResult],
    all_entries: list,
    output_file: str,
    duration: float,
) -> BatchProcessingResult:
    """
    Aggregate results from multiple file processing operations.

    This function combines the results from processing multiple files into
    a single BatchProcessingResult, calculating totals and collecting errors.

    Args:
        file_results: List of per-file processing results
        all_entries: List of all generated dataset entries
        output_file: Path to the output file
        duration: Total processing duration in seconds

    Returns:
        BatchProcessingResult with aggregated statistics

    Requirements satisfied:
    - 6.5: Combine results from all files into single output
    - 6.6: Report failures at the end
    """
    total_files = len(file_results)
    successful_files = sum(1 for r in file_results if r.success)
    failed_files = sum(1 for r in file_results if not r.success)
    total_entries = len(all_entries)

    # Collect all global errors (errors not associated with specific files)
    global_errors: list[str] = []

    return BatchProcessingResult(
        total_files=total_files,
        successful_files=successful_files,
        failed_files=failed_files,
        total_entries=total_entries,
        output_file=output_file,
        duration=duration,
        file_results=file_results,
        errors=global_errors,
    )


def merge_entries(entries_lists: list[list]) -> list:
    """
    Merge multiple lists of entries into a single list.

    This function flattens a list of entry lists into a single list,
    preserving the order of entries.

    Args:
        entries_lists: List of lists of entries to merge

    Returns:
        Single flattened list of all entries

    Requirements satisfied:
    - 6.5: Combine results from all files into single output
    """
    merged: list = []
    for entries in entries_lists:
        if entries:
            merged.extend(entries)
    return merged


class BatchProcessor:
    """
    Batch processor for document-to-dataset conversion.

    This class orchestrates the batch processing of multiple documents,
    handling file collection, processing, and result aggregation.

    Requirements satisfied:
    - 6.1: Accept directory path to process all supported files
    - 6.2: Recursively find all supported document files
    - 6.3: Filter files by glob pattern
    - 6.5: Combine results from all files into single output
    - 6.6: Continue processing on individual file failures
    """

    def __init__(self, config: BatchConfig = None):
        """
        Initialize the batch processor.

        Args:
            config: Optional batch configuration
        """
        self.config = config or BatchConfig()
        self._file_results: list[DocProcessingResult] = []
        self._all_entries: list = []

    def collect_files(self, source: Path) -> list[Path]:
        """
        Collect files to process from a source path.

        If source is a file, returns a list containing just that file.
        If source is a directory, collects all supported files.

        Args:
            source: Path to file or directory

        Returns:
            List of files to process
        """
        if source.is_file():
            # Single file - check if supported
            if DocumentParserFactory.is_supported(str(source)):
                return [source]
            else:
                raise UnsupportedFormatError(
                    f"Unsupported format: {source.suffix}",
                    supported_formats=DocumentParserFactory.get_supported_formats(),
                )
        elif source.is_dir():
            return collect_supported_files(
                source,
                pattern=self.config.file_pattern,
                recursive=self.config.recursive,
            )
        else:
            raise FileNotFoundError(f"Source not found: {source}")

    def add_file_result(self, result: DocProcessingResult) -> None:
        """
        Add a file processing result.

        Args:
            result: The processing result for a single file
        """
        self._file_results.append(result)

    def add_entries(self, entries: list) -> None:
        """
        Add generated entries to the collection.

        Args:
            entries: List of generated entries
        """
        if entries:
            self._all_entries.extend(entries)

    def get_results(self, output_file: str, duration: float) -> BatchProcessingResult:
        """
        Get the aggregated batch processing results.

        Args:
            output_file: Path to the output file
            duration: Total processing duration

        Returns:
            Aggregated batch processing result
        """
        return aggregate_results(
            self._file_results, self._all_entries, output_file, duration
        )

    def get_entries(self) -> list:
        """
        Get all collected entries.

        Returns:
            List of all generated entries
        """
        return self._all_entries

    def get_file_results(self) -> list[DocProcessingResult]:
        """
        Get all file processing results.

        Returns:
            List of per-file processing results
        """
        return self._file_results

    def reset(self) -> None:
        """Reset the processor state for a new batch."""
        self._file_results = []
        self._all_entries = []


def get_supported_extensions() -> list[str]:
    """
    Get list of all supported file extensions.

    Returns:
        List of supported file extensions (with leading dots)
    """
    return DocumentParserFactory.get_supported_formats()
