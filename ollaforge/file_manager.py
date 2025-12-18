"""
File operations and JSONL output management for OllaForge.

This module provides robust file operations with atomic writes, disk space checking,
interruption handling, and comprehensive error recovery. It ensures data integrity
and provides graceful handling of various file system edge cases.

Key features:
- Atomic file operations using temporary files for data integrity
- Comprehensive disk space checking before operations
- Graceful interruption handling with partial result saving
- JSONL format validation and compliance checking
- File permission and path validation
- Backup creation for partial results during interruptions

Requirements satisfied:
- 3.2: Ensures each line contains a valid JSON object in JSONL format
- 3.5: Verifies all written entries are valid JSONL format
- 6.2: Handles file overwriting appropriately
- 6.3: Detects and reports storage/disk space issues
- 6.5: Saves partial results and reports status on interruption
"""

import json
import os
import tempfile
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console

from .models import DataEntry, GenerationResult
from .formats import FileFormat, FormatError, read_file, write_file, detect_format

console = Console()


class FileOperationError(Exception):
    """Raised when file operations fail."""
    pass


class DiskSpaceError(FileOperationError):
    """Raised when insufficient disk space is detected."""
    pass


def write_jsonl_file(entries: List[DataEntry], output_path: str, overwrite: bool = True) -> None:
    """
    Write data entries to a JSONL file with atomic operations.
    
    Args:
        entries: List of DataEntry objects to write
        output_path: Path to the output file
        overwrite: Whether to overwrite existing files
        
    Raises:
        FileOperationError: If file operations fail
        DiskSpaceError: If insufficient disk space
    """
    if not entries:
        raise FileOperationError("No entries provided to write")
    
    output_file = Path(output_path)
    
    # Check if file exists and handle overwrite logic
    try:
        file_exists = output_file.exists()
    except (OSError, PermissionError):
        # If we can't check if file exists due to permissions, assume it doesn't exist
        file_exists = False
        
    if file_exists and not overwrite:
        raise FileOperationError(f"Output file {output_path} already exists and overwrite is disabled")
    
    # Create parent directories if they don't exist
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise FileOperationError(f"Cannot create directory {output_file.parent}: {str(e)}")
    
    # Check disk space before writing
    estimated_size = estimate_file_size(len(entries))
    check_disk_space(output_path, estimated_size)
    
    # Use atomic write operation with temporary file
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', 
            encoding='utf-8', 
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix='.tmp',
            delete=False
        ) as temp_file:
            temp_path = temp_file.name
            
            # Write each entry as a JSON line
            for entry in entries:
                json_line = entry.model_dump_json()
                temp_file.write(json_line + '\n')
            
            temp_file.flush()
            os.fsync(temp_file.fileno())
        
        # Atomically move temporary file to final location
        shutil.move(temp_path, output_file)
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise FileOperationError(f"Failed to write JSONL file: {str(e)}")


def append_jsonl_entries(entries: List[DataEntry], output_path: str) -> None:
    """
    Append data entries to an existing JSONL file.
    
    Args:
        entries: List of DataEntry objects to append
        output_path: Path to the output file
        
    Raises:
        FileOperationError: If file operations fail
    """
    if not entries:
        return
    
    output_file = Path(output_path)
    
    try:
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Append entries to file
        with open(output_file, 'a', encoding='utf-8') as f:
            for entry in entries:
                json_line = entry.model_dump_json()
                f.write(json_line + '\n')
                
    except Exception as e:
        raise FileOperationError(f"Failed to append to JSONL file: {str(e)}")


def validate_jsonl_file(file_path: str) -> bool:
    """
    Validate that a file contains valid JSONL format.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if file is valid JSONL, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    console.print(f"[red]Invalid JSON on line {line_num}: {line[:50]}...[/red]")
                    return False
        return True
        
    except Exception as e:
        console.print(f"[red]Error validating JSONL file: {str(e)}[/red]")
        return False


def read_jsonl_file(file_path: str, return_field_names: bool = False) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], List[str]]:
    """
    Read and parse a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        return_field_names: If True, also return unique field names from all entries
        
    Returns:
        List[Dict[str, Any]]: List of parsed JSON objects (if return_field_names=False)
        Tuple[List[Dict[str, Any]], List[str]]: Tuple of (entries, field_names) (if return_field_names=True)
        
    Raises:
        FileOperationError: If file operations fail
    """
    try:
        entries = []
        field_names_set: set[str] = set()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        raise FileOperationError(
                            f"Invalid JSONL on line {line_num}: expected JSON object, got {type(entry).__name__}"
                        )
                    entries.append(entry)
                    # Collect field names from this entry
                    field_names_set.update(entry.keys())
                except json.JSONDecodeError as e:
                    raise FileOperationError(
                        f"Invalid JSON on line {line_num}: {str(e)}"
                    )
        
        if return_field_names:
            # Return sorted list of unique field names for consistent ordering
            return entries, sorted(field_names_set)
        return entries
        
    except FileNotFoundError:
        raise FileOperationError(f"File not found: {file_path}")
    except FileOperationError:
        # Re-raise FileOperationError without wrapping
        raise
    except Exception as e:
        raise FileOperationError(f"Failed to read JSONL file: {str(e)}")


def check_file_overwrite(file_path: str) -> bool:
    """
    Check if a file exists and would be overwritten.
    
    Args:
        file_path: Path to check
        
    Returns:
        bool: True if file exists and would be overwritten
    """
    return Path(file_path).exists()


def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        int: File size in bytes
        
    Raises:
        FileOperationError: If file operations fail
    """
    try:
        return Path(file_path).stat().st_size
    except FileNotFoundError:
        raise FileOperationError(f"File not found: {file_path}")
    except Exception as e:
        raise FileOperationError(f"Failed to get file size: {str(e)}")


def ensure_jsonl_extension(filename: str) -> str:
    """
    Ensure filename has .jsonl extension.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Filename with .jsonl extension
    """
    if not filename.endswith('.jsonl'):
        return f"{filename}.jsonl"
    return filename


def check_disk_space(path: str, required_bytes: int = 1024 * 1024) -> bool:
    """
    Check if sufficient disk space is available for file operations.
    
    Args:
        path: Path to check disk space for
        required_bytes: Minimum required bytes (default: 1MB)
        
    Returns:
        bool: True if sufficient space available
        
    Raises:
        DiskSpaceError: If insufficient disk space detected
    """
    try:
        # Get the directory path
        dir_path = Path(path).parent if Path(path).is_file() or not Path(path).exists() else Path(path)
        
        # Ensure directory exists
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get disk usage statistics
        stat = os.statvfs(dir_path)
        
        # Calculate available space in bytes
        available_bytes = stat.f_bavail * stat.f_frsize
        
        if available_bytes < required_bytes:
            available_mb = available_bytes / (1024 * 1024)
            required_mb = required_bytes / (1024 * 1024)
            raise DiskSpaceError(
                f"Insufficient disk space. Available: {available_mb:.1f}MB, Required: {required_mb:.1f}MB"
            )
        
        return True
        
    except DiskSpaceError:
        raise
    except Exception as e:
        # If we can't check disk space, assume it's available
        console.print(f"[yellow]⚠️  Warning: Could not check disk space: {str(e)}[/yellow]")
        return True


def estimate_file_size(entry_count: int, avg_entry_size: int = 500) -> int:
    """
    Estimate the file size for a given number of entries.
    
    Args:
        entry_count: Number of entries to be written
        avg_entry_size: Average size per entry in bytes (default: 500)
        
    Returns:
        int: Estimated file size in bytes
    """
    # Add some buffer for JSON formatting and newlines
    return entry_count * avg_entry_size + 1024


def create_partial_backup(entries: List[DataEntry], output_path: str) -> str:
    """
    Create a backup file with partial results.
    
    Args:
        entries: List of DataEntry objects to backup
        output_path: Original output path
        
    Returns:
        str: Path to the backup file
        
    Raises:
        FileOperationError: If backup creation fails
    """
    if not entries:
        return ""
    
    try:
        # Create backup filename
        output_file = Path(output_path)
        timestamp = int(time.time())
        backup_path = output_file.parent / f"{output_file.stem}_partial_{timestamp}.jsonl"
        
        # Write entries to backup file
        write_jsonl_file(entries, str(backup_path), overwrite=True)
        
        return str(backup_path)
        
    except Exception as e:
        raise FileOperationError(f"Failed to create partial backup: {str(e)}")


# Global variable to track interruption state
_interrupted = False
_partial_results = []
_output_path = ""


def _signal_handler(signum, frame):
    """Handle interruption signals (Ctrl+C, etc.)."""
    global _interrupted, _partial_results, _output_path
    
    console.print("\n[yellow]⚠️  Generation interrupted by user[/yellow]")
    
    if _partial_results and _output_path:
        try:
            backup_path = create_partial_backup(_partial_results, _output_path)
            console.print(f"[green]✅ Partial results saved to: {backup_path}[/green]")
            console.print(f"[cyan]📊 Saved {len(_partial_results)} entries before interruption[/cyan]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save partial results: {str(e)}[/red]")
    
    _interrupted = True
    sys.exit(130)  # Standard exit code for Ctrl+C


def setup_interruption_handling(partial_results: List[DataEntry], output_path: str) -> None:
    """
    Set up signal handlers for graceful interruption handling.
    
    Args:
        partial_results: List to store partial results
        output_path: Output file path for backup
    """
    global _partial_results, _output_path
    
    _partial_results = partial_results
    _output_path = output_path
    
    # Set up signal handlers for common interruption signals
    signal.signal(signal.SIGINT, _signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, _signal_handler)  # Termination signal


def is_interrupted() -> bool:
    """
    Check if generation was interrupted.
    
    Returns:
        bool: True if interrupted
    """
    return _interrupted


# ============================================================================
# Multi-Format File Operations
# ============================================================================

def read_dataset_file(file_path: str, format_hint: Optional[FileFormat] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Read dataset file in any supported format and return entries with field names.
    
    This function automatically detects the file format from the extension or uses
    the provided format hint. It supports JSONL, JSON, CSV, TSV, and Parquet formats.
    
    Args:
        file_path: Path to the input file
        format_hint: Optional format hint to override automatic detection
        
    Returns:
        Tuple of (entries, field_names)
        
    Raises:
        FileOperationError: If file cannot be read or parsed
    """
    try:
        entries, field_names = read_file(file_path, format_hint)
        
        if not entries:
            console.print(f"[yellow]⚠️  Warning: No entries found in {file_path}[/yellow]")
        
        return entries, field_names
        
    except FormatError as e:
        raise FileOperationError(f"Format error: {str(e)}")
    except Exception as e:
        raise FileOperationError(f"Failed to read dataset file: {str(e)}")


def write_dataset_file(entries: List[Dict[str, Any]], file_path: str, 
                      format_hint: Optional[FileFormat] = None, 
                      overwrite: bool = True) -> None:
    """
    Write dataset entries to file in any supported format.
    
    This function automatically detects the output format from the file extension
    or uses the provided format hint. It supports JSONL, JSON, CSV, TSV, and Parquet formats.
    
    Args:
        entries: List of data entries to write
        file_path: Output file path
        format_hint: Optional format hint to override automatic detection
        overwrite: Whether to overwrite existing files
        
    Raises:
        FileOperationError: If file cannot be written
    """
    if not entries:
        raise FileOperationError("No entries to write")
    
    # Check if file exists and handle overwrite
    output_path = Path(file_path)
    if output_path.exists() and not overwrite:
        raise FileOperationError(f"File already exists: {file_path}")
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Detect format if not provided
        if format_hint is None:
            try:
                file_format = detect_format(file_path)
            except FormatError:
                # Default to JSONL if detection fails
                file_format = FileFormat.JSONL
                console.print(f"[yellow]⚠️  Could not detect format, defaulting to JSONL[/yellow]")
        else:
            file_format = format_hint
        
        # Check disk space before writing
        estimated_size = estimate_file_size(len(entries))
        check_disk_space(file_path, estimated_size)
        
        # Write using atomic operation for data integrity
        temp_file = None
        try:
            # Create temporary file in same directory
            temp_fd, temp_file = tempfile.mkstemp(
                suffix=output_path.suffix,
                dir=output_path.parent,
                prefix=f".{output_path.name}_tmp_"
            )
            os.close(temp_fd)  # Close file descriptor, we'll use the path
            
            # Write to temporary file
            write_file(entries, temp_file, file_format)
            
            # Atomic move to final location
            shutil.move(temp_file, file_path)
            temp_file = None  # Successfully moved
            
            console.print(f"[green]✅ Successfully wrote {len(entries)} entries to {file_path}[/green]")
            
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
            raise
            
    except FormatError as e:
        raise FileOperationError(f"Format error: {str(e)}")
    except DiskSpaceError:
        raise  # Re-raise disk space errors as-is
    except Exception as e:
        raise FileOperationError(f"Failed to write dataset file: {str(e)}")


def get_supported_extensions() -> List[str]:
    """
    Get list of supported file extensions.
    
    Returns:
        List of supported file extensions (with dots)
    """
    return ['.jsonl', '.json', '.csv', '.tsv', '.parquet']


def validate_file_format(file_path: str) -> Tuple[bool, str]:
    """
    Validate if file format is supported and provide feedback.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Tuple of (is_supported, message)
    """
    try:
        file_format = detect_format(file_path)
        from .formats import get_format_description
        description = get_format_description(file_format)
        return True, f"Detected format: {description}"
    except FormatError as e:
        supported = get_supported_extensions()
        return False, f"Unsupported format. Supported extensions: {', '.join(supported)}"


def convert_file_format(input_path: str, output_path: str, 
                       input_format: Optional[FileFormat] = None,
                       output_format: Optional[FileFormat] = None) -> None:
    """
    Convert dataset file from one format to another.
    
    Args:
        input_path: Source file path
        output_path: Destination file path
        input_format: Optional input format hint
        output_format: Optional output format hint
        
    Raises:
        FileOperationError: If conversion fails
    """
    try:
        # Read from source format
        entries, field_names = read_dataset_file(input_path, input_format)
        
        if not entries:
            raise FileOperationError("No entries found in source file")
        
        # Validate compatibility with target format
        if output_format is None:
            try:
                output_format = detect_format(output_path)
            except FormatError:
                output_format = FileFormat.JSONL
        
        from .formats import validate_format_compatibility
        if not validate_format_compatibility(entries, output_format):
            console.print(f"[yellow]⚠️  Warning: Some data may be modified for {output_format.value} format[/yellow]")
        
        # Write to target format
        write_dataset_file(entries, output_path, output_format, overwrite=True)
        
        console.print(f"[green]✅ Converted {len(entries)} entries from {input_path} to {output_path}[/green]")
        
    except Exception as e:
        raise FileOperationError(f"Format conversion failed: {str(e)}")


# Backward compatibility aliases
def read_jsonl_file_with_field_names(file_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Backward compatibility wrapper for read_jsonl_file with field names.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Tuple of (entries, field_names)
    """
    return read_jsonl_file(file_path, return_field_names=True)