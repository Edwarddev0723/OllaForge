#!/usr/bin/env python3
"""
OllaForge - CLI tool for generating datasets using local Ollama models

A Python-based CLI application that leverages local Ollama models (Llama 3, Mistral, etc.) 
to automatically generate topic-specific datasets and output them in JSONL format.

This application provides:
- Rich CLI interface with progress tracking and error handling
- Robust data validation using Pydantic models
- Atomic file operations with interruption handling
- Comprehensive error recovery and user feedback
- Performance optimization for large dataset generation

Requirements satisfied:
- 1.1-1.5: CLI parameter handling with validation
- 2.1-2.5: Ollama API communication with error handling
- 3.1-3.5: Data processing and JSONL output validation
- 4.1-4.5: Rich progress tracking and user feedback
- 5.1-5.5: Modular architecture with proper separation of concerns
- 6.1-6.5: Comprehensive error handling and edge case management

Author: OllaForge Development Team
Version: 1.0.0
License: MIT
"""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
import time
import sys
import os
from pathlib import Path
from typing import Optional
from pydantic import ValidationError

from ollaforge.models import GenerationConfig, GenerationResult
from ollaforge.progress import ProgressTracker

# Initialize Rich console for beautiful output
console = Console()

# Initialize Typer app
app = typer.Typer(
    name="ollaforge",
    help="Generate datasets using local Ollama models",
    add_completion=False,
)


def validate_parameters(topic: str, count: int, model: str, output: str, raise_on_error: bool = True) -> GenerationConfig:
    """
    Validate CLI parameters using Pydantic models.
    
    Args:
        topic: Topic description for dataset generation
        count: Number of entries to generate
        model: Ollama model name
        output: Output filename
        raise_on_error: Whether to raise typer.Exit on validation errors (default: True)
        
    Returns:
        GenerationConfig: Validated configuration object
        
    Raises:
        ValidationError: If validation fails and raise_on_error is False
        typer.Exit: If validation fails and raise_on_error is True
    """
    try:
        config = GenerationConfig(
            topic=topic,
            count=count,
            model=model,
            output=output
        )
        return config
    except ValidationError as e:
        if not raise_on_error:
            raise  # Re-raise the ValidationError for testing
            
        console.print("[red]❌ Parameter validation failed:[/red]")
        for error in e.errors():
            field = error['loc'][0] if error['loc'] else 'unknown'
            message = error['msg']
            console.print(f"  • {field}: {message}")
        console.print("\n[yellow]💡 Use --help for usage information[/yellow]")
        raise typer.Exit(1)


def validate_count_range(value: int) -> int:
    """
    Validate count parameter is within acceptable range.
    
    Args:
        value: Count value to validate
        
    Returns:
        int: Validated count value
        
    Raises:
        typer.BadParameter: If count is out of range
    """
    if value < 1:
        raise typer.BadParameter("Count must be at least 1")
    if value > 10000:
        raise typer.BadParameter("Count cannot exceed 10,000 entries")
    return value


def validate_output_path(value: str) -> str:
    """
    Validate output path is writable and has proper extension.
    
    Args:
        value: Output filename to validate
        
    Returns:
        str: Validated output filename
        
    Raises:
        typer.BadParameter: If output path is invalid
    """
    if not value or not value.strip():
        raise typer.BadParameter("Output filename cannot be empty")
    
    # Check if directory exists and is writable
    output_path = Path(value)
    parent_dir = output_path.parent
    
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            raise typer.BadParameter(f"Cannot create directory: {parent_dir}")
    
    if not parent_dir.is_dir():
        raise typer.BadParameter(f"Parent path is not a directory: {parent_dir}")
    
    # Check write permissions
    if parent_dir.exists() and not os.access(parent_dir, os.W_OK):
        raise typer.BadParameter(f"No write permission for directory: {parent_dir}")
    
    return value.strip()


@app.command()
def main(
    topic: str = typer.Argument(
        ..., 
        help="Description of the dataset content to generate (e.g., 'customer service conversations')"
    ),
    count: int = typer.Option(
        10, 
        "--count", 
        "-c", 
        help="Number of data entries to generate (1-10,000)",
        callback=lambda ctx, param, value: validate_count_range(value) if value is not None else value
    ),
    model: str = typer.Option(
        "llama3", 
        "--model", 
        "-m", 
        help="Ollama model to use for generation"
    ),
    output: str = typer.Option(
        "dataset.jsonl", 
        "--output", 
        "-o", 
        help="Output filename (will be created if it doesn't exist)",
        callback=lambda ctx, param, value: validate_output_path(value) if value is not None else value
    ),
) -> None:
    """
    Generate a structured dataset using local Ollama models.
    
    OllaForge connects to your local Ollama instance to generate topic-specific
    datasets in JSONL format. Each entry contains instruction, input, and output
    fields suitable for training or fine-tuning language models.
    
    Examples:
    
        # Generate 50 customer service examples
        ollaforge "customer service conversations" --count 50
        
        # Use a specific model and output file
        ollaforge "code documentation" --model mistral --output docs.jsonl
        
        # Generate a large dataset
        ollaforge "creative writing prompts" --count 1000 --output creative.jsonl
    """
    try:
        # Validate all parameters using Pydantic
        config = validate_parameters(topic, count, model, output)
        
        # Display welcome message
        console.print(Panel.fit(
            Text("🔥 OllaForge Dataset Generator", style="bold magenta"),
            border_style="bright_blue"
        ))
        
        console.print(f"📝 Topic: {config.topic}")
        console.print(f"🔢 Count: {config.count}")
        console.print(f"🤖 Model: {config.model}")
        console.print(f"📄 Output: {config.output}")
        console.print()
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(console)
        
        # Start progress tracking
        progress_tracker.start_progress(config.count, f"Generating {config.count} entries")
        
        # Import required modules for generation
        # Performance optimization: Import modules only when needed to reduce startup time
        from ollaforge.client import generate_data, OllamaConnectionError, OllamaGenerationError
        from ollaforge.processor import process_model_response
        from ollaforge.file_manager import (
            write_jsonl_file, FileOperationError, DiskSpaceError,
            setup_interruption_handling, is_interrupted, check_disk_space, estimate_file_size
        )
        
        # Start actual generation process
        # This implements the core generation orchestration logic that integrates
        # all components: CLI, API, processing, output, and progress tracking
        # Requirements satisfied: 1.2 (count handling), 2.3 (batch processing), 3.4 (data validation)
        generated_entries = []
        start_time = time.time()
        
        # Set up interruption handling for graceful shutdown
        setup_interruption_handling(generated_entries, config.output)
        
        # Check disk space before starting generation
        try:
            estimated_size = estimate_file_size(config.count)
            check_disk_space(config.output, estimated_size)
        except DiskSpaceError as e:
            progress_tracker.stop_progress()
            console.print(f"[red]❌ {str(e)}[/red]")
            console.print("[yellow]💡 Free up disk space and try again[/yellow]")
            raise typer.Exit(1)
        
        try:
            # Generate data in optimized batches to prevent context window overflow
            # Adaptive batch sizing based on total count for better performance
            if config.count <= 10:
                batch_size = 1  # Small datasets: process individually for better quality
            elif config.count <= 100:
                batch_size = 3  # Medium datasets: small batches for balance
            else:
                batch_size = 5  # Large datasets: larger batches for efficiency
            
            remaining_count = config.count
            
            while remaining_count > 0:
                # Performance optimization: Check for interruption early to avoid unnecessary work
                if is_interrupted():
                    break
                    
                # Adaptive batch sizing: Use smaller batches for remaining entries
                # This prevents over-generation and improves memory efficiency
                current_batch_size = min(batch_size, remaining_count)
                
                try:
                    # Generate batch of raw responses from Ollama
                    raw_responses = generate_data(
                        topic=config.topic,
                        model=config.model,
                        count=current_batch_size
                    )
                    
                    # Process each response in the batch
                    for raw_response in raw_responses:
                        if 'raw_content' in raw_response:
                            # Process the raw response into a structured entry
                            processed_entry = process_model_response(raw_response['raw_content'])
                            
                            if processed_entry is not None:
                                generated_entries.append(processed_entry)
                                progress_tracker.update_progress(1, f"Generated {len(generated_entries)}/{config.count} entries")
                            else:
                                # Failed to process response
                                progress_tracker.display_error(f"Failed to process response: invalid JSON format", show_immediately=False)
                                progress_tracker.update_progress(1, f"Processed {len(generated_entries) + len(progress_tracker.errors)}/{config.count} entries")
                        else:
                            # Invalid response format
                            progress_tracker.display_error(f"Invalid response format from model", show_immediately=False)
                            progress_tracker.update_progress(1, f"Processed {len(generated_entries) + len(progress_tracker.errors)}/{config.count} entries")
                    
                    remaining_count -= current_batch_size
                    
                except OllamaConnectionError as e:
                    # Connection errors are critical - we can't continue without API access
                    progress_tracker.stop_progress()
                    console.print(f"[red]❌ Generation failed: {str(e)}[/red]")
                    console.print("[yellow]💡 Make sure Ollama is running locally on port 11434[/yellow]")
                    console.print("[yellow]💡 You can start Ollama with: ollama serve[/yellow]")
                    raise typer.Exit(1)
                    
                except OllamaGenerationError as e:
                    # Generation errors can be recovered from - continue processing
                    progress_tracker.display_error(f"Generation error: {str(e)}", show_immediately=False)
                    # Skip this batch and continue - resilient error handling
                    remaining_count -= current_batch_size
                    progress_tracker.update_progress(current_batch_size, f"Processed {len(generated_entries) + len(progress_tracker.errors)}/{config.count} entries")
                    
                except (RuntimeError, MemoryError, SystemError) as e:
                    # Critical system errors - we should exit immediately
                    progress_tracker.stop_progress()
                    console.print(f"[red]❌ Unexpected error during generation: {str(e)}[/red]")
                    console.print("[yellow]💡 Please report this issue if it persists[/yellow]")
                    raise typer.Exit(1)
                    
                except Exception as e:
                    # Handle other unexpected errors - log and continue for maximum resilience
                    # This satisfies requirement 6.4: Continue processing remaining entries on malformed responses
                    progress_tracker.display_error(f"Unexpected error: {str(e)}", show_immediately=False)
                    remaining_count -= current_batch_size
                    progress_tracker.update_progress(current_batch_size, f"Processed {len(generated_entries) + len(progress_tracker.errors)}/{config.count} entries")
            
            # Stop progress tracking
            elapsed_time = progress_tracker.stop_progress()
            
            # Write generated entries to file if we have any
            if generated_entries:
                try:
                    write_jsonl_file(generated_entries, config.output, overwrite=True)
                except DiskSpaceError as e:
                    progress_tracker.display_error(f"Disk space error: {str(e)}", show_immediately=True)
                    console.print("[yellow]💡 Free up disk space and try again[/yellow]")
                    raise typer.Exit(1)
                except FileOperationError as e:
                    progress_tracker.display_error(f"File write error: {str(e)}", show_immediately=True)
                    console.print("[yellow]💡 Check file permissions and path validity[/yellow]")
                    raise typer.Exit(1)
            else:
                if is_interrupted():
                    console.print("[yellow]⚠️  Generation was interrupted before any entries were completed[/yellow]")
                else:
                    progress_tracker.display_error("No valid entries were generated", show_immediately=True)
                    console.print("[yellow]💡 Try a different topic or check your Ollama model[/yellow]")
            
            # Create generation result with statistics
            result = GenerationResult(
                success_count=len(generated_entries),
                total_requested=config.count,
                output_file=config.output,
                duration=elapsed_time,
                errors=progress_tracker.errors
            )
            
            # Display summary
            progress_tracker.display_summary(result)
            
        except OllamaConnectionError as e:
            # This should already be handled in the batch processing loop
            # But just in case it escapes, handle it here too
            progress_tracker.stop_progress()
            console.print(f"[red]❌ Generation failed: {str(e)}[/red]")
            console.print("[yellow]💡 Make sure Ollama is running locally on port 11434[/yellow]")
            console.print("[yellow]💡 You can start Ollama with: ollama serve[/yellow]")
            raise typer.Exit(1)
            
        except OllamaGenerationError as e:
            progress_tracker.stop_progress()
            console.print(f"[red]❌ Generation failed: {str(e)}[/red]")
            console.print("[yellow]💡 Try a different model or check your Ollama setup[/yellow]")
            raise typer.Exit(1)
            
        except DiskSpaceError as e:
            progress_tracker.stop_progress()
            console.print(f"[red]❌ {str(e)}[/red]")
            console.print("[yellow]💡 Free up disk space and try again[/yellow]")
            raise typer.Exit(1)
            
        except KeyboardInterrupt:
            # This should be handled by the signal handler, but just in case
            progress_tracker.stop_progress()
            console.print("\n[yellow]⚠️  Generation interrupted by user[/yellow]")
            raise typer.Exit(130)
            
        except Exception as e:
            progress_tracker.stop_progress()
            console.print(f"[red]❌ Unexpected error during generation: {str(e)}[/red]")
            console.print("[yellow]💡 Please report this issue if it persists[/yellow]")
            raise typer.Exit(1)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        raise typer.Exit(130)
        
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
        console.print("[yellow]💡 Use --help for usage information[/yellow]")
        console.print("[yellow]💡 Please report this issue if it persists[/yellow]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()