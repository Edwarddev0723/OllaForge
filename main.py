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

from ollaforge.models import GenerationConfig, GenerationResult, DatasetType, OutputLanguage
from ollaforge.progress import ProgressTracker

# Initialize Rich console for beautiful output
console = Console()

# Initialize Typer app
app = typer.Typer(
    name="ollaforge",
    help="Generate datasets using local Ollama models",
    add_completion=False,
    invoke_without_command=True,
)


def validate_parameters(topic: str, count: int, model: str, output: str, 
                        dataset_type: DatasetType = DatasetType.SFT,
                        language: OutputLanguage = OutputLanguage.EN,
                        raise_on_error: bool = True) -> GenerationConfig:
    """
    Validate CLI parameters using Pydantic models.
    
    Args:
        topic: Topic description for dataset generation
        count: Number of entries to generate
        model: Ollama model name
        output: Output filename
        dataset_type: Type of dataset to generate
        language: Output language for generated content
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
            output=output,
            dataset_type=dataset_type,
            language=language
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


def validate_concurrency(value: int) -> int:
    """
    Validate concurrency parameter is within acceptable range.
    
    Args:
        value: Concurrency value to validate
        
    Returns:
        int: Validated concurrency value
        
    Raises:
        typer.BadParameter: If concurrency is out of range
    """
    if value < 1:
        raise typer.BadParameter("Concurrency must be at least 1")
    if value > 20:
        raise typer.BadParameter("Concurrency cannot exceed 20 (to avoid overloading Ollama)")
    return value


def validate_dataset_type(value: str) -> DatasetType:
    """
    Validate and convert dataset type string to enum.
    
    Args:
        value: Dataset type string
        
    Returns:
        DatasetType: Validated dataset type enum
        
    Raises:
        typer.BadParameter: If dataset type is invalid
    """
    try:
        return DatasetType(value.lower())
    except ValueError:
        valid_types = [t.value for t in DatasetType]
        raise typer.BadParameter(
            f"Invalid dataset type '{value}'. Valid options: {', '.join(valid_types)}"
        )


# Dataset type descriptions for help text
DATASET_TYPE_HELP = """Dataset type to generate:
• sft: Supervised Fine-tuning (Alpaca format: instruction/input/output)
• pretrain: Pre-training (raw text format)
• sft_conv: SFT Conversation (ShareGPT/ChatML multi-turn format)
• dpo: Direct Preference Optimization (prompt/chosen/rejected)"""

# Language descriptions for help text
LANGUAGE_HELP = """Output language for generated content:
• en: English (default)
• zh-tw: 繁體中文（台灣用語）"""


def validate_language(value: str) -> OutputLanguage:
    """
    Validate and convert language string to enum.
    """
    try:
        return OutputLanguage(value.lower())
    except ValueError:
        valid_langs = [l.value for l in OutputLanguage]
        raise typer.BadParameter(
            f"Invalid language '{value}'. Valid options: {', '.join(valid_langs)}"
        )


@app.command()
def generate(
    topic: str = typer.Argument(
        None, 
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
        "gpt-oss:20b", 
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
    concurrency: int = typer.Option(
        5,
        "--concurrency",
        "-j",
        help="Number of parallel requests (1-20, higher = faster but more resource intensive)",
        callback=lambda ctx, param, value: validate_concurrency(value) if value is not None else value
    ),
    dataset_type: str = typer.Option(
        "sft",
        "--type",
        "-t",
        help=DATASET_TYPE_HELP,
        callback=lambda ctx, param, value: validate_dataset_type(value) if value is not None else value
    ),
    language: str = typer.Option(
        "en",
        "--lang",
        "-l",
        help=LANGUAGE_HELP,
        callback=lambda ctx, param, value: validate_language(value) if value is not None else value
    ),
    qc_enabled: bool = typer.Option(
        True,
        "--qc/--no-qc",
        help="Enable/disable QC for Traditional Chinese (Taiwan) - filters out Mainland expressions"
    ),
    qc_confidence: float = typer.Option(
        0.9,
        "--qc-confidence",
        help="QC confidence threshold (0.0-1.0) for Taiwan Chinese classification",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Launch interactive mode with step-by-step wizard"
    ),
) -> None:
    """
    Generate a structured dataset using local Ollama models.
    
    OllaForge connects to your local Ollama instance to generate topic-specific
    datasets in JSONL format. Supports multiple dataset formats for different
    training stages (SFT, Pre-training, Conversation, DPO).
    
    Examples:
    
        # Launch interactive mode
        ollaforge -i
        
        # Generate 50 SFT examples (default)
        ollaforge "customer service conversations" --count 50
        
        # Generate pre-training data
        ollaforge "machine learning concepts" --type pretrain --count 100
        
        # Generate multi-turn conversation data
        ollaforge "technical support" --type sft_conv --output conversations.jsonl
        
        # Generate DPO preference pairs
        ollaforge "code review feedback" --type dpo --count 50 --output dpo_data.jsonl
        
        # Use a specific model
        ollaforge "code documentation" --model mistral --output docs.jsonl
    """
    try:
        # Check if interactive mode or no topic provided
        if interactive or topic is None:
            from ollaforge.interactive import main_interactive, display_generation_start
            
            result = main_interactive()
            if result is None:
                raise typer.Exit(0)
            
            config, concurrency = result
            topic = config.topic
            count = config.count
            model = config.model
            output = config.output
            dataset_type = config.dataset_type
            language = config.language
            qc_enabled = config.qc_enabled
            qc_confidence = config.qc_confidence
            
            # Show generation start
            display_generation_start(config)
        else:
            # Validate all parameters using Pydantic
            config = validate_parameters(topic, count, model, output, dataset_type, language)
            # Add QC settings to config
            config.qc_enabled = qc_enabled
            config.qc_confidence = qc_confidence
            
            # Dataset type display names
            type_display = {
                DatasetType.SFT: "SFT (Alpaca format)",
                DatasetType.PRETRAIN: "Pre-training (text)",
                DatasetType.SFT_CONVERSATION: "SFT Conversation (ShareGPT)",
                DatasetType.DPO: "DPO (Preference pairs)"
            }
            
            # Language display names
            lang_display = {
                OutputLanguage.EN: "English",
                OutputLanguage.ZH_TW: "繁體中文（台灣）"
            }
            
            # Display welcome message
            console.print(Panel.fit(
                Text("🔥 OllaForge Dataset Generator", style="bold magenta"),
                border_style="bright_blue"
            ))
            
            console.print(f"📝 Topic: {config.topic}")
            console.print(f"🔢 Count: {config.count}")
            console.print(f"🤖 Model: {config.model}")
            console.print(f"📄 Output: {config.output}")
            console.print(f"📊 Type: {type_display.get(config.dataset_type, config.dataset_type.value)}")
            console.print(f"🌐 Language: {lang_display.get(config.language, config.language.value)}")
            if config.language == OutputLanguage.ZH_TW and config.qc_enabled:
                console.print(f"🔍 QC: Enabled (confidence ≥ {config.qc_confidence:.0%})")
            console.print(f"⚡ Concurrency: {concurrency}")
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
        
        # Initialize QC if needed for Traditional Chinese
        qc_controller = None
        if config.language == OutputLanguage.ZH_TW and config.qc_enabled:
            from ollaforge.qc import QualityController
            qc_controller = QualityController(
                enabled=True,
                confidence_threshold=config.qc_confidence,
                max_retries=3
            )
            console.print("[dim]🔍 QC model loading for Taiwan Chinese validation...[/dim]")
        
        try:
            # Use batch generation: each API call generates multiple entries
            # This dramatically reduces the number of API calls needed
            # For 50 entries with batch_size=10, we only need 5 API calls instead of 50
            BATCH_SIZE = 10  # Number of entries per API call
            MAX_QC_RETRIES = 3  # Maximum retries for QC failures
            
            remaining_count = config.count
            batch_number = 0
            qc_retry_count = 0
            
            while remaining_count > 0 and len(generated_entries) < config.count:
                if is_interrupted():
                    break
                
                # Calculate how many entries to request in this batch
                entries_needed = config.count - len(generated_entries)
                current_batch_size = min(BATCH_SIZE, entries_needed)
                batch_number += 1
                
                try:
                    # Generate batch - single API call for multiple entries
                    raw_responses = generate_data(
                        topic=config.topic,
                        model=config.model,
                        count=current_batch_size,
                        concurrency=concurrency,
                        dataset_type=config.dataset_type,
                        language=config.language
                    )
                    
                    # Process responses (may contain batch JSON array)
                    for raw_response in raw_responses:
                        if 'raw_content' in raw_response:
                            is_batch = raw_response.get('is_batch', False)
                            processed_entries = process_model_response(
                                raw_response['raw_content'], 
                                is_batch=is_batch,
                                dataset_type=config.dataset_type
                            )
                            
                            for entry in processed_entries:
                                if len(generated_entries) >= config.count:
                                    break
                                
                                # Apply QC check for Traditional Chinese
                                if qc_controller is not None:
                                    entry_dict = entry.model_dump()
                                    passed, failed_fields = qc_controller.check_entry(entry_dict)
                                    
                                    if not passed:
                                        qc_retry_count += 1
                                        if qc_retry_count <= MAX_QC_RETRIES * config.count:
                                            progress_tracker.display_error(
                                                f"QC failed (Mainland Chinese detected in {', '.join(failed_fields)}), regenerating...",
                                                show_immediately=False
                                            )
                                            # Don't add this entry, will regenerate
                                            continue
                                        else:
                                            progress_tracker.display_error(
                                                f"QC retry limit reached, skipping entry",
                                                show_immediately=False
                                            )
                                            continue
                                
                                generated_entries.append(entry)
                            
                            if not processed_entries:
                                progress_tracker.display_error("Failed to parse batch response", show_immediately=False)
                        else:
                            progress_tracker.display_error("Invalid response format", show_immediately=False)
                    
                    # Update progress
                    progress_tracker.update_progress(
                        min(current_batch_size, len(generated_entries) - (batch_number - 1) * BATCH_SIZE + BATCH_SIZE),
                        f"Generated {len(generated_entries)}/{config.count} entries"
                    )
                    remaining_count = config.count - len(generated_entries)
                    
                except OllamaConnectionError as e:
                    progress_tracker.stop_progress()
                    console.print(f"[red]❌ Connection failed: {str(e)}[/red]")
                    console.print("[yellow]💡 Make sure Ollama is running: ollama serve[/yellow]")
                    raise typer.Exit(1)
                    
                except OllamaGenerationError as e:
                    progress_tracker.display_error(f"Generation error: {str(e)}", show_immediately=False)
                    remaining_count -= current_batch_size
                    
                except Exception as e:
                    progress_tracker.display_error(f"Error: {str(e)}", show_immediately=False)
                    remaining_count -= current_batch_size
            
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