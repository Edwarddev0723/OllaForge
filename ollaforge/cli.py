#!/usr/bin/env python3
"""
OllaForge CLI - Command Line Interface

This module contains the main CLI entry point, moved from main.py
for better project structure following Python packaging conventions.
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time
import sys
import os
from pathlib import Path
from typing import Optional, List
from pydantic import ValidationError

from .models import (
    GenerationConfig, GenerationResult, DatasetType, OutputLanguage,
    AugmentationConfig, AugmentationResult, FieldValidationError
)
from .formats import FileFormat
from .progress import ProgressTracker
from .file_manager import (
    FileOperationError, 
    DiskSpaceError,
    write_dataset_file,
    check_disk_space,
    estimate_file_size,
)

# Initialize Rich console for beautiful output
console = Console()

# Initialize Typer app
app = typer.Typer(
    name="ollaforge",
    help="Generate and augment datasets using local Ollama models",
    add_completion=False,
    invoke_without_command=True,
)


def validate_parameters(
    topic: str,
    count: int,
    model: str,
    output: str,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
    raise_on_error: bool = True,
) -> GenerationConfig:
    """Validate CLI parameters using Pydantic models."""
    try:
        config = GenerationConfig(
            topic=topic,
            count=count,
            model=model,
            output=output,
            dataset_type=dataset_type,
            language=language,
        )
        return config
    except ValidationError as e:
        if not raise_on_error:
            raise
        console.print("[red]❌ Parameter validation failed:[/red]")
        for error in e.errors():
            field = error["loc"][0] if error["loc"] else "unknown"
            message = error["msg"]
            console.print(f"  • {field}: {message}")
        console.print("\n[yellow]💡 Use --help for usage information[/yellow]")
        raise typer.Exit(1)


def validate_count_range(value: int) -> int:
    """Validate count parameter is within acceptable range."""
    if value < 1:
        raise typer.BadParameter("Count must be at least 1")
    if value > 10000:
        raise typer.BadParameter("Count cannot exceed 10,000 entries")
    return value


def validate_output_path(value: str) -> str:
    """Validate output path is writable and has proper extension."""
    if not value or not value.strip():
        raise typer.BadParameter("Output filename cannot be empty")

    output_path = Path(value)
    parent_dir = output_path.parent

    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            raise typer.BadParameter(f"Cannot create directory: {parent_dir}")

    if not parent_dir.is_dir():
        raise typer.BadParameter(f"Parent path is not a directory: {parent_dir}")

    if parent_dir.exists() and not os.access(parent_dir, os.W_OK):
        raise typer.BadParameter(f"No write permission for directory: {parent_dir}")

    return value.strip()


def validate_concurrency(value: int) -> int:
    """Validate concurrency parameter is within acceptable range."""
    if value < 1:
        raise typer.BadParameter("Concurrency must be at least 1")
    if value > 20:
        raise typer.BadParameter(
            "Concurrency cannot exceed 20 (to avoid overloading Ollama)"
        )
    return value


def validate_dataset_type(value: str) -> DatasetType:
    """Validate and convert dataset type string to enum."""
    try:
        return DatasetType(value.lower())
    except ValueError:
        valid_types = [t.value for t in DatasetType]
        raise typer.BadParameter(
            f"Invalid dataset type '{value}'. Valid options: {', '.join(valid_types)}"
        )


def validate_language(value: str) -> OutputLanguage:
    """Validate and convert language string to enum."""
    try:
        return OutputLanguage(value.lower())
    except ValueError:
        valid_langs = [l.value for l in OutputLanguage]
        raise typer.BadParameter(
            f"Invalid language '{value}'. Valid options: {', '.join(valid_langs)}"
        )


# Help text constants
DATASET_TYPE_HELP = """Dataset type to generate:
• sft: Supervised Fine-tuning (Alpaca format: instruction/input/output)
• pretrain: Pre-training (raw text format)
• sft_conv: SFT Conversation (ShareGPT/ChatML multi-turn format)
• dpo: Direct Preference Optimization (prompt/chosen/rejected)"""

LANGUAGE_HELP = """Output language for generated content:
• en: English (default)
• zh-tw: 繁體中文（台灣用語）
• zh-cn: 简体中文（中国大陆用语）"""


@app.command()
def generate(
    topic: str = typer.Argument(
        None,
        help="Description of the dataset content to generate",
    ),
    count: int = typer.Option(
        10,
        "--count",
        "-c",
        help="Number of data entries to generate (1-10,000)",
        callback=lambda ctx, param, value: validate_count_range(value)
        if value is not None
        else value,
    ),
    model: str = typer.Option(
        "llama3.2",
        "--model",
        "-m",
        help="Ollama model to use for generation",
    ),
    output: str = typer.Option(
        "dataset.jsonl",
        "--output",
        "-o",
        help="Output filename (will be created if it doesn't exist)",
        callback=lambda ctx, param, value: validate_output_path(value)
        if value is not None
        else value,
    ),
    concurrency: int = typer.Option(
        5,
        "--concurrency",
        "-j",
        help="Number of parallel requests (1-20)",
        callback=lambda ctx, param, value: validate_concurrency(value)
        if value is not None
        else value,
    ),
    dataset_type: str = typer.Option(
        "sft",
        "--type",
        "-t",
        help=DATASET_TYPE_HELP,
        callback=lambda ctx, param, value: validate_dataset_type(value)
        if value is not None
        else value,
    ),
    language: str = typer.Option(
        "en",
        "--lang",
        "-l",
        help=LANGUAGE_HELP,
        callback=lambda ctx, param, value: validate_language(value)
        if value is not None
        else value,
    ),
    qc_enabled: bool = typer.Option(
        True,
        "--qc/--no-qc",
        help="Enable/disable QC for Traditional Chinese (Taiwan)",
    ),
    qc_confidence: float = typer.Option(
        0.9,
        "--qc-confidence",
        help="QC confidence threshold (0.0-1.0)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Launch interactive mode with step-by-step wizard",
    ),
) -> None:
    """
    Generate a structured dataset using local Ollama models.

    Examples:

        # Launch interactive mode
        ollaforge -i

        # Generate 50 SFT examples
        ollaforge "customer service conversations" --count 50

        # Generate conversation data in Traditional Chinese
        ollaforge "咖啡點餐對話" --type sft_conv --lang zh-tw --count 100
    """
    try:
        # Check if interactive mode or no topic provided
        if interactive or topic is None:
            from .interactive import main_interactive, display_generation_start

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

            display_generation_start(config)
        else:
            config = validate_parameters(
                topic, count, model, output, dataset_type, language
            )
            config.qc_enabled = qc_enabled
            config.qc_confidence = qc_confidence

            _display_config(config, concurrency)

        # Run generation
        _run_generation(config, concurrency)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


def _display_config(config: GenerationConfig, concurrency: int) -> None:
    """Display generation configuration."""
    type_display = {
        DatasetType.SFT: "SFT (Alpaca format)",
        DatasetType.PRETRAIN: "Pre-training (text)",
        DatasetType.SFT_CONVERSATION: "SFT Conversation (ShareGPT)",
        DatasetType.DPO: "DPO (Preference pairs)",
    }
    lang_display = {
        OutputLanguage.EN: "English",
        OutputLanguage.ZH_TW: "繁體中文（台灣）",
        OutputLanguage.ZH_CN: "简体中文（中国大陆）",
    }

    console.print(
        Panel.fit(
            Text("🔥 OllaForge Dataset Generator", style="bold magenta"),
            border_style="bright_blue",
        )
    )
    console.print(f"📝 Topic: {config.topic}")
    console.print(f"🔢 Count: {config.count}")
    console.print(f"🤖 Model: {config.model}")
    console.print(f"📄 Output: {config.output}")
    console.print(
        f"📊 Type: {type_display.get(config.dataset_type, config.dataset_type.value)}"
    )
    console.print(
        f"🌐 Language: {lang_display.get(config.language, config.language.value)}"
    )
    if config.language == OutputLanguage.ZH_TW and config.qc_enabled:
        console.print(f"🔍 QC: Enabled (confidence ≥ {config.qc_confidence:.0%})")
    console.print(f"⚡ Concurrency: {concurrency}")
    console.print()


def _run_generation(config: GenerationConfig, concurrency: int) -> None:
    """Execute the generation process."""
    from .client import (
        generate_data_concurrent,
        OllamaConnectionError,
        OllamaGenerationError,
        DEFAULT_BATCH_SIZE,
    )
    from .processor import process_model_response
    from .file_manager import (
        write_jsonl_file,
        FileOperationError,
        DiskSpaceError,
        setup_interruption_handling,
        is_interrupted,
        check_disk_space,
        estimate_file_size,
    )

    progress_tracker = ProgressTracker(console)
    generated_entries = []
    start_time = time.time()

    setup_interruption_handling(generated_entries, config.output)

    # Check disk space
    try:
        estimated_size = estimate_file_size(config.count)
        check_disk_space(config.output, estimated_size)
    except DiskSpaceError as e:
        console.print(f"[red]❌ {str(e)}[/red]")
        raise typer.Exit(1)

    # Initialize QC if needed
    qc_controller = None
    request_count = config.count

    if config.language == OutputLanguage.ZH_TW and config.qc_enabled:
        from .qc import QualityController

        qc_controller = QualityController(
            enabled=True,
            confidence_threshold=config.qc_confidence,
            estimated_pass_rate=0.7,
        )
        request_count = qc_controller.calculate_request_count(config.count)
        console.print(
            "[dim]🔍 QC model loading for Taiwan Chinese validation...[/dim]"
        )
        console.print(
            f"[dim]📊 Funnel mode: requesting {request_count} entries to get {config.count} valid ones[/dim]"
        )

    # Start progress tracking
    if qc_controller is not None:
        progress_tracker.start_progress(
            request_count, f"Generating {request_count} entries (target: {config.count})"
        )
    else:
        progress_tracker.start_progress(config.count, f"Generating {config.count} entries")

    try:
        BATCH_SIZE = DEFAULT_BATCH_SIZE
        MAX_CONCURRENT = min(concurrency, 10)

        last_completed = [0]

        def on_batch_progress(completed: int, total: int):
            advance = (completed - last_completed[0]) * BATCH_SIZE
            last_completed[0] = completed
            progress_tracker.update_progress(
                advance, f"⚡ Generating batches: {completed}/{total}"
            )

        console.print(
            f"[dim]⚡ Sending {(request_count + BATCH_SIZE - 1) // BATCH_SIZE} concurrent batch requests...[/dim]"
        )

        raw_responses = generate_data_concurrent(
            topic=config.topic,
            model=config.model,
            total_count=request_count,
            batch_size=BATCH_SIZE,
            max_concurrent=MAX_CONCURRENT,
            dataset_type=config.dataset_type,
            language=config.language,
            progress_callback=on_batch_progress,
        )

        console.print(f"[dim]🔍 Processing {len(raw_responses)} batch responses...[/dim]")

        all_entries = []
        for raw_response in raw_responses:
            if is_interrupted():
                break
            if "raw_content" in raw_response:
                is_batch = raw_response.get("is_batch", False)
                processed_entries = process_model_response(
                    raw_response["raw_content"],
                    is_batch=is_batch,
                    dataset_type=config.dataset_type,
                )
                if isinstance(processed_entries, list):
                    all_entries.extend(processed_entries)
                elif processed_entries is not None:
                    all_entries.append(processed_entries)

        console.print(f"[dim]📝 Parsed {len(all_entries)} entries from responses[/dim]")

        # QC filtering
        for entry in all_entries:
            if is_interrupted():
                break
            if len(generated_entries) >= config.count:
                break

            if qc_controller is not None:
                entry_dict = entry.model_dump()
                passed, _ = qc_controller.check_entry(entry_dict)
                if not passed:
                    continue

            generated_entries.append(entry)

        if qc_controller is not None:
            qc_controller.update_pass_rate()
            qc_stats = qc_controller.get_stats()
            console.print(
                f"[dim]📊 QC stats: {qc_stats['passed']}/{qc_stats['total_checked']} passed "
                f"({qc_stats['pass_rate']:.1f}%), {qc_stats['discarded']} discarded[/dim]"
            )

    except OllamaConnectionError as e:
        progress_tracker.stop_progress()
        console.print(f"[red]❌ Connection failed: {str(e)}[/red]")
        console.print("[yellow]💡 Make sure Ollama is running: ollama serve[/yellow]")
        raise typer.Exit(1)
    except OllamaGenerationError as e:
        progress_tracker.display_error(f"Generation error: {str(e)}", show_immediately=False)

    elapsed_time = progress_tracker.stop_progress()

    # Write output
    if generated_entries:
        try:
            write_jsonl_file(generated_entries, config.output, overwrite=True)
        except (DiskSpaceError, FileOperationError) as e:
            console.print(f"[red]❌ {str(e)}[/red]")
            raise typer.Exit(1)
    else:
        if not is_interrupted():
            progress_tracker.display_error("No valid entries were generated", show_immediately=True)
            console.print("[yellow]💡 Try a different topic or check your Ollama model[/yellow]")

    result = GenerationResult(
        success_count=len(generated_entries),
        total_requested=config.count,
        output_file=config.output,
        duration=elapsed_time,
        errors=progress_tracker.errors,
    )
    progress_tracker.display_summary(result)


def validate_input_file(value: str) -> str:
    """Validate input file exists and is readable, or is a HuggingFace dataset identifier."""
    if not value or not value.strip():
        raise typer.BadParameter("Input file path cannot be empty")
    
    value = value.strip()
    
    # Check if it's a HuggingFace dataset identifier
    from .hf_loader import is_huggingface_dataset
    if is_huggingface_dataset(value):
        return value  # Return as-is for HuggingFace datasets
    
    # Otherwise, validate as a local file
    input_path = Path(value)
    if not input_path.exists():
        raise typer.BadParameter(f"Input file not found: {value}")
    if not input_path.is_file():
        raise typer.BadParameter(f"Path is not a file: {value}")
    if not os.access(input_path, os.R_OK):
        raise typer.BadParameter(f"No read permission for file: {value}")
    
    return value


def validate_field_name(value: str) -> str:
    """Validate field name is not empty."""
    if not value or not value.strip():
        raise typer.BadParameter("Field name cannot be empty")
    return value.strip()


def validate_instruction(value: str) -> str:
    """Validate instruction is not empty."""
    if not value or not value.strip():
        raise typer.BadParameter("Instruction cannot be empty")
    return value.strip()


def validate_preview_count(value: int) -> int:
    """Validate preview count is within acceptable range."""
    if value < 1:
        raise typer.BadParameter("Preview count must be at least 1")
    if value > 10:
        raise typer.BadParameter("Preview count cannot exceed 10")
    return value


# Help text for augment command
AUGMENT_FIELD_HELP = """Target field to augment or create. Must exist in the dataset unless --new-field is specified."""

AUGMENT_INSTRUCTION_HELP = """AI instruction describing how to augment the field. 
Example: "Translate to English" or "Add difficulty rating (easy/medium/hard)" """

AUGMENT_CONTEXT_HELP = """Additional fields to include as context for the AI. Can be specified multiple times.
Example: --context instruction --context input"""

FORMAT_HELP = """Input/output file format. Auto-detected from extension if not specified.
Supported formats: jsonl, json, csv, tsv, parquet"""


@app.command()
def augment(
    input_file: str = typer.Argument(
        ...,
        help="Source dataset file or HuggingFace dataset name (e.g., 'renhehuang/govQA-database-zhtw')",
        callback=lambda ctx, param, value: validate_input_file(value) if value else value,
    ),
    field: str = typer.Option(
        ...,
        "--field",
        "-f",
        help=AUGMENT_FIELD_HELP,
        callback=lambda ctx, param, value: validate_field_name(value) if value else value,
    ),
    instruction: str = typer.Option(
        ...,
        "--instruction",
        "-I",
        help=AUGMENT_INSTRUCTION_HELP,
        callback=lambda ctx, param, value: validate_instruction(value) if value else value,
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (default: input_augmented.jsonl)",
    ),
    model: str = typer.Option(
        "llama3.2",
        "--model",
        "-m",
        help="Ollama model to use for augmentation",
    ),
    concurrency: int = typer.Option(
        5,
        "--concurrency",
        "-j",
        help="Number of parallel requests (1-20)",
        callback=lambda ctx, param, value: validate_concurrency(value) if value is not None else value,
    ),
    new_field: bool = typer.Option(
        False,
        "--new-field",
        help="Create a new field instead of modifying existing",
    ),
    context: Optional[List[str]] = typer.Option(
        None,
        "--context",
        "-c",
        help=AUGMENT_CONTEXT_HELP,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Preview augmentation on sample entries before full processing",
    ),
    preview_count: int = typer.Option(
        3,
        "--preview-count",
        help="Number of entries to preview (1-10)",
        callback=lambda ctx, param, value: validate_preview_count(value) if value is not None else value,
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Interactive mode with step-by-step wizard",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-y",
        help="Overwrite output file without confirmation",
    ),
    language: str = typer.Option(
        "en",
        "--lang",
        "-l",
        help=LANGUAGE_HELP,
        callback=lambda ctx, param, value: validate_language(value) if value is not None else value,
    ),
    input_format: Optional[str] = typer.Option(
        None,
        "--input-format",
        help="Input file format (auto-detected if not specified): jsonl, json, csv, tsv, parquet",
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--output-format",
        help="Output file format (auto-detected if not specified): jsonl, json, csv, tsv, parquet",
    ),
    hf_split: str = typer.Option(
        "train",
        "--hf-split",
        help="HuggingFace dataset split to use (default: train)",
    ),
    hf_config: Optional[str] = typer.Option(
        None,
        "--hf-config",
        help="HuggingFace dataset configuration name",
    ),
    max_entries: Optional[int] = typer.Option(
        None,
        "--max-entries",
        help="Maximum number of entries to load (for large datasets)",
    ),
) -> None:
    """
    Augment an existing dataset by modifying or adding fields using AI.

    Supports local files (JSONL, JSON, CSV, TSV, Parquet) and HuggingFace datasets.

    Examples:

        # Augment a local file
        ollaforge augment data.jsonl --field output --instruction "Translate to English"

        # Augment a HuggingFace dataset
        ollaforge augment renhehuang/govQA-database-zhtw --field answer \\
            --instruction "Translate to English" --output translated.jsonl

        # Add a new 'difficulty' field using context from other fields
        ollaforge augment data.jsonl --field difficulty --new-field \\
            --instruction "Rate difficulty as easy/medium/hard" \\
            --context instruction --context input

        # Preview before full processing
        ollaforge augment data.jsonl --field output -I "Improve grammar" --preview

        # Load specific split from HuggingFace
        ollaforge augment username/dataset --hf-split test --field text -I "Summarize"
    """
    try:
        # Handle interactive mode
        if interactive:
            from .interactive import augment_interactive
            result = augment_interactive()
            if result is None:
                raise typer.Exit(0)
            # Unpack interactive result and continue with augmentation
            input_file, field, instruction, output, model, concurrency, new_field, context, preview, language = result
        
        # Check if input is a HuggingFace dataset
        from .hf_loader import is_huggingface_dataset
        is_hf_dataset = is_huggingface_dataset(input_file)
        
        # Generate default output filename if not specified
        if is_hf_dataset:
            # For HuggingFace datasets, use dataset name as base
            dataset_name = input_file.replace('/', '_')
            output_file = output or f"{dataset_name}_augmented.jsonl"
        else:
            output_file = _generate_output_filename(input_file, output)
        
        # Check for existing output file
        if not force and _check_output_exists(output_file):
            if not typer.confirm(f"Output file '{output_file}' already exists. Overwrite?"):
                console.print("[yellow]Operation cancelled.[/yellow]")
                raise typer.Exit(0)
        
        # Run augmentation
        _run_augmentation(
            input_file=input_file,
            output_file=output_file,
            field=field,
            instruction=instruction,
            model=model,
            concurrency=concurrency,
            new_field=new_field,
            context_fields=context or [],
            preview_mode=preview,
            preview_count=preview_count,
            language=language,
            force=force,
            input_format=input_format,
            output_format=output_format,
            hf_split=hf_split,
            hf_config=hf_config,
            max_entries=max_entries,
        )
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        raise typer.Exit(130)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


def _generate_output_filename(input_file: str, output: Optional[str]) -> str:
    """Generate default output filename if not specified."""
    if output:
        return output
    
    input_path = Path(input_file)
    stem = input_path.stem
    suffix = input_path.suffix or ".jsonl"
    return str(input_path.parent / f"{stem}_augmented{suffix}")


def _check_output_exists(output_file: str) -> bool:
    """Check if output file already exists."""
    return Path(output_file).exists()


def _run_augmentation(
    input_file: str,
    output_file: str,
    field: str,
    instruction: str,
    model: str,
    concurrency: int,
    new_field: bool,
    context_fields: List[str],
    preview_mode: bool,
    preview_count: int,
    language: OutputLanguage,
    force: bool,
    input_format: Optional[str] = None,
    output_format: Optional[str] = None,
    hf_split: str = "train",
    hf_config: Optional[str] = None,
    max_entries: Optional[int] = None,
) -> None:
    """Execute the augmentation process."""
    from .augmentor import DatasetAugmentor
    from .file_manager import (
        read_jsonl_file,
        write_dataset_file,
        validate_file_format,
        FileOperationError,
        DiskSpaceError,
        check_disk_space,
        estimate_file_size,
    )
    from .formats import FileFormat, FormatError
    from .hf_loader import is_huggingface_dataset, load_huggingface_dataset, HuggingFaceLoaderError
    import json
    import signal
    
    # Check if input is a HuggingFace dataset
    is_hf_dataset = is_huggingface_dataset(input_file)
    
    # Create augmentation config
    config = AugmentationConfig(
        input_file=input_file,
        output_file=output_file,
        target_field=field,
        instruction=instruction,
        model=model,
        language=language,
        create_new_field=new_field,
        context_fields=context_fields,
        preview_count=preview_count,
    )
    
    # Initialize augmentor
    augmentor = DatasetAugmentor(config)
    
    # Validate file formats (only for local files)
    if not is_hf_dataset:
        if input_format:
            try:
                input_fmt = FileFormat(input_format.lower())
            except ValueError:
                console.print(f"[red]❌ Invalid input format: {input_format}[/red]")
                console.print(f"[yellow]Supported formats: jsonl, json, csv, tsv, parquet[/yellow]")
                raise typer.Exit(1)
        else:
            input_fmt = None
        
        # Validate input file format
        is_supported, message = validate_file_format(input_file)
        if not is_supported and not input_format:
            console.print(f"[red]❌ {message}[/red]")
            raise typer.Exit(1)
    else:
        input_fmt = None
    
    if output_format:
        try:
            output_fmt = FileFormat(output_format.lower())
        except ValueError:
            console.print(f"[red]❌ Invalid output format: {output_format}[/red]")
            console.print(f"[yellow]Supported formats: jsonl, json, csv, tsv, parquet[/yellow]")
            raise typer.Exit(1)
    else:
        output_fmt = None
    
    # Load dataset and display info
    console.print(Panel.fit(
        Text("🔧 OllaForge Dataset Augmentor", style="bold magenta"),
        border_style="bright_blue",
    ))
    
    try:
        if is_hf_dataset:
            # Load from HuggingFace
            console.print(f"[cyan]🤗 Loading from HuggingFace: {input_file}[/cyan]")
            entries, field_names = load_huggingface_dataset(
                dataset_name=input_file,
                config_name=hf_config,
                split=hf_split,
                max_entries=max_entries,
            )
        else:
            # Load from local file
            entries, field_names = augmentor.load_dataset()
    except HuggingFaceLoaderError as e:
        console.print(f"[red]❌ HuggingFace error: {str(e)}[/red]")
        raise typer.Exit(1)
    except FileOperationError as e:
        console.print(f"[red]❌ {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Display dataset info (Requirement 1.4)
    if is_hf_dataset:
        console.print(f"🤗 Dataset: {input_file}")
        console.print(f"📂 Split: {hf_split}")
        if hf_config:
            console.print(f"⚙️  Config: {hf_config}")
    else:
        console.print(f"📂 Input: {input_file}")
    console.print(f"📊 Entries: {len(entries)}")
    console.print(f"📋 Fields: {', '.join(field_names)}")
    console.print(f"🎯 Target field: {field}")
    console.print(f"📝 Instruction: {instruction[:50]}{'...' if len(instruction) > 50 else ''}")
    console.print(f"🤖 Model: {model}")
    console.print(f"⚡ Concurrency: {concurrency}")
    if context_fields:
        console.print(f"📎 Context fields: {', '.join(context_fields)}")
    if new_field:
        console.print("[cyan]➕ Creating new field[/cyan]")
    console.print()
    
    # Validate target field (Requirement 2.1, 2.2)
    try:
        augmentor.validate_field(entries, field)
    except FieldValidationError as e:
        console.print(f"[red]❌ {e.message}[/red]")
        if e.available_fields:
            console.print(f"[yellow]Available fields: {', '.join(e.available_fields)}[/yellow]")
        raise typer.Exit(1)
    
    # Handle preview mode (Requirement 7.1, 7.2)
    if preview_mode:
        console.print(f"[cyan]🔍 Preview mode: processing {min(len(entries), preview_count)} sample entries...[/cyan]")
        console.print()
        
        preview_results = augmentor.preview(entries)
        
        for i, (original, augmented) in enumerate(preview_results, 1):
            console.print(f"[bold]--- Entry {i} ---[/bold]")
            original_value = original.get(field, "(not present)")
            augmented_value = augmented.get(field, "(not present)")
            
            console.print(f"[dim]Original {field}:[/dim] {original_value}")
            console.print(f"[green]Augmented {field}:[/green] {augmented_value}")
            console.print()
        
        # Ask for confirmation to proceed (Requirement 7.3, 7.4)
        if not typer.confirm("Proceed with full dataset augmentation?"):
            console.print("[yellow]Operation cancelled. You can modify the instruction and retry.[/yellow]")
            raise typer.Exit(0)
        
        console.print()
    
    # Check disk space before processing
    try:
        estimated_size = estimate_file_size(len(entries))
        check_disk_space(output_file, estimated_size)
    except DiskSpaceError as e:
        console.print(f"[red]❌ {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Set up interruption handling for partial results (Requirement 4.4)
    partial_entries: List[dict] = []
    interrupted = False
    
    def handle_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        console.print("\n[yellow]⚠️  Interruption detected, saving partial results...[/yellow]")
    
    original_handler = signal.signal(signal.SIGINT, handle_interrupt)
    
    try:
        # Execute augmentation (Requirement 3.5, 5.1)
        result = augmentor.augment_dataset(entries, concurrency=concurrency)
        augmented_entries = augmentor.get_augmented_entries()
        
        if interrupted:
            # Save partial results
            _save_partial_results(augmented_entries, output_file)
            raise typer.Exit(130)
        
        # Write output file (Requirement 4.1)
        _write_augmented_output(augmented_entries, output_file, output_fmt)
        
        # Display summary (Requirement 5.2, 5.3)
        _display_augmentation_summary(result)
        
    except Exception as e:
        if interrupted:
            # Try to save whatever we have
            augmented_entries = augmentor.get_augmented_entries()
            if augmented_entries:
                _save_partial_results(augmented_entries, output_file)
            raise typer.Exit(130)
        raise
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


def _save_partial_results(entries: List[dict], output_file: str) -> None:
    """Save partial results on interruption."""
    import json
    import time
    
    if not entries:
        console.print("[yellow]No entries to save.[/yellow]")
        return
    
    # Filter out None entries
    valid_entries = [e for e in entries if e is not None]
    
    if not valid_entries:
        console.print("[yellow]No valid entries to save.[/yellow]")
        return
    
    # Generate partial output filename
    output_path = Path(output_file)
    timestamp = int(time.time())
    partial_file = output_path.parent / f"{output_path.stem}_partial_{timestamp}.jsonl"
    
    try:
        with open(partial_file, 'w', encoding='utf-8') as f:
            for entry in valid_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        console.print(f"[green]✅ Partial results saved to: {partial_file}[/green]")
        console.print(f"[cyan]📊 Saved {len(valid_entries)} entries[/cyan]")
    except Exception as e:
        console.print(f"[red]❌ Failed to save partial results: {str(e)}[/red]")


def _write_augmented_output(entries: List[dict], output_file: str, 
                           output_format: Optional[FileFormat] = None) -> None:
    """Write augmented entries to output file in specified format."""
    # Filter out None entries
    valid_entries = [e for e in entries if e is not None]
    
    if not valid_entries:
        console.print("[yellow]⚠️  No valid entries to write[/yellow]")
        return
    
    try:
        # Use multi-format writing
        write_dataset_file(valid_entries, output_file, output_format, overwrite=True)
        
    except FileOperationError as e:
        console.print(f"[red]❌ Failed to write output: {str(e)}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error writing output: {str(e)}[/red]")
        raise typer.Exit(1)


def _display_augmentation_summary(result: AugmentationResult) -> None:
    """Display augmentation summary statistics."""
    console.print()
    console.print(Panel.fit(
        Text("📊 Augmentation Summary", style="bold green"),
        border_style="green",
    ))
    console.print(f"📁 Output: {result.output_file}")
    console.print(f"📊 Total entries: {result.total_entries}")
    console.print(f"✅ Successful: {result.success_count}")
    console.print(f"❌ Failed: {result.failure_count}")
    console.print(f"📈 Success rate: {result.success_rate:.1f}%")
    console.print(f"⏱️  Duration: {result.duration:.1f}s")
    
    if result.errors:
        console.print(f"\n[yellow]⚠️  {len(result.errors)} error(s) occurred during processing[/yellow]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Launch interactive mode with step-by-step wizard",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version information",
    ),
) -> None:
    """
    OllaForge - AI-Powered Dataset Generator & Augmentor for LLM Fine-tuning
    
    Generate new datasets or augment existing ones using local Ollama models.
    
    Examples:
    
        # Interactive mode (recommended for beginners)
        ollaforge -i
        
        # Generate dataset directly
        ollaforge generate "Python tutorials" --count 100
        
        # Augment existing dataset
        ollaforge augment data.jsonl --field output --instruction "Add more detail"
    """
    if version:
        from . import __version__
        console.print(f"OllaForge version {__version__}")
        raise typer.Exit(0)
    
    if interactive:
        from .interactive import main_interactive_router
        try:
            main_interactive_router()
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
            raise typer.Exit(1)
        return
    
    if ctx.invoked_subcommand is None:
        # Show help if no subcommand is provided
        console.print(app.get_help(ctx))
        console.print("\n[yellow]💡 Use 'ollaforge -i' for interactive mode or 'ollaforge --help' for more options[/yellow]")


def validate_chunk_size(value: int) -> int:
    """Validate chunk size parameter is within acceptable range."""
    if value < 500:
        raise typer.BadParameter("Chunk size must be at least 500 characters")
    if value > 10000:
        raise typer.BadParameter("Chunk size cannot exceed 10,000 characters")
    return value


def validate_chunk_overlap(value: int) -> int:
    """Validate chunk overlap parameter is within acceptable range."""
    if value < 0:
        raise typer.BadParameter("Chunk overlap cannot be negative")
    if value > 1000:
        raise typer.BadParameter("Chunk overlap cannot exceed 1,000 characters")
    return value


def validate_entries_per_chunk(value: int) -> int:
    """Validate entries per chunk parameter is within acceptable range."""
    if value < 1:
        raise typer.BadParameter("Entries per chunk must be at least 1")
    if value > 10:
        raise typer.BadParameter("Entries per chunk cannot exceed 10")
    return value


def validate_source_path(value: str) -> str:
    """Validate source path exists and is readable."""
    if not value or not value.strip():
        raise typer.BadParameter("Source path cannot be empty")
    
    source_path = Path(value.strip())
    if not source_path.exists():
        raise typer.BadParameter(f"Source path not found: {value}")
    if not os.access(source_path, os.R_OK):
        raise typer.BadParameter(f"No read permission for: {value}")
    
    return value.strip()


# Help text for doc2dataset command
DOC2DATASET_TYPE_HELP = """Dataset type to generate:
• sft: Supervised Fine-tuning (Alpaca format: instruction/input/output)
• pretrain: Pre-training (raw text format)
• sft_conv: SFT Conversation (ShareGPT/ChatML multi-turn format)
• dpo: Direct Preference Optimization (prompt/chosen/rejected)"""


@app.command()
def doc2dataset(
    source: str = typer.Argument(
        ...,
        help="Source document or directory path",
        callback=lambda ctx, param, value: validate_source_path(value) if value else value,
    ),
    output: str = typer.Option(
        "dataset.jsonl",
        "--output", "-o",
        help="Output dataset file path",
        callback=lambda ctx, param, value: validate_output_path(value) if value else value,
    ),
    dataset_type: str = typer.Option(
        "sft",
        "--type", "-t",
        help=DOC2DATASET_TYPE_HELP,
        callback=lambda ctx, param, value: validate_dataset_type(value) if value else value,
    ),
    model: str = typer.Option(
        "llama3.2",
        "--model", "-m",
        help="Ollama model to use for generation",
    ),
    chunk_size: int = typer.Option(
        2000,
        "--chunk-size",
        help="Chunk size in characters (500-10000)",
        callback=lambda ctx, param, value: validate_chunk_size(value) if value is not None else value,
    ),
    chunk_overlap: int = typer.Option(
        200,
        "--chunk-overlap",
        help="Overlap between chunks in characters (0-1000)",
        callback=lambda ctx, param, value: validate_chunk_overlap(value) if value is not None else value,
    ),
    count: int = typer.Option(
        3,
        "--count", "-c",
        help="Number of entries to generate per chunk (1-10)",
        callback=lambda ctx, param, value: validate_entries_per_chunk(value) if value is not None else value,
    ),
    language: str = typer.Option(
        "en",
        "--lang", "-l",
        help=LANGUAGE_HELP,
        callback=lambda ctx, param, value: validate_language(value) if value is not None else value,
    ),
    pattern: Optional[str] = typer.Option(
        None,
        "--pattern", "-p",
        help="File pattern for directory processing (e.g., '*.md')",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Recursively process directories",
    ),
    qc_enabled: bool = typer.Option(
        True,
        "--qc/--no-qc",
        help="Enable quality control filtering",
    ),
    qc_confidence: float = typer.Option(
        0.9,
        "--qc-confidence",
        help="QC confidence threshold (0.0-1.0)",
    ),
) -> None:
    """
    Convert documents to fine-tuning datasets.
    
    Supports Markdown, PDF, HTML, TXT, JSON, and code files.
    Documents are split into chunks and processed by Ollama to generate
    training data in the specified format.
    
    Examples:
    
        # Convert a single Markdown file to SFT format
        ollaforge doc2dataset README.md --type sft
        
        # Convert all Python files in a directory
        ollaforge doc2dataset ./src --pattern "*.py" --type pretrain
        
        # Generate conversation data from documentation
        ollaforge doc2dataset docs/ --type sft_conv --lang zh-tw
        
        # Convert PDF with custom chunk settings
        ollaforge doc2dataset manual.pdf --chunk-size 3000 --chunk-overlap 300
    """
    try:
        # Validate chunk overlap is less than chunk size
        if chunk_overlap >= chunk_size:
            console.print("[red]❌ Chunk overlap must be less than chunk size[/red]")
            raise typer.Exit(1)
        
        # Run the document to dataset conversion
        _run_doc2dataset(
            source=source,
            output=output,
            dataset_type=dataset_type,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            entries_per_chunk=count,
            language=language,
            pattern=pattern,
            recursive=recursive,
            qc_enabled=qc_enabled,
            qc_confidence=qc_confidence,
        )
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        raise typer.Exit(130)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


class Doc2DatasetInterruptHandler:
    """
    Handler for SIGINT interruption during doc2dataset processing.
    
    This class manages the interruption state and partial results saving
    for the doc2dataset command. It captures SIGINT signals and ensures
    that any processed entries are saved before the program exits.
    
    Requirements satisfied:
    - 5.5: Save partial results when generation is interrupted
    """
    
    def __init__(self, output_file: str):
        """
        Initialize the interrupt handler.
        
        Args:
            output_file: Path to the output file for saving partial results
        """
        self._interrupted = False
        self._output_file = output_file
        self._entries: List = []
        self._original_handler = None
    
    def setup(self) -> None:
        """Set up the signal handler for SIGINT."""
        import signal
        self._original_handler = signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def cleanup(self) -> None:
        """Restore the original signal handler."""
        import signal
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
            self._original_handler = None
    
    def _handle_interrupt(self, signum, frame) -> None:
        """Handle SIGINT signal."""
        self._interrupted = True
        console.print("\n[yellow]⚠️  Interruption detected, saving partial results...[/yellow]")
    
    @property
    def interrupted(self) -> bool:
        """Check if processing was interrupted."""
        return self._interrupted
    
    def set_entries(self, entries: List) -> None:
        """
        Set the current entries list for potential partial save.
        
        Args:
            entries: List of entries to save if interrupted
        """
        self._entries = entries
    
    def add_entries(self, new_entries: List) -> None:
        """
        Add entries to the current collection.
        
        Args:
            new_entries: New entries to add
        """
        if new_entries:
            self._entries.extend(new_entries)
    
    def get_entries(self) -> List:
        """Get the current entries list."""
        return self._entries
    
    def save_partial_results(self) -> Optional[str]:
        """
        Save partial results to a timestamped file.
        
        Returns:
            Path to the partial results file, or None if no entries to save
            
        Requirements satisfied:
        - 5.5: Save partial results when generation is interrupted
        """
        import json
        import time
        
        if not self._entries:
            console.print("[yellow]No entries to save.[/yellow]")
            return None
        
        output_path = Path(self._output_file)
        timestamp = int(time.time())
        partial_file = output_path.parent / f"{output_path.stem}_partial_{timestamp}.jsonl"
        
        # Ensure parent directory exists
        partial_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            saved_count = 0
            with open(partial_file, 'w', encoding='utf-8') as f:
                for entry in self._entries:
                    if entry is None:
                        continue
                    
                    # Convert entry to dict if needed
                    if hasattr(entry, 'model_dump'):
                        entry_dict = entry.model_dump()
                    elif hasattr(entry, 'dict'):
                        entry_dict = entry.dict()
                    elif isinstance(entry, dict):
                        entry_dict = entry
                    else:
                        continue
                    
                    f.write(json.dumps(entry_dict, ensure_ascii=False) + '\n')
                    saved_count += 1
            
            console.print(f"[green]✅ Partial results saved to: {partial_file}[/green]")
            console.print(f"[cyan]📊 Saved {saved_count} entries[/cyan]")
            return str(partial_file)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to save partial results: {str(e)}[/red]")
            return None


def _run_doc2dataset(
    source: str,
    output: str,
    dataset_type: DatasetType,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    entries_per_chunk: int,
    language: OutputLanguage,
    pattern: Optional[str],
    recursive: bool,
    qc_enabled: bool,
    qc_confidence: float,
) -> None:
    """Execute the document to dataset conversion process."""
    import time
    import signal
    from .doc_parser import (
        DocumentParserFactory,
        UnsupportedFormatError,
        ParsedDocument,
    )
    from .chunk_splitter import ChunkSplitter, ChunkConfig, SplitStrategy
    from .doc_generator import DocumentDatasetGenerator, DocGenerationConfig
    from .models import DocToDatasetConfig, DocProcessingResult, BatchProcessingResult
    from .file_manager import (
        write_jsonl_file,
        FileOperationError,
        DiskSpaceError,
        check_disk_space,
        estimate_file_size,
    )
    from .client import OllamaConnectionError
    from .batch_processor import BatchProcessor, BatchConfig, aggregate_results
    
    start_time = time.time()
    source_path = Path(source)
    
    # Initialize interrupt handler for graceful shutdown
    interrupt_handler = Doc2DatasetInterruptHandler(output)
    
    # Display configuration
    _display_doc2dataset_config(
        source=source,
        output=output,
        dataset_type=dataset_type,
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        entries_per_chunk=entries_per_chunk,
        language=language,
        pattern=pattern,
        recursive=recursive,
        qc_enabled=qc_enabled,
    )
    
    # Initialize batch processor
    batch_config = BatchConfig(
        recursive=recursive,
        file_pattern=pattern,
        continue_on_error=True
    )
    batch_processor = BatchProcessor(batch_config)
    
    # Collect files to process
    try:
        files_to_process = batch_processor.collect_files(source_path)
    except (FileNotFoundError, UnsupportedFormatError) as e:
        console.print(f"[red]❌ {str(e)}[/red]")
        raise typer.Exit(1)
    
    if not files_to_process:
        console.print("[yellow]⚠️  No supported files found to process[/yellow]")
        console.print(f"[dim]Supported formats: {', '.join(DocumentParserFactory.get_supported_formats())}[/dim]")
        raise typer.Exit(0)
    
    console.print(f"[dim]📁 Found {len(files_to_process)} file(s) to process[/dim]")
    console.print()
    
    # Initialize components
    chunk_config = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=SplitStrategy.HYBRID,
    )
    splitter = ChunkSplitter(chunk_config)
    
    gen_config = DocGenerationConfig(
        dataset_type=dataset_type,
        model=model,
        language=language,
        entries_per_chunk=entries_per_chunk,
        qc_enabled=qc_enabled,
        qc_confidence=qc_confidence,
    )
    generator = DocumentDatasetGenerator(gen_config)
    
    # Set up progress tracking
    progress_tracker = ProgressTracker(console)
    
    # Set up interruption handling using the dedicated handler
    interrupt_handler.setup()
    
    try:
        # Check disk space
        try:
            estimated_size = estimate_file_size(len(files_to_process) * entries_per_chunk * 10)
            check_disk_space(output, estimated_size)
        except DiskSpaceError as e:
            console.print(f"[red]❌ {str(e)}[/red]")
            raise typer.Exit(1)
        
        # Process each file
        total_files = len(files_to_process)
        total_chunks = 0
        
        for file_idx, file_path in enumerate(files_to_process):
            if interrupt_handler.interrupted:
                break
            
            file_errors: List[str] = []
            file_entries_count = 0
            file_chunks_count = 0
            
            console.print(f"[cyan]📄 Processing ({file_idx + 1}/{total_files}): {file_path.name}[/cyan]")
            
            try:
                # Parse document
                parser = DocumentParserFactory.get_parser(str(file_path))
                document = parser.parse(str(file_path))
                
                # Split into chunks
                chunks = splitter.split(document)
                file_chunks_count = len(chunks)
                total_chunks += file_chunks_count
                
                if not chunks:
                    console.print(f"[dim]  ⚠️  No content to process in {file_path.name}[/dim]")
                    batch_processor.add_file_result(DocProcessingResult(
                        source_file=str(file_path),
                        chunks_processed=0,
                        entries_generated=0,
                        errors=["No content to process"]
                    ))
                    continue
                
                console.print(f"[dim]  📦 Split into {file_chunks_count} chunk(s)[/dim]")
                
                # Generate entries from chunks with progress
                progress_tracker.start_progress(
                    file_chunks_count,
                    f"Generating from {file_path.name}"
                )
                
                def on_chunk_progress(completed: int, total: int):
                    progress_tracker.update_progress(1, f"Chunk {completed}/{total}")
                
                try:
                    entries = generator.generate_from_chunks(chunks, on_chunk_progress)
                    file_entries_count = len(entries)
                    batch_processor.add_entries(entries)
                    # Update interrupt handler with current entries for potential partial save
                    interrupt_handler.set_entries(batch_processor.get_entries())
                    
                except OllamaConnectionError as e:
                    progress_tracker.stop_progress()
                    console.print(f"[red]❌ Ollama connection failed: {str(e)}[/red]")
                    console.print("[yellow]💡 Make sure Ollama is running: ollama serve[/yellow]")
                    console.print(f"[yellow]💡 Ensure model '{model}' is available: ollama pull {model}[/yellow]")
                    raise typer.Exit(1)
                
                progress_tracker.stop_progress()
                console.print(f"[green]  ✅ Generated {file_entries_count} entries[/green]")
                
            except UnsupportedFormatError as e:
                file_errors.append(str(e))
                console.print(f"[yellow]  ⚠️  {str(e)}[/yellow]")
            except FileNotFoundError as e:
                file_errors.append(f"File not found: {file_path}")
                console.print(f"[red]  ❌ File not found: {file_path}[/red]")
            except PermissionError as e:
                file_errors.append(f"Permission denied: {file_path}")
                console.print(f"[red]  ❌ Permission denied: {file_path}[/red]")
            except Exception as e:
                file_errors.append(f"Error processing {file_path}: {str(e)}")
                console.print(f"[red]  ❌ Error: {str(e)}[/red]")
            
            batch_processor.add_file_result(DocProcessingResult(
                source_file=str(file_path),
                chunks_processed=file_chunks_count,
                entries_generated=file_entries_count,
                errors=file_errors
            ))
        
        # Get all collected entries
        all_entries = batch_processor.get_entries()
        file_results = batch_processor.get_file_results()
        
        # Write output
        if all_entries:
            try:
                # Convert entries to dicts for writing
                entries_dicts = []
                for entry in all_entries:
                    if hasattr(entry, 'model_dump'):
                        entries_dicts.append(entry.model_dump())
                    elif hasattr(entry, 'dict'):
                        entries_dicts.append(entry.dict())
                    elif isinstance(entry, dict):
                        entries_dicts.append(entry)
                
                write_jsonl_file(entries_dicts, output, overwrite=True)
                console.print(f"\n[green]✅ Output written to: {output}[/green]")
            except (DiskSpaceError, FileOperationError) as e:
                console.print(f"[red]❌ Failed to write output: {str(e)}[/red]")
                raise typer.Exit(1)
        else:
            if not interrupt_handler.interrupted:
                console.print("\n[yellow]⚠️  No entries were generated[/yellow]")
                console.print("[yellow]💡 Check if the documents contain processable content[/yellow]")
        
        # Calculate results
        elapsed_time = time.time() - start_time
        successful_files = sum(1 for r in file_results if not r.errors)
        failed_files = sum(1 for r in file_results if r.errors)
        
        # Display summary
        _display_doc2dataset_summary(
            total_files=total_files,
            successful_files=successful_files,
            failed_files=failed_files,
            total_entries=len(all_entries),
            output_file=output,
            duration=elapsed_time,
            interrupted=interrupt_handler.interrupted,
        )
        
    finally:
        # Restore original signal handler
        interrupt_handler.cleanup()
        
        # Save partial results if interrupted
        if interrupt_handler.interrupted:
            interrupt_handler.save_partial_results()


def _collect_files_from_directory(
    directory: Path,
    pattern: Optional[str],
    recursive: bool
) -> List[Path]:
    """
    Collect supported files from a directory.
    
    This function delegates to the batch_processor module for consistent
    file collection behavior across the application.
    
    Args:
        directory: Path to the directory to search
        pattern: Optional glob pattern to filter files
        recursive: Whether to search subdirectories recursively
        
    Returns:
        List of Path objects for all matching supported files
        
    Requirements satisfied:
    - 6.1: Accept directory path to process all supported files
    - 6.2: Recursively find all supported document files
    - 6.3: Filter files by glob pattern
    """
    from .batch_processor import collect_supported_files
    
    return collect_supported_files(directory, pattern, recursive)


def _display_doc2dataset_config(
    source: str,
    output: str,
    dataset_type: DatasetType,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    entries_per_chunk: int,
    language: OutputLanguage,
    pattern: Optional[str],
    recursive: bool,
    qc_enabled: bool,
) -> None:
    """Display document to dataset configuration."""
    type_display = {
        DatasetType.SFT: "SFT (Alpaca format)",
        DatasetType.PRETRAIN: "Pre-training (text)",
        DatasetType.SFT_CONVERSATION: "SFT Conversation (ShareGPT)",
        DatasetType.DPO: "DPO (Preference pairs)",
    }
    lang_display = {
        OutputLanguage.EN: "English",
        OutputLanguage.ZH_TW: "繁體中文（台灣）",
        OutputLanguage.ZH_CN: "简体中文（中国大陆）",
    }
    
    console.print(
        Panel.fit(
            Text("📚 OllaForge Document to Dataset", style="bold magenta"),
            border_style="bright_blue",
        )
    )
    console.print(f"📂 Source: {source}")
    console.print(f"📄 Output: {output}")
    console.print(f"📊 Type: {type_display.get(dataset_type, dataset_type.value)}")
    console.print(f"🤖 Model: {model}")
    console.print(f"📏 Chunk size: {chunk_size} chars")
    console.print(f"🔗 Chunk overlap: {chunk_overlap} chars")
    console.print(f"🔢 Entries per chunk: {entries_per_chunk}")
    console.print(f"🌐 Language: {lang_display.get(language, language.value)}")
    if pattern:
        console.print(f"🔍 Pattern: {pattern}")
    console.print(f"📁 Recursive: {'Yes' if recursive else 'No'}")
    if language == OutputLanguage.ZH_TW and qc_enabled:
        console.print("[dim]🔍 QC: Enabled for Taiwan Chinese[/dim]")
    console.print()


def _display_doc2dataset_summary(
    total_files: int,
    successful_files: int,
    failed_files: int,
    total_entries: int,
    output_file: str,
    duration: float,
    interrupted: bool,
) -> None:
    """Display document to dataset processing summary."""
    console.print()
    
    if interrupted:
        console.print(Panel.fit(
            Text("⚠️  Processing Interrupted", style="bold yellow"),
            border_style="yellow",
        ))
    else:
        console.print(Panel.fit(
            Text("📊 Processing Summary", style="bold green"),
            border_style="green",
        ))
    
    console.print(f"📁 Files processed: {successful_files}/{total_files}")
    if failed_files > 0:
        console.print(f"[yellow]❌ Failed files: {failed_files}[/yellow]")
    console.print(f"📝 Total entries: {total_entries}")
    console.print(f"📄 Output: {output_file}")
    console.print(f"⏱️  Duration: {duration:.1f}s")
    
    if total_files > 0:
        success_rate = (successful_files / total_files) * 100
        console.print(f"📈 Success rate: {success_rate:.1f}%")


def _save_partial_doc2dataset_results(entries: List, output_file: str) -> None:
    """Save partial results when processing is interrupted."""
    import json
    import time
    
    if not entries:
        console.print("[yellow]No entries to save.[/yellow]")
        return
    
    output_path = Path(output_file)
    timestamp = int(time.time())
    partial_file = output_path.parent / f"{output_path.stem}_partial_{timestamp}.jsonl"
    
    try:
        with open(partial_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                if hasattr(entry, 'model_dump'):
                    entry_dict = entry.model_dump()
                elif hasattr(entry, 'dict'):
                    entry_dict = entry.dict()
                elif isinstance(entry, dict):
                    entry_dict = entry
                else:
                    continue
                f.write(json.dumps(entry_dict, ensure_ascii=False) + '\n')
        
        console.print(f"[green]✅ Partial results saved to: {partial_file}[/green]")
        console.print(f"[cyan]📊 Saved {len(entries)} entries[/cyan]")
    except Exception as e:
        console.print(f"[red]❌ Failed to save partial results: {str(e)}[/red]")


if __name__ == "__main__":
    app()

@app.command()
def convert(
    input_file: str = typer.Argument(
        ...,
        help="Source dataset file to convert",
    ),
    output_file: str = typer.Argument(
        ...,
        help="Output file path",
    ),
    input_format: Optional[str] = typer.Option(
        None,
        "--input-format",
        help="Input file format (auto-detected if not specified): jsonl, json, csv, tsv, parquet",
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--output-format", 
        help="Output file format (auto-detected if not specified): jsonl, json, csv, tsv, parquet",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-y",
        help="Overwrite output file without confirmation",
    ),
) -> None:
    """
    Convert dataset files between different formats.
    
    Supports conversion between JSONL, JSON, CSV, TSV, and Parquet formats.
    Format is automatically detected from file extensions if not specified.
    
    Examples:
    
        # Convert JSONL to CSV
        ollaforge convert data.jsonl data.csv
        
        # Convert CSV to JSON with explicit formats
        ollaforge convert data.csv data.json --input-format csv --output-format json
        
        # Convert to Parquet (requires pandas)
        ollaforge convert data.jsonl data.parquet
    """
    try:
        from .file_manager import convert_file_format, validate_file_format
        from .formats import FileFormat
        
        # Validate input file
        if not Path(input_file).exists():
            console.print(f"[red]❌ Input file not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        # Validate formats if specified
        input_fmt = None
        if input_format:
            try:
                input_fmt = FileFormat(input_format.lower())
            except ValueError:
                console.print(f"[red]❌ Invalid input format: {input_format}[/red]")
                console.print(f"[yellow]Supported formats: jsonl, json, csv, tsv, parquet[/yellow]")
                raise typer.Exit(1)
        
        output_fmt = None
        if output_format:
            try:
                output_fmt = FileFormat(output_format.lower())
            except ValueError:
                console.print(f"[red]❌ Invalid output format: {output_format}[/red]")
                console.print(f"[yellow]Supported formats: jsonl, json, csv, tsv, parquet[/yellow]")
                raise typer.Exit(1)
        
        # Check if output file exists
        if Path(output_file).exists() and not force:
            if not typer.confirm(f"Output file '{output_file}' already exists. Overwrite?"):
                console.print("[yellow]Operation cancelled.[/yellow]")
                raise typer.Exit(0)
        
        # Display conversion info
        console.print(Panel.fit(
            Text("🔄 OllaForge Format Converter", style="bold cyan"),
            border_style="bright_blue",
        ))
        
        console.print(f"📂 Input: {input_file}")
        console.print(f"📁 Output: {output_file}")
        
        # Validate input format
        is_supported, message = validate_file_format(input_file)
        if is_supported:
            console.print(f"[dim]📄 {message}[/dim]")
        elif not input_format:
            console.print(f"[red]❌ {message}[/red]")
            raise typer.Exit(1)
        
        console.print()
        
        # Perform conversion
        convert_file_format(input_file, output_file, input_fmt, output_fmt)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]❌ Conversion failed: {str(e)}[/red]")
        raise typer.Exit(1)