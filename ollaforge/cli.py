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
from typing import Optional
from pydantic import ValidationError

from .models import GenerationConfig, GenerationResult, DatasetType, OutputLanguage
from .progress import ProgressTracker

# Initialize Rich console for beautiful output
console = Console()

# Initialize Typer app
app = typer.Typer(
    name="ollaforge",
    help="Generate datasets using local Ollama models",
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
• zh-tw: 繁體中文（台灣用語）"""


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
                all_entries.extend(processed_entries)

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


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
