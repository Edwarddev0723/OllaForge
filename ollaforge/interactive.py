"""
Interactive CLI interface for OllaForge.

This module provides a beautiful, user-friendly interactive interface using
Rich for rendering and questionary for prompts.
"""

import sys
from typing import Optional, List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich.columns import Columns
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.style import Style
from rich.box import ROUNDED, DOUBLE, HEAVY
from rich.live import Live
from rich.layout import Layout
from rich.markdown import Markdown

from .models import DatasetType, GenerationConfig, OutputLanguage


console = Console()


# ============================================================================
# ASCII Art & Branding
# ============================================================================

LOGO = """
   ____  _ _       _____                    
  / __ \\| | |     |  ___|                   
 | |  | | | | __ _| |_ ___  _ __ __ _  ___  
 | |  | | | |/ _` |  _/ _ \\| '__/ _` |/ _ \\ 
 | |__| | | | (_| | || (_) | | | (_| |  __/ 
  \\____/|_|_|\\__,_\\_| \\___/|_|  \\__, |\\___| 
                                 __/ |      
                                |___/       
"""

TAGLINE = "🔥 AI-Powered Dataset Generator for LLM Training"


def display_banner() -> None:
    """Display the application banner with logo and tagline."""
    logo_text = Text(LOGO, style="bold cyan")
    tagline_text = Text(TAGLINE, style="italic bright_magenta")
    
    banner_content = Text()
    banner_content.append(LOGO, style="bold cyan")
    banner_content.append("\n")
    banner_content.append(TAGLINE, style="italic bright_magenta")
    
    panel = Panel(
        Align.center(banner_content),
        border_style="bright_blue",
        padding=(0, 2),
    )
    console.print(panel)
    console.print()


# ============================================================================
# Dataset Type Selection Menu
# ============================================================================

DATASET_TYPE_INFO = {
    DatasetType.SFT: {
        "name": "📝 SFT (Supervised Fine-tuning)",
        "description": "Alpaca format with instruction/input/output",
        "format": '{"instruction": "...", "input": "...", "output": "..."}',
        "use_case": "General instruction following, task completion",
        "color": "green",
    },
    DatasetType.PRETRAIN: {
        "name": "📚 Pre-training",
        "description": "Raw text for continued pre-training",
        "format": '{"text": "..."}',
        "use_case": "Domain adaptation, knowledge injection",
        "color": "blue",
    },
    DatasetType.SFT_CONVERSATION: {
        "name": "💬 SFT Conversation",
        "description": "Multi-turn dialogue (ShareGPT/ChatML)",
        "format": '{"conversations": [{"role": "...", "content": "..."}]}',
        "use_case": "Chatbot training, dialogue systems",
        "color": "yellow",
    },
    DatasetType.DPO: {
        "name": "⚖️  DPO (Direct Preference Optimization)",
        "description": "Preference pairs with chosen/rejected",
        "format": '{"prompt": "...", "chosen": "...", "rejected": "..."}',
        "use_case": "RLHF, preference alignment",
        "color": "magenta",
    },
}


def display_dataset_type_menu() -> DatasetType:
    """Display an interactive menu for selecting dataset type."""
    console.print("[bold cyan]📊 Select Dataset Type[/bold cyan]\n")
    
    # Create a table showing all options
    table = Table(
        show_header=True,
        header_style="bold white on blue",
        box=ROUNDED,
        border_style="bright_blue",
        padding=(0, 1),
    )
    
    table.add_column("#", style="bold cyan", justify="center", width=3)
    table.add_column("Type", style="bold", width=35)
    table.add_column("Use Case", style="dim", width=35)
    
    type_list = list(DatasetType)
    for i, dtype in enumerate(type_list, 1):
        info = DATASET_TYPE_INFO[dtype]
        table.add_row(
            str(i),
            f"[{info['color']}]{info['name']}[/{info['color']}]",
            info["use_case"],
        )
    
    console.print(table)
    console.print()
    
    # Get user selection
    while True:
        choice = Prompt.ask(
            "[bold]Enter your choice[/bold]",
            choices=[str(i) for i in range(1, len(type_list) + 1)],
            default="1",
        )
        selected_type = type_list[int(choice) - 1]
        
        # Show selected type details
        info = DATASET_TYPE_INFO[selected_type]
        detail_panel = Panel(
            f"[bold]{info['name']}[/bold]\n\n"
            f"[dim]Description:[/dim] {info['description']}\n"
            f"[dim]Format:[/dim] [cyan]{info['format']}[/cyan]",
            title="[bold green]✓ Selected[/bold green]",
            border_style=info["color"],
            padding=(0, 2),
        )
        console.print(detail_panel)
        
        if Confirm.ask("\n[bold]Confirm this selection?[/bold]", default=True):
            return selected_type
        
        console.print()


def display_model_selection() -> str:
    """Display model selection with suggestions."""
    console.print("[bold cyan]🤖 Select Ollama Model[/bold cyan]\n")
    
    # Show popular models
    popular_models = [
        ("gpt-oss:20b", "GPT-OSS 20B - Default model, great performance"),
        ("deepseek-r1:14b", "DeepSeek-R1 - Open reasoning model, O3/Gemini 2.5 Pro level"),
        ("qwen3:14b", "Qwen3 - Latest gen with dense and MoE models"),
        ("ministral-3:14b", "Ministral 3 - Designed for edge deployment"),
        ("gemma3:12b", "Gemma3 - Most capable model on single GPU"),
    ]
    
    table = Table(
        show_header=True,
        header_style="bold white on dark_green",
        box=ROUNDED,
        border_style="green",
    )
    
    table.add_column("#", style="bold cyan", justify="center", width=3)
    table.add_column("Model", style="bold green", width=18)
    table.add_column("Description", style="dim", width=50)
    
    for i, (model, desc) in enumerate(popular_models, 1):
        table.add_row(str(i), model, desc)
    
    table.add_row("C", "[yellow]Custom...[/yellow]", "Enter a custom model name")
    
    console.print(table)
    console.print()
    
    choice = Prompt.ask(
        "[bold]Enter choice (1-5) or 'C' for custom[/bold]",
        default="1",
    )
    
    if choice.upper() == "C":
        return Prompt.ask("[bold]Enter custom model name[/bold]")
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(popular_models):
            return popular_models[idx][0]
    except ValueError:
        pass
    
    return "gpt-oss:20b"  # Default fallback


# ============================================================================
# Language Selection
# ============================================================================

LANGUAGE_INFO = {
    OutputLanguage.EN: {
        "name": "🇺🇸 English",
        "description": "Generate content in English",
        "color": "blue",
    },
    OutputLanguage.ZH_TW: {
        "name": "🇹🇼 繁體中文（台灣）",
        "description": "使用繁體中文（台灣用語）生成內容",
        "color": "red",
    },
}


def display_language_selection() -> OutputLanguage:
    """Display language selection menu."""
    console.print("[bold cyan]🌐 Select Output Language[/bold cyan]\n")
    
    table = Table(
        show_header=True,
        header_style="bold white on dark_blue",
        box=ROUNDED,
        border_style="blue",
    )
    
    table.add_column("#", style="bold cyan", justify="center", width=3)
    table.add_column("Language", style="bold", width=25)
    table.add_column("Description", style="dim", width=40)
    
    lang_list = list(OutputLanguage)
    for i, lang in enumerate(lang_list, 1):
        info = LANGUAGE_INFO[lang]
        table.add_row(
            str(i),
            f"[{info['color']}]{info['name']}[/{info['color']}]",
            info["description"],
        )
    
    console.print(table)
    console.print()
    
    choice = Prompt.ask(
        "[bold]Enter your choice[/bold]",
        choices=[str(i) for i in range(1, len(lang_list) + 1)],
        default="1",
    )
    
    return lang_list[int(choice) - 1]


# ============================================================================
# Configuration Summary & Confirmation
# ============================================================================

def display_config_summary(config: GenerationConfig, concurrency: int = 5) -> bool:
    """
    Display a summary of the configuration and ask for confirmation.
    
    Returns:
        bool: True if user confirms, False otherwise
    """
    info = DATASET_TYPE_INFO[config.dataset_type]
    lang_info = LANGUAGE_INFO[config.language]
    
    table = Table(
        show_header=False,
        box=ROUNDED,
        border_style="cyan",
        padding=(0, 2),
    )
    
    table.add_column("Setting", style="bold cyan", width=20)
    table.add_column("Value", style="white", width=50)
    
    table.add_row("📝 Topic", f"[bold]{config.topic}[/bold]")
    table.add_row("📊 Dataset Type", f"[{info['color']}]{info['name']}[/{info['color']}]")
    table.add_row("🌐 Language", f"[{lang_info['color']}]{lang_info['name']}[/{lang_info['color']}]")
    
    # Show QC settings for Traditional Chinese with funnel mode info
    if config.language == OutputLanguage.ZH_TW:
        if config.qc_enabled:
            # Calculate over-request amount
            from ollaforge.qc import QualityController
            qc = QualityController(enabled=True, estimated_pass_rate=0.7)
            request_count = qc.calculate_request_count(config.count)
            table.add_row("🔍 QC", f"[bold green]Enabled[/bold green] (confidence ≥ {config.qc_confidence:.0%})")
            table.add_row("🔄 Funnel Mode", f"[dim]Will request ~{request_count} entries to get {config.count} valid[/dim]")
        else:
            table.add_row("🔍 QC", "[dim]Disabled[/dim]")
    
    table.add_row("🔢 Count", f"[bold yellow]{config.count}[/bold yellow] entries")
    table.add_row("🤖 Model", f"[bold green]{config.model}[/bold green]")
    table.add_row("📄 Output", f"[bold]{config.output}[/bold]")
    table.add_row("⚡ Concurrency", f"{concurrency} parallel requests")
    
    panel = Panel(
        table,
        title="[bold white on blue] 📋 Configuration Summary [/bold white on blue]",
        border_style="blue",
        padding=(1, 1),
    )
    
    console.print()
    console.print(panel)
    console.print()
    
    return Confirm.ask("[bold green]🚀 Start generation?[/bold green]", default=True)


# ============================================================================
# Interactive Wizard
# ============================================================================

def run_interactive_wizard() -> Optional[Tuple[GenerationConfig, int]]:
    """
    Run the full interactive configuration wizard.
    
    Returns:
        Tuple of (GenerationConfig, concurrency) if completed, None if cancelled
    """
    try:
        # Display banner
        display_banner()
        
        # Step 1: Topic
        console.print("[bold cyan]📝 Step 1/6: Dataset Topic[/bold cyan]")
        console.print("[dim]Describe what kind of data you want to generate[/dim]\n")
        
        topic = Prompt.ask(
            "[bold]Enter topic description[/bold]",
            default="customer service conversations",
        )
        console.print()
        
        # Step 2: Dataset Type
        console.print("[bold cyan]📊 Step 2/6: Dataset Type[/bold cyan]\n")
        dataset_type = display_dataset_type_menu()
        console.print()
        
        # Step 3: Language
        console.print("[bold cyan]🌐 Step 3/6: Output Language[/bold cyan]\n")
        language = display_language_selection()
        console.print()
        
        # Step 4: Count
        console.print("[bold cyan]🔢 Step 4/6: Number of Entries[/bold cyan]")
        console.print("[dim]How many data entries to generate (1-10,000)[/dim]\n")
        
        count = IntPrompt.ask(
            "[bold]Enter count[/bold]",
            default=50,
        )
        count = max(1, min(10000, count))  # Clamp to valid range
        console.print()
        
        # Step 5: Model
        model = display_model_selection()
        console.print()
        
        # Step 6: Output & Advanced
        console.print("[bold cyan]📄 Step 6/6: Output Settings[/bold cyan]\n")
        
        default_output = f"{dataset_type.value}_dataset.jsonl"
        output = Prompt.ask(
            "[bold]Output filename[/bold]",
            default=default_output,
        )
        
        # QC settings for Traditional Chinese
        qc_enabled = False
        qc_confidence = 0.9
        if language == OutputLanguage.ZH_TW:
            console.print("\n[bold cyan]🔍 Quality Control Settings[/bold cyan]")
            console.print("[dim]QC uses a BERT model to filter out Mainland Chinese expressions[/dim]\n")
            qc_enabled = Confirm.ask(
                "[bold]Enable Taiwan Chinese QC?[/bold]",
                default=True,
            )
            if qc_enabled:
                console.print("[dim]Higher confidence = stricter filtering (0.9 recommended)[/dim]")
                qc_confidence = float(Prompt.ask(
                    "[bold]QC confidence threshold (0.0-1.0)[/bold]",
                    default="0.9",
                ))
                qc_confidence = max(0.0, min(1.0, qc_confidence))
        
        # Advanced settings
        if Confirm.ask("\n[dim]Configure advanced settings?[/dim]", default=False):
            concurrency = IntPrompt.ask(
                "[bold]Concurrency (1-20)[/bold]",
                default=5,
            )
            concurrency = max(1, min(20, concurrency))
        else:
            concurrency = 5
        
        console.print()
        
        # Create config
        config = GenerationConfig(
            topic=topic,
            count=count,
            model=model,
            output=output,
            dataset_type=dataset_type,
            language=language,
            qc_enabled=qc_enabled,
            qc_confidence=qc_confidence,
        )
        
        # Show summary and confirm
        if display_config_summary(config, concurrency):
            return config, concurrency
        else:
            console.print("[yellow]❌ Generation cancelled[/yellow]")
            return None
            
    except KeyboardInterrupt:
        console.print("\n[yellow]❌ Wizard cancelled[/yellow]")
        return None


# ============================================================================
# Quick Menu
# ============================================================================

def display_main_menu() -> str:
    """
    Display the main menu and return the selected action.
    
    Returns:
        str: Selected action ('generate', 'interactive', 'help', 'quit')
    """
    display_banner()
    
    menu_items = [
        ("1", "🚀 Quick Generate", "Generate with command-line arguments"),
        ("2", "✨ Interactive Mode", "Step-by-step configuration wizard"),
        ("3", "📖 Help", "Show usage information"),
        ("4", "🚪 Exit", "Quit OllaForge"),
    ]
    
    table = Table(
        show_header=False,
        box=ROUNDED,
        border_style="bright_blue",
        padding=(0, 2),
    )
    
    table.add_column("Key", style="bold cyan", justify="center", width=5)
    table.add_column("Action", style="bold white", width=25)
    table.add_column("Description", style="dim", width=40)
    
    for key, action, desc in menu_items:
        table.add_row(key, action, desc)
    
    panel = Panel(
        table,
        title="[bold white] 🎯 Main Menu [/bold white]",
        border_style="bright_blue",
        padding=(1, 1),
    )
    
    console.print(panel)
    console.print()
    
    choice = Prompt.ask(
        "[bold]Select an option[/bold]",
        choices=["1", "2", "3", "4"],
        default="2",
    )
    
    return {
        "1": "generate",
        "2": "interactive",
        "3": "help",
        "4": "quit",
    }.get(choice, "interactive")


# ============================================================================
# Help Display
# ============================================================================

def display_help() -> None:
    """Display comprehensive help information."""
    help_content = """
## 🔥 OllaForge - Dataset Generator

### Quick Start
```bash
# Basic usage
python main.py "your topic" --count 50

# With all options
python main.py "topic" -c 100 -m gpt-oss:20b -t sft -o output.jsonl
```

### Dataset Types

| Type | Description | Format |
|------|-------------|--------|
| `sft` | Supervised Fine-tuning | instruction/input/output |
| `pretrain` | Pre-training | raw text |
| `sft_conv` | Conversation | multi-turn dialogue |
| `dpo` | Preference Optimization | chosen/rejected pairs |

### Options

- `-c, --count`: Number of entries (1-10,000)
- `-m, --model`: Ollama model name
- `-t, --type`: Dataset type (sft/pretrain/sft_conv/dpo)
- `-l, --lang`: Output language (en/zh-tw)
- `-o, --output`: Output filename
- `-j, --concurrency`: Parallel requests (1-20)

### Examples

```bash
# Generate SFT data
python main.py "Python coding tutorials" -c 100 -t sft

# Generate conversation data
python main.py "customer support" -t sft_conv -o chats.jsonl

# Generate DPO preference pairs
python main.py "code review" -t dpo -c 50
```
"""
    
    panel = Panel(
        Markdown(help_content),
        title="[bold white] 📖 Help [/bold white]",
        border_style="green",
        padding=(1, 2),
    )
    
    console.print(panel)
    console.print()
    Prompt.ask("[dim]Press Enter to continue...[/dim]", default="")


# ============================================================================
# Status Displays
# ============================================================================

def display_generation_start(config: GenerationConfig) -> None:
    """Display a nice header when generation starts."""
    info = DATASET_TYPE_INFO[config.dataset_type]
    lang_info = LANGUAGE_INFO[config.language]
    
    content = Text()
    content.append("🚀 Starting Generation\n\n", style="bold white")
    content.append(f"Topic: ", style="dim")
    content.append(f"{config.topic}\n", style="bold")
    content.append(f"Type: ", style="dim")
    content.append(f"{info['name']}\n", style=info['color'])
    content.append(f"Language: ", style="dim")
    content.append(f"{lang_info['name']}\n", style=lang_info['color'])
    
    # Show QC status for Traditional Chinese
    if config.language == OutputLanguage.ZH_TW:
        content.append(f"QC: ", style="dim")
        if config.qc_enabled:
            content.append(f"Enabled (≥{config.qc_confidence:.0%})\n", style="green")
        else:
            content.append(f"Disabled\n", style="dim")
    
    content.append(f"Target: ", style="dim")
    content.append(f"{config.count} entries", style="bold yellow")
    
    panel = Panel(
        content,
        border_style="green",
        padding=(0, 2),
    )
    
    console.print(panel)
    console.print()


def display_success(output_file: str, count: int, duration: float) -> None:
    """Display success message after generation completes."""
    entries_per_sec = count / duration if duration > 0 else 0
    
    content = Text()
    content.append("✅ Generation Complete!\n\n", style="bold green")
    content.append(f"📄 Output: ", style="dim")
    content.append(f"{output_file}\n", style="bold cyan")
    content.append(f"📊 Entries: ", style="dim")
    content.append(f"{count}\n", style="bold yellow")
    content.append(f"⏱️  Duration: ", style="dim")
    content.append(f"{duration:.1f}s ", style="bold")
    content.append(f"({entries_per_sec:.1f} entries/sec)", style="dim")
    
    panel = Panel(
        Align.center(content),
        border_style="green",
        padding=(1, 2),
    )
    
    console.print()
    console.print(panel)


def display_error(message: str, hint: Optional[str] = None) -> None:
    """Display an error message with optional hint."""
    content = Text()
    content.append("❌ Error\n\n", style="bold red")
    content.append(message, style="white")
    
    if hint:
        content.append(f"\n\n💡 ", style="yellow")
        content.append(hint, style="dim")
    
    panel = Panel(
        content,
        border_style="red",
        padding=(0, 2),
    )
    
    console.print(panel)


# ============================================================================
# Entry Point for Interactive Mode
# ============================================================================

def main_interactive() -> Optional[Tuple[GenerationConfig, int]]:
    """
    Main entry point for interactive mode.
    
    Returns:
        Tuple of (config, concurrency) if user wants to generate, None otherwise
    """
    while True:
        action = display_main_menu()
        
        if action == "quit":
            console.print("[cyan]👋 Goodbye![/cyan]")
            return None
        
        elif action == "help":
            display_help()
            continue
        
        elif action == "interactive":
            result = run_interactive_wizard()
            if result:
                return result
            continue
        
        elif action == "generate":
            console.print("[yellow]Use command-line arguments for quick generation:[/yellow]")
            console.print("[dim]python main.py \"your topic\" --count 50 --type sft[/dim]\n")
            Prompt.ask("[dim]Press Enter to continue...[/dim]", default="")
            continue
    
    return None


# ============================================================================
# Augmentation Interactive Wizard
# ============================================================================

def display_file_selection(prompt_text: str = "Enter JSONL file path") -> Optional[str]:
    """
    Prompt user to enter a file path and validate it exists.
    
    Args:
        prompt_text: The prompt text to display
        
    Returns:
        Valid file path or None if cancelled
    """
    from pathlib import Path
    import os
    
    while True:
        file_path = Prompt.ask(f"[bold]{prompt_text}[/bold]")
        
        if not file_path or not file_path.strip():
            if Confirm.ask("[yellow]No file entered. Cancel?[/yellow]", default=False):
                return None
            continue
        
        file_path = file_path.strip()
        path = Path(file_path)
        
        if not path.exists():
            console.print(f"[red]❌ File not found: {file_path}[/red]")
            continue
        
        if not path.is_file():
            console.print(f"[red]❌ Not a file: {file_path}[/red]")
            continue
        
        if not os.access(path, os.R_OK):
            console.print(f"[red]❌ No read permission: {file_path}[/red]")
            continue
        
        return file_path


def display_field_selection(field_names: List[str], allow_new: bool = True) -> Tuple[str, bool]:
    """
    Display available fields and let user select one or create new.
    
    Args:
        field_names: List of available field names
        allow_new: Whether to allow creating a new field
        
    Returns:
        Tuple of (field_name, is_new_field)
    """
    console.print("[bold cyan]📋 Available Fields[/bold cyan]\n")
    
    table = Table(
        show_header=True,
        header_style="bold white on dark_blue",
        box=ROUNDED,
        border_style="blue",
    )
    
    table.add_column("#", style="bold cyan", justify="center", width=3)
    table.add_column("Field Name", style="bold", width=30)
    
    for i, field in enumerate(field_names, 1):
        table.add_row(str(i), field)
    
    if allow_new:
        table.add_row("N", "[yellow]Create new field...[/yellow]")
    
    console.print(table)
    console.print()
    
    valid_choices = [str(i) for i in range(1, len(field_names) + 1)]
    if allow_new:
        valid_choices.append("N")
        valid_choices.append("n")
    
    while True:
        choice = Prompt.ask(
            "[bold]Select field number or 'N' for new[/bold]" if allow_new else "[bold]Select field number[/bold]",
            default="1",
        )
        
        if choice.upper() == "N" and allow_new:
            new_field = Prompt.ask("[bold]Enter new field name[/bold]")
            if new_field and new_field.strip():
                return new_field.strip(), True
            console.print("[red]Field name cannot be empty[/red]")
            continue
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(field_names):
                return field_names[idx], False
        except ValueError:
            pass
        
        console.print(f"[red]Invalid choice. Enter 1-{len(field_names)}" + (" or 'N'" if allow_new else "") + "[/red]")


def display_context_field_selection(field_names: List[str], target_field: str) -> List[str]:
    """
    Let user select context fields to include in augmentation prompt.
    
    Args:
        field_names: List of available field names
        target_field: The target field (excluded from selection)
        
    Returns:
        List of selected context field names
    """
    # Filter out target field from options
    available_fields = [f for f in field_names if f != target_field]
    
    if not available_fields:
        console.print("[dim]No other fields available for context[/dim]")
        return []
    
    console.print("[bold cyan]📎 Select Context Fields[/bold cyan]")
    console.print("[dim]These fields will be included as context for the AI[/dim]\n")
    
    table = Table(
        show_header=True,
        header_style="bold white on dark_green",
        box=ROUNDED,
        border_style="green",
    )
    
    table.add_column("#", style="bold cyan", justify="center", width=3)
    table.add_column("Field Name", style="bold", width=30)
    
    for i, field in enumerate(available_fields, 1):
        table.add_row(str(i), field)
    
    table.add_row("A", "[green]Select all[/green]")
    table.add_row("S", "[yellow]Skip (no context)[/yellow]")
    
    console.print(table)
    console.print()
    
    choice = Prompt.ask(
        "[bold]Enter field numbers (comma-separated), 'A' for all, or 'S' to skip[/bold]",
        default="A",
    )
    
    if choice.upper() == "S":
        return []
    
    if choice.upper() == "A":
        return available_fields
    
    # Parse comma-separated numbers
    selected = []
    for part in choice.split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(available_fields):
                field = available_fields[idx]
                if field not in selected:
                    selected.append(field)
        except ValueError:
            continue
    
    return selected


def display_augmentation_preview(
    preview_results: List[Tuple[dict, dict]],
    target_field: str
) -> bool:
    """
    Display preview results and ask for confirmation.
    
    Args:
        preview_results: List of (original, augmented) entry tuples
        target_field: The field being augmented
        
    Returns:
        True if user confirms, False to retry with different instruction
    """
    console.print()
    console.print(Panel.fit(
        Text("🔍 Preview Results", style="bold cyan"),
        border_style="cyan",
    ))
    console.print()
    
    for i, (original, augmented) in enumerate(preview_results, 1):
        console.print(f"[bold]━━━ Entry {i} ━━━[/bold]")
        
        original_value = original.get(target_field, "[dim](not present)[/dim]")
        augmented_value = augmented.get(target_field, "[dim](not present)[/dim]")
        
        # Truncate long values for display
        if isinstance(original_value, str) and len(original_value) > 200:
            original_value = original_value[:200] + "..."
        if isinstance(augmented_value, str) and len(augmented_value) > 200:
            augmented_value = augmented_value[:200] + "..."
        
        console.print(f"[dim]Original {target_field}:[/dim]")
        console.print(f"  {original_value}")
        console.print(f"[green]Augmented {target_field}:[/green]")
        console.print(f"  {augmented_value}")
        console.print()
    
    return Confirm.ask(
        "[bold green]✓ Results look good? Proceed with full augmentation?[/bold green]",
        default=True
    )


def display_augmentation_config_summary(
    input_file: str,
    output_file: str,
    target_field: str,
    instruction: str,
    model: str,
    concurrency: int,
    is_new_field: bool,
    context_fields: List[str],
    entry_count: int,
    language: OutputLanguage,
) -> bool:
    """
    Display augmentation configuration summary and ask for confirmation.
    
    Returns:
        True if user confirms, False otherwise
    """
    lang_info = LANGUAGE_INFO[language]
    
    table = Table(
        show_header=False,
        box=ROUNDED,
        border_style="cyan",
        padding=(0, 2),
    )
    
    table.add_column("Setting", style="bold cyan", width=20)
    table.add_column("Value", style="white", width=50)
    
    table.add_row("📂 Input File", f"[bold]{input_file}[/bold]")
    table.add_row("📊 Entries", f"[bold yellow]{entry_count}[/bold yellow]")
    table.add_row("🎯 Target Field", f"[bold]{target_field}[/bold]" + (" [cyan](new)[/cyan]" if is_new_field else ""))
    table.add_row("📝 Instruction", f"{instruction[:50]}{'...' if len(instruction) > 50 else ''}")
    table.add_row("🌐 Language", f"[{lang_info['color']}]{lang_info['name']}[/{lang_info['color']}]")
    table.add_row("🤖 Model", f"[bold green]{model}[/bold green]")
    table.add_row("📄 Output", f"[bold]{output_file}[/bold]")
    table.add_row("⚡ Concurrency", f"{concurrency} parallel requests")
    
    if context_fields:
        table.add_row("📎 Context", ", ".join(context_fields))
    
    panel = Panel(
        table,
        title="[bold white on blue] 🔧 Augmentation Configuration [/bold white on blue]",
        border_style="blue",
        padding=(1, 1),
    )
    
    console.print()
    console.print(panel)
    console.print()
    
    return Confirm.ask("[bold green]🚀 Start augmentation?[/bold green]", default=True)


def augment_interactive() -> Optional[Tuple[str, str, str, str, str, int, bool, List[str], bool, OutputLanguage]]:
    """
    Run the interactive augmentation wizard.
    
    Guides the user through:
    1. Input file selection
    2. Field selection (existing or new)
    3. Instruction input
    4. Context field selection
    5. Model and output configuration
    6. Preview before full processing
    
    Returns:
        Tuple of (input_file, field, instruction, output, model, concurrency, 
                  new_field, context_fields, preview, language) if completed,
        None if cancelled
        
    Requirements satisfied:
    - 6.4: Interactive mode guides user through field selection and instruction input
    - 7.3: Offer preview before full processing
    - 7.4: Allow modifying instruction and retry on preview rejection
    """
    from pathlib import Path
    from .file_manager import read_jsonl_file, FileOperationError
    from .augmentor import DatasetAugmentor
    from .models import AugmentationConfig
    
    try:
        # Display banner
        display_banner()
        console.print("[bold cyan]🔧 Dataset Augmentation Wizard[/bold cyan]\n")
        
        # Step 1: Input file selection
        console.print("[bold cyan]📂 Step 1/6: Select Input File[/bold cyan]")
        console.print("[dim]Enter the path to your JSONL dataset file[/dim]\n")
        
        input_file = display_file_selection("Enter JSONL file path")
        if input_file is None:
            console.print("[yellow]❌ Wizard cancelled[/yellow]")
            return None
        
        # Load dataset to get field names
        console.print(f"\n[dim]Loading dataset from {input_file}...[/dim]")
        try:
            entries, field_names = read_jsonl_file(input_file, return_field_names=True)
        except FileOperationError as e:
            console.print(f"[red]❌ {str(e)}[/red]")
            return None
        
        console.print(f"[green]✓ Loaded {len(entries)} entries with {len(field_names)} fields[/green]\n")
        
        if not entries:
            console.print("[red]❌ Dataset is empty[/red]")
            return None
        
        # Step 2: Field selection
        console.print("[bold cyan]🎯 Step 2/6: Select Target Field[/bold cyan]")
        console.print("[dim]Choose a field to augment or create a new one[/dim]\n")
        
        target_field, is_new_field = display_field_selection(field_names, allow_new=True)
        console.print(f"\n[green]✓ Selected field: {target_field}" + (" (new)" if is_new_field else "") + "[/green]\n")
        
        # Step 3: Instruction input (with retry loop for preview)
        instruction = None
        preview_confirmed = False
        
        while not preview_confirmed:
            console.print("[bold cyan]📝 Step 3/6: Enter Augmentation Instruction[/bold cyan]")
            console.print("[dim]Describe how the AI should augment this field[/dim]\n")
            
            console.print("[dim]Examples:[/dim]")
            console.print("[dim]  • Translate to English[/dim]")
            console.print("[dim]  • Rate difficulty as easy/medium/hard[/dim]")
            console.print("[dim]  • Summarize in one sentence[/dim]")
            console.print("[dim]  • Add more detail and context[/dim]\n")
            
            if instruction:
                console.print(f"[yellow]Previous instruction: {instruction}[/yellow]\n")
            
            new_instruction = Prompt.ask("[bold]Enter instruction[/bold]")
            if not new_instruction or not new_instruction.strip():
                console.print("[red]Instruction cannot be empty[/red]")
                continue
            
            instruction = new_instruction.strip()
            console.print()
            
            # Step 4: Context field selection
            console.print("[bold cyan]📎 Step 4/6: Select Context Fields[/bold cyan]")
            console.print("[dim]Choose which fields to include as context for the AI[/dim]\n")
            
            context_fields = display_context_field_selection(field_names, target_field)
            if context_fields:
                console.print(f"\n[green]✓ Context fields: {', '.join(context_fields)}[/green]\n")
            else:
                console.print("\n[dim]No context fields selected[/dim]\n")
            
            # Step 5: Model and output configuration
            console.print("[bold cyan]🤖 Step 5/6: Model & Output Settings[/bold cyan]\n")
            
            model = display_model_selection()
            console.print()
            
            # Language selection
            language = display_language_selection()
            console.print()
            
            # Output file
            input_path = Path(input_file)
            default_output = str(input_path.parent / f"{input_path.stem}_augmented{input_path.suffix}")
            output_file = Prompt.ask(
                "[bold]Output filename[/bold]",
                default=default_output,
            )
            console.print()
            
            # Concurrency
            if Confirm.ask("[dim]Configure advanced settings?[/dim]", default=False):
                concurrency = IntPrompt.ask(
                    "[bold]Concurrency (1-20)[/bold]",
                    default=5,
                )
                concurrency = max(1, min(20, concurrency))
            else:
                concurrency = 5
            
            console.print()
            
            # Step 6: Preview
            console.print("[bold cyan]🔍 Step 6/6: Preview[/bold cyan]")
            console.print("[dim]Testing augmentation on sample entries...[/dim]\n")
            
            # Create config for preview
            preview_config = AugmentationConfig(
                input_file=input_file,
                output_file=output_file,
                target_field=target_field,
                instruction=instruction,
                model=model,
                language=language,
                create_new_field=is_new_field,
                context_fields=context_fields,
                preview_count=3,
            )
            
            # Run preview
            augmentor = DatasetAugmentor(preview_config)
            console.print("[dim]Generating preview (this may take a moment)...[/dim]\n")
            
            try:
                preview_results = augmentor.preview(entries)
            except Exception as e:
                console.print(f"[red]❌ Preview failed: {str(e)}[/red]")
                if Confirm.ask("[yellow]Try again with different instruction?[/yellow]", default=True):
                    continue
                return None
            
            # Display preview and ask for confirmation
            preview_confirmed = display_augmentation_preview(preview_results, target_field)
            
            if not preview_confirmed:
                # Requirement 7.4: Allow modifying instruction and retry
                if Confirm.ask("[yellow]Modify instruction and try again?[/yellow]", default=True):
                    console.print()
                    continue
                else:
                    console.print("[yellow]❌ Wizard cancelled[/yellow]")
                    return None
        
        # Show final summary and confirm
        if not display_augmentation_config_summary(
            input_file=input_file,
            output_file=output_file,
            target_field=target_field,
            instruction=instruction,
            model=model,
            concurrency=concurrency,
            is_new_field=is_new_field,
            context_fields=context_fields,
            entry_count=len(entries),
            language=language,
        ):
            console.print("[yellow]❌ Augmentation cancelled[/yellow]")
            return None
        
        # Return all configuration values
        return (
            input_file,
            target_field,
            instruction,
            output_file,
            model,
            concurrency,
            is_new_field,
            context_fields,
            False,  # preview already done
            language,
        )
        
    except KeyboardInterrupt:
        console.print("\n[yellow]❌ Wizard cancelled[/yellow]")
        return None


def main_interactive_router() -> None:
    """
    Main router for interactive mode - allows user to choose between generation and augmentation.
    """
    display_banner()
    
    console.print()
    console.print("[bold cyan]Welcome to OllaForge Interactive Mode![/bold cyan]")
    console.print()
    
    # Create choice table
    table = Table(show_header=False, box=ROUNDED, border_style="cyan")
    table.add_column("Option", style="bold", width=8)
    table.add_column("Description", style="")
    
    table.add_row("1", "🆕 [green]Generate[/green] - Create new datasets from scratch")
    table.add_row("2", "🔄 [blue]Augment[/blue] - Enhance existing datasets")
    table.add_row("3", "❌ [red]Exit[/red] - Quit the application")
    
    console.print(Align.center(table))
    console.print()
    
    while True:
        try:
            choice = Prompt.ask(
                "[bold]What would you like to do?[/bold]",
                choices=["1", "2", "3"],
                default="1"
            )
            
            if choice == "1":
                # Generation mode
                console.print("\n[green]🆕 Entering Dataset Generation Mode...[/green]\n")
                result = main_interactive()
                if result is not None:
                    config, concurrency = result
                    from .cli import _run_generation
                    _run_generation(config, concurrency)
                break
                
            elif choice == "2":
                # Augmentation mode
                console.print("\n[blue]🔄 Entering Dataset Augmentation Mode...[/blue]\n")
                result = augment_interactive()
                if result is not None:
                    input_file, field, instruction, output, model, concurrency, new_field, context, preview, language = result
                    from .cli import _run_augmentation
                    _run_augmentation(
                        input_file=input_file,
                        output_file=output,
                        field=field,
                        instruction=instruction,
                        model=model,
                        concurrency=concurrency,
                        new_field=new_field,
                        context_fields=context,
                        preview_mode=preview,
                        preview_count=3,
                        language=language,
                        force=False,
                    )
                break
                
            elif choice == "3":
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
                
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Error: {str(e)}[/red]")
            console.print("[yellow]Please try again or press Ctrl+C to exit.[/yellow]\n")