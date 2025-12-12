"""
OllaForge - Dataset generation using local Ollama models
"""

__version__ = "1.0.0"
__author__ = "OllaForge Team"
__description__ = "CLI tool for generating datasets using local Ollama models"

from .models import DatasetType, GenerationConfig, GenerationResult
from .interactive import (
    display_banner,
    display_main_menu,
    run_interactive_wizard,
    main_interactive,
)

__all__ = [
    "DatasetType",
    "GenerationConfig", 
    "GenerationResult",
    "display_banner",
    "display_main_menu",
    "run_interactive_wizard",
    "main_interactive",
]