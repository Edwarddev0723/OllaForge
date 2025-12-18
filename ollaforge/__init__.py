"""
OllaForge - Dataset generation using local Ollama models

A Python CLI tool for generating high-quality training datasets using local
Ollama models. Supports multiple dataset formats (SFT, Pre-training, DPO)
with built-in quality control for Traditional Chinese (Taiwan).

Features:
- Multiple dataset formats (SFT, Pre-training, Conversation, DPO)
- Structured JSON output with schema validation
- Quality control for Traditional Chinese (Taiwan)
- Concurrent batch generation for performance
- Interactive wizard mode

Usage:
    ollaforge "topic description" --count 100 --type sft_conv --lang zh-tw
    ollaforge -i  # Interactive mode
"""

__version__ = "1.1.0"
__author__ = "OllaForge Team"
__description__ = "CLI tool for generating datasets using local Ollama models"

# Core models
from .models import (
    DatasetType,
    OutputLanguage,
    GenerationConfig,
    GenerationResult,
    DataEntry,
    PretrainEntry,
    SFTConversationEntry,
    DPOEntry,
)

# Client functions
from .client import (
    generate_data,
    generate_data_concurrent,
    get_available_models,
    OllamaConnectionError,
    OllamaGenerationError,
)

# Processing
from .processor import (
    clean_json,
    clean_json_array,
    validate_entry,
    process_model_response,
)

# File operations
from .file_manager import (
    write_jsonl_file,
    read_jsonl_file,
    FileOperationError,
    DiskSpaceError,
)

# Augmentation
from .augmentor import (
    DatasetAugmentor,
    create_augmentation_prompt,
)

# Quality control
from .qc import (
    QualityController,
    predict_language,
    is_taiwan_chinese,
)

# CLI
from .cli import app

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    # Models
    "DatasetType",
    "OutputLanguage",
    "GenerationConfig",
    "GenerationResult",
    "DataEntry",
    "PretrainEntry",
    "SFTConversationEntry",
    "DPOEntry",
    # Client
    "generate_data",
    "generate_data_concurrent",
    "get_available_models",
    "OllamaConnectionError",
    "OllamaGenerationError",
    # Processing
    "clean_json",
    "clean_json_array",
    "validate_entry",
    "process_model_response",
    # File operations
    "write_jsonl_file",
    "read_jsonl_file",
    "FileOperationError",
    "DiskSpaceError",
    # Augmentation
    "DatasetAugmentor",
    "create_augmentation_prompt",
    # QC
    "QualityController",
    "predict_language",
    "is_taiwan_chinese",
    # CLI
    "app",
]
