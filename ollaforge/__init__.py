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

__version__ = "1.3.0"
__author__ = "OllaForge Team"
__description__ = "AI-Powered Dataset Generator & Augmentor for LLM Fine-tuning with HuggingFace Integration"

# Core models
# Augmentation
from .augmentor import (
    DatasetAugmentor,
    create_augmentation_prompt,
)

# CLI
from .cli import app

# Client functions
from .client import (
    OllamaConnectionError,
    OllamaGenerationError,
    generate_data,
    generate_data_concurrent,
    get_available_models,
)

# File operations
from .file_manager import (
    DiskSpaceError,
    FileOperationError,
    read_jsonl_file,
    write_jsonl_file,
)

# Multi-format support
from .formats import (
    FileFormat,
    FormatError,
    detect_format,
    get_format_description,
    get_supported_formats,
)
from .models import (
    DataEntry,
    DatasetType,
    DPOEntry,
    GenerationConfig,
    GenerationResult,
    OutputLanguage,
    PretrainEntry,
    SFTConversationEntry,
)

# Processing
from .processor import (
    clean_json,
    clean_json_array,
    process_model_response,
    validate_entry,
)

# Quality control
from .qc import (
    QualityController,
    is_taiwan_chinese,
    predict_language,
)

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
    # Multi-format support
    "FileFormat",
    "FormatError",
    "detect_format",
    "get_supported_formats",
    "get_format_description",
    # QC
    "QualityController",
    "predict_language",
    "is_taiwan_chinese",
    # CLI
    "app",
]
