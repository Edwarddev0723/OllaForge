"""
Data models for OllaForge using Pydantic for validation and structure.

This module defines the core data structures used throughout OllaForge, providing
robust validation, type safety, and clear interfaces between components. All models
use Pydantic for automatic validation and serialization.

Key features:
- Comprehensive parameter validation with meaningful error messages
- Type-safe data structures for all application components
- Automatic JSON serialization/deserialization
- Built-in validation rules and constraints
- Clear separation of concerns between different data types
- Support for multiple dataset types (SFT, Pre-training, DPO, etc.)

Requirements satisfied:
- 1.2: Count parameter validation with range constraints (1-10,000)
- 1.3: Model parameter handling with string validation
- 1.4: Output parameter validation with path checking
- 5.5: Uses Pydantic models for data validation and structure
- All data validation requirements across the application
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Union
from pathlib import Path
from enum import Enum


class DatasetType(str, Enum):
    """
    Supported dataset types for different training stages.
    
    Formats follow HuggingFace/LLaMA-Factory conventions:
    - SFT: Supervised Fine-tuning with instruction/input/output
    - PRETRAIN: Pre-training with raw text
    - SFT_CONVERSATION: Multi-turn conversation format
    - DPO: Direct Preference Optimization with chosen/rejected pairs
    """
    SFT = "sft"                      # 推理資料集 (Supervised Fine-tuning)
    PRETRAIN = "pretrain"            # Pre-trained stage
    SFT_CONVERSATION = "sft_conv"    # Supervised Fine-tuning (conversation)
    DPO = "dpo"                      # Preference Stage (DPO)


class OutputLanguage(str, Enum):
    """
    Supported output languages for generated datasets.
    """
    EN = "en"           # English
    ZH_TW = "zh-tw"     # 繁體中文（台灣用語）


class GenerationConfig(BaseModel):
    """Configuration for dataset generation."""
    topic: str = Field(..., description="Description of the dataset content to generate")
    count: int = Field(10, ge=1, le=10000, description="Number of data entries to generate")
    model: str = Field("gpt-oss:20b", description="Ollama model to use")
    output: str = Field("dataset.jsonl", description="Output filename")
    dataset_type: DatasetType = Field(DatasetType.SFT, description="Type of dataset to generate")
    language: OutputLanguage = Field(OutputLanguage.EN, description="Output language for generated content")
    qc_enabled: bool = Field(True, description="Enable QC for Traditional Chinese (Taiwan)")
    qc_confidence: float = Field(0.9, ge=0.0, le=1.0, description="QC confidence threshold for Taiwan Chinese")
    
    @validator('topic')
    def topic_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Topic cannot be empty')
        return v.strip()
    
    @validator('output')
    def output_must_be_valid_path(cls, v):
        # Basic validation for output path
        if not v or not v.strip():
            raise ValueError('Output filename cannot be empty')
        return v.strip()


# ============================================================================
# Dataset Entry Models - HuggingFace/LLaMA-Factory Compatible Formats
# ============================================================================

class DataEntry(BaseModel):
    """
    SFT (Supervised Fine-tuning) format - Alpaca style.
    Compatible with: HuggingFace datasets, LLaMA-Factory alpaca format
    
    Example:
    {"instruction": "Summarize the text", "input": "Long text...", "output": "Summary..."}
    """
    instruction: str = Field(..., description="The instruction or task description")
    input: str = Field(..., description="The input context or data")
    output: str = Field(..., description="The expected output or response")


class PretrainEntry(BaseModel):
    """
    Pre-training format - Raw text for continued pre-training.
    Compatible with: HuggingFace datasets, LLaMA-Factory pretrain format
    
    Example:
    {"text": "This is a long document for pre-training..."}
    """
    text: str = Field(..., description="Raw text content for pre-training")


class ConversationMessage(BaseModel):
    """Single message in a conversation."""
    role: Literal["system", "user", "assistant"] = Field(..., description="Role of the speaker")
    content: str = Field(..., description="Message content")


class SFTConversationEntry(BaseModel):
    """
    SFT Conversation format - Multi-turn dialogue.
    Compatible with: HuggingFace ChatML, LLaMA-Factory sharegpt format
    
    Example:
    {"conversations": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"}
    ]}
    """
    conversations: List[ConversationMessage] = Field(..., description="List of conversation turns")


class DPOEntry(BaseModel):
    """
    DPO (Direct Preference Optimization) format.
    Compatible with: HuggingFace TRL DPOTrainer, LLaMA-Factory dpo format
    
    Example:
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "France is a country in Europe."
    }
    """
    prompt: str = Field(..., description="The input prompt/question")
    chosen: str = Field(..., description="The preferred/better response")
    rejected: str = Field(..., description="The less preferred/worse response")


# Type alias for all entry types
DatasetEntry = Union[DataEntry, PretrainEntry, SFTConversationEntry, DPOEntry]


class GenerationResult(BaseModel):
    """Result of a dataset generation operation."""
    success_count: int = Field(..., ge=0, description="Number of successfully generated entries")
    total_requested: int = Field(..., ge=0, description="Total number of entries requested")
    output_file: str = Field(..., description="Path to the output file")
    duration: float = Field(..., ge=0, description="Total generation time in seconds")
    errors: List[str] = Field(default_factory=list, description="List of error messages encountered")
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_requested == 0:
            return 0.0
        return (self.success_count / self.total_requested) * 100


# ============================================================================
# Dataset Augmentation Models
# ============================================================================

class AugmentationConfig(BaseModel):
    """
    Configuration for dataset augmentation.
    
    This model defines all parameters needed to augment an existing JSONL dataset
    by modifying or adding fields using AI-generated content.
    
    Requirements satisfied:
    - 2.1: Target field specification and validation
    - 2.2: Field existence validation
    - 2.3: Augmentation instruction handling
    - 2.4: New field creation support
    """
    input_file: str = Field(..., description="Path to source JSONL file")
    output_file: str = Field(..., description="Path to output JSONL file")
    target_field: str = Field(..., description="Field name to augment or create")
    instruction: str = Field(..., description="AI instruction for augmentation")
    model: str = Field("llama3.2", description="Ollama model to use")
    language: OutputLanguage = Field(OutputLanguage.EN, description="Output language")
    create_new_field: bool = Field(False, description="Whether to create a new field")
    context_fields: List[str] = Field(default_factory=list, description="Additional fields to include as context")
    preview_count: int = Field(3, ge=1, le=10, description="Number of entries for preview")
    
    @validator('input_file')
    def input_file_must_be_valid(cls, v):
        """Validate that input file path is not empty."""
        if not v or not v.strip():
            raise ValueError('Input file path cannot be empty')
        return v.strip()
    
    @validator('output_file')
    def output_file_must_be_valid(cls, v):
        """Validate that output file path is not empty."""
        if not v or not v.strip():
            raise ValueError('Output file path cannot be empty')
        return v.strip()
    
    @validator('target_field')
    def target_field_must_be_valid(cls, v):
        """Validate that target field name is not empty."""
        if not v or not v.strip():
            raise ValueError('Target field name cannot be empty')
        return v.strip()
    
    @validator('instruction')
    def instruction_must_not_be_empty(cls, v):
        """Validate that instruction is not empty."""
        if not v or not v.strip():
            raise ValueError('Augmentation instruction cannot be empty')
        return v.strip()


class FieldValidationError(Exception):
    """Raised when field validation fails."""
    def __init__(self, message: str, available_fields: List[str]):
        self.message = message
        self.available_fields = available_fields
        super().__init__(message)


def validate_target_field(
    entries: List[dict],
    target_field: str,
    create_new_field: bool = False
) -> bool:
    """
    Validate that target field exists in dataset entries or can be created.
    
    Args:
        entries: List of dataset entries (dictionaries)
        target_field: The field name to validate
        create_new_field: If True, allows non-existing fields
        
    Returns:
        bool: True if field is valid
        
    Raises:
        FieldValidationError: If field doesn't exist and create_new_field is False
        
    Requirements satisfied:
    - 2.1: Accept existing field names
    - 2.2: Reject non-existing fields with available field list
    """
    if not entries:
        if create_new_field:
            return True
        raise FieldValidationError(
            f"Cannot validate field '{target_field}': dataset is empty",
            available_fields=[]
        )
    
    # Collect all unique field names from all entries
    available_fields: set = set()
    for entry in entries:
        if isinstance(entry, dict):
            available_fields.update(entry.keys())
    
    # Check if target field exists
    if target_field in available_fields:
        return True
    
    # If creating new field is allowed, accept any valid field name
    if create_new_field:
        return True
    
    # Field doesn't exist and we're not creating new - raise error
    raise FieldValidationError(
        f"Field '{target_field}' not found in dataset",
        available_fields=sorted(available_fields)
    )


class AugmentationResult(BaseModel):
    """
    Result of dataset augmentation operation.
    
    This model captures the outcome of an augmentation run, including
    success/failure counts and any errors encountered.
    
    Requirements satisfied:
    - 5.2: Summary statistics (total, success, failures)
    - 5.3: Error accumulation and reporting
    """
    total_entries: int = Field(..., ge=0, description="Total entries in source dataset")
    success_count: int = Field(..., ge=0, description="Successfully augmented entries")
    failure_count: int = Field(..., ge=0, description="Failed augmentation attempts")
    output_file: str = Field(..., description="Path to output file")
    duration: float = Field(..., ge=0, description="Total processing time in seconds")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_entries == 0:
            return 0.0
        return (self.success_count / self.total_entries) * 100