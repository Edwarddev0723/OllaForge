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


class GenerationConfig(BaseModel):
    """Configuration for dataset generation."""
    topic: str = Field(..., description="Description of the dataset content to generate")
    count: int = Field(10, ge=1, le=10000, description="Number of data entries to generate")
    model: str = Field("llama3", description="Ollama model to use")
    output: str = Field("dataset.jsonl", description="Output filename")
    dataset_type: DatasetType = Field(DatasetType.SFT, description="Type of dataset to generate")
    
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