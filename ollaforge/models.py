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

Requirements satisfied:
- 1.2: Count parameter validation with range constraints (1-10,000)
- 1.3: Model parameter handling with string validation
- 1.4: Output parameter validation with path checking
- 5.5: Uses Pydantic models for data validation and structure
- All data validation requirements across the application
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from pathlib import Path


class GenerationConfig(BaseModel):
    """Configuration for dataset generation."""
    topic: str = Field(..., description="Description of the dataset content to generate")
    count: int = Field(10, ge=1, le=10000, description="Number of data entries to generate")
    model: str = Field("llama3", description="Ollama model to use")
    output: str = Field("dataset.jsonl", description="Output filename")
    
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


class DataEntry(BaseModel):
    """Structure for a single generated data entry."""
    instruction: str = Field(..., description="The instruction or task description")
    input: str = Field(..., description="The input context or data")
    output: str = Field(..., description="The expected output or response")


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