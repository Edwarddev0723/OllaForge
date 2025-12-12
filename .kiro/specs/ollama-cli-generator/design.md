# OllaForge CLI Design Document

## Overview

OllaForge is a Python CLI application that generates structured datasets using local Ollama AI models. The application follows a modular architecture with clear separation of concerns, utilizing modern Python libraries for CLI interaction, data validation, and user experience enhancement.

## Architecture

The application follows a layered architecture:

1. **CLI Layer**: Handles command-line argument parsing and user interaction using Typer
2. **Generation Layer**: Manages AI model communication and data generation logic
3. **Processing Layer**: Handles data cleaning, validation, and formatting
4. **Output Layer**: Manages file operations and result persistence
5. **UI Layer**: Provides rich terminal feedback and progress tracking

## Components and Interfaces

### CLI Interface (Typer)
- **Purpose**: Parse command-line arguments and orchestrate application flow
- **Key Functions**:
  - `main()`: Entry point with Typer decorators for argument parsing
  - Parameter validation and default value handling
  - Error handling for invalid inputs

### Ollama Client
- **Purpose**: Communicate with local Ollama API
- **Key Functions**:
  - `generate_data()`: Send prompts to Ollama and receive responses
  - Connection management and error handling
  - Model selection and configuration

### Data Processor
- **Purpose**: Clean and validate generated content
- **Key Functions**:
  - `clean_json()`: Extract valid JSON from model responses
  - `validate_entry()`: Ensure data structure compliance
  - Markdown and noise removal

### File Manager
- **Purpose**: Handle output file operations
- **Key Functions**:
  - JSONL file writing with proper formatting
  - File existence and permission checking
  - Atomic write operations for data integrity

### Progress Tracker (Rich)
- **Purpose**: Provide visual feedback during generation
- **Key Functions**:
  - Progress bar updates
  - Status messages and error display
  - Summary report generation

## Data Models

### GenerationConfig (Pydantic)
```python
class GenerationConfig(BaseModel):
    topic: str
    count: int = 10
    model: str = "llama3"
    output: str = "dataset.jsonl"
```

### DataEntry (Pydantic)
```python
class DataEntry(BaseModel):
    instruction: str
    input: str
    output: str
```

### GenerationResult
```python
class GenerationResult(BaseModel):
    success_count: int
    total_requested: int
    output_file: str
    duration: float
    errors: List[str]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*
Property 1: Topic parameter acceptance
*For any* valid topic string provided as CLI parameter, the system should accept and pass it to the generation process
**Validates: Requirements 1.1**

Property 2: Count parameter controls output quantity
*For any* valid count parameter, the generated output should contain exactly that number of entries
**Validates: Requirements 1.2**

Property 3: Model parameter selection
*For any* valid Ollama model name provided, the system should use that specific model for generation requests
**Validates: Requirements 1.3**

Property 4: Output filename specification
*For any* valid filename provided as output parameter, the results should be written to that exact file location
**Validates: Requirements 1.4**

Property 5: Ollama API connection establishment
*For any* generation request, the system should attempt to establish connection with Ollama API on localhost:11434
**Validates: Requirements 2.1**

Property 6: JSON format prompt engineering
*For any* request sent to Ollama, the prompt should include instructions for JSON output format
**Validates: Requirements 2.2**

Property 7: API error handling
*For any* API timeout or connection error, the system should handle the error gracefully and continue processing
**Validates: Requirements 2.5**

Property 8: JSON extraction from responses
*For any* model response containing JSON within markdown or noise, the clean_json function should extract valid JSON
**Validates: Requirements 3.1**

Property 9: JSONL format compliance
*For any* data written to output file, each line should contain a valid JSON object
**Validates: Requirements 3.2**

Property 10: Invalid JSON recovery
*For any* invalid JSON encountered during processing, the system should skip that entry and continue with remaining entries
**Validates: Requirements 3.3**

Property 11: Output validation
*For any* completed generation, all entries in the output file should be valid JSONL format
**Validates: Requirements 3.5**

Property 12: Progress tracking updates
*For any* generation process, progress indicators should update at appropriate intervals showing current and total counts
**Validates: Requirements 4.2**

Property 13: Summary information completeness
*For any* completed generation, the summary should include total time, successful entry count, and output file path
**Validates: Requirements 4.3**

Property 14: Parameter validation error handling
*For any* invalid CLI parameters provided, the system should display helpful error messages and usage information
**Validates: Requirements 6.1**

Property 15: File overwrite handling
*For any* output file that already exists, the system should handle the overwrite operation appropriately
**Validates: Requirements 6.2**

Property 16: Malformed response recovery
*For any* malformed response from the model, the system should continue processing remaining entries without termination
**Validates: Requirements 6.4**

## Error Handling

The system implements comprehensive error handling across multiple layers:

### CLI Layer Errors
- Invalid parameter validation with helpful messages
- Missing required arguments detection
- Type conversion errors for numeric parameters

### API Communication Errors
- Connection timeout handling with retry logic
- Network connectivity issues detection
- Invalid model name error reporting
- API response format validation

### Data Processing Errors
- JSON parsing failures with graceful recovery
- Malformed response handling
- Data validation errors with detailed reporting

### File System Errors
- Permission denied error handling
- Disk space insufficient detection
- File path validation and creation
- Atomic write operations to prevent corruption

## Testing Strategy

The testing approach combines unit testing and property-based testing to ensure comprehensive coverage:

### Unit Testing
- Test specific CLI parameter combinations
- Test JSON extraction with known malformed inputs
- Test file operations with various scenarios
- Test error conditions with mocked failures
- Test progress tracking with controlled inputs

### Property-Based Testing
The system uses Hypothesis for property-based testing to verify universal properties across all valid inputs:

- **Testing Framework**: Hypothesis (Python property-based testing library)
- **Test Configuration**: Minimum 100 iterations per property test
- **Property Test Tagging**: Each test tagged with format '**Feature: ollama-cli-generator, Property {number}: {property_text}**'

Property-based tests will validate:
- Parameter handling across all valid input ranges
- JSON extraction with randomly generated malformed responses
- File operations with various filename patterns
- Error recovery with simulated failure conditions
- Data validation with randomly generated structures

### Integration Testing
- End-to-end CLI execution with real Ollama models
- File system integration with actual file operations
- Progress tracking integration with real generation processes

The dual testing approach ensures both specific edge cases are covered (unit tests) and general correctness properties hold across all inputs (property tests), providing comprehensive validation of system behavior.