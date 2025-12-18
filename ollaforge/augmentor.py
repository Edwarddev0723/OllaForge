"""
Dataset augmentation engine for OllaForge.

This module provides the core functionality for augmenting existing JSONL datasets
by modifying or adding fields using AI-generated content. It integrates with the
Ollama API for AI generation and supports concurrent processing.

Key features:
- Load and validate JSONL datasets
- Field validation (existing and new fields)
- AI-powered field augmentation
- Concurrent processing with configurable parallelism
- Preview functionality for testing augmentation
- Error handling with original entry preservation
- Progress tracking during augmentation

Requirements satisfied:
- 1.1: Load and parse JSONL files
- 1.4: Display entry count and field names
- 2.1: Accept existing field names
- 2.2: Reject non-existing fields with available field list
- 2.3: Use instruction to guide AI processing
- 2.4: Support new field creation
- 3.1: Send entry context and instruction to AI
- 3.2: Parse response and update target field
- 3.3: Preserve original entry on failure
- 3.5: Concurrent processing with configurable parallelism
- 5.1: Progress tracking during augmentation
- 6.4: Continue processing on errors
"""

import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Callable
from rich.console import Console

import ollama

from .models import (
    AugmentationConfig,
    AugmentationResult,
    OutputLanguage,
    FieldValidationError,
    validate_target_field,
)
from .file_manager import read_jsonl_file, read_dataset_file, FileOperationError
from .formats import FileFormat, FormatError
from .progress import ProgressTracker


console = Console()


def create_augmentation_prompt(
    entry: Dict[str, Any],
    target_field: str,
    instruction: str,
    context_fields: List[str],
    language: OutputLanguage
) -> Tuple[str, str]:
    """
    Create system and user prompts for augmentation.
    
    Args:
        entry: The dataset entry to augment
        target_field: The field name to augment or create
        instruction: User-provided instruction for augmentation
        context_fields: List of field names to include as context
        language: Output language for the augmentation
        
    Returns:
        Tuple of (system_prompt, user_prompt)
        
    Requirements satisfied:
    - 2.3: Include instruction in prompt
    - 3.1: Include entry context in prompt
    """
    # Build context from specified fields
    context_parts = []
    for field in context_fields:
        if field in entry:
            context_parts.append(f"{field}: {entry[field]}")
    
    context_str = "\n".join(context_parts) if context_parts else "No additional context"
    
    # Language-specific instruction
    lang_instruction = ""
    if language == OutputLanguage.ZH_TW:
        lang_instruction = "\n\n請使用繁體中文（台灣用語）回應。"
    
    system_prompt = f"""You are a data augmentation assistant. Your task is to generate or modify the "{target_field}" field based on the given instruction and context.

Instruction: {instruction}

Output ONLY the value for the "{target_field}" field. No JSON wrapping, no explanation, just the raw value.{lang_instruction}"""

    current_value = entry.get(target_field, "(empty - create new)")
    
    user_prompt = f"""Context:
{context_str}

Current value of "{target_field}": {current_value}

Generate the augmented value for "{target_field}":"""

    return system_prompt, user_prompt


class DatasetAugmentor:
    """
    Core engine for dataset augmentation.
    
    This class coordinates the entire augmentation workflow:
    - Loading and validating datasets
    - Generating augmentation prompts
    - Calling the Ollama API
    - Processing responses and updating entries
    - Handling errors gracefully
    
    Requirements satisfied:
    - 1.1, 1.4: Dataset loading and info display
    - 2.1, 2.2: Field validation
    - 2.3, 3.1: Prompt generation with context
    - 2.4, 3.2: Field augmentation and update
    - 3.3, 6.4: Error handling and preservation
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize the augmentor with configuration.
        
        Args:
            config: AugmentationConfig with all augmentation parameters
        """
        self.config = config
        self.console = Console()
    
    def load_dataset(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Load dataset and extract available fields.
        
        Supports multiple file formats: JSONL, JSON, CSV, TSV, Parquet.
        Format is automatically detected from file extension.
        
        Returns:
            Tuple of (entries, field_names)
            
        Raises:
            FileOperationError: If file cannot be read or parsed
            
        Requirements satisfied:
        - 1.1: Read and parse dataset files in multiple formats
        - 1.4: Return entry count and field names
        """
        try:
            # Try multi-format reading first
            entries, field_names = read_dataset_file(self.config.input_file)
            
            # Display format information
            from .formats import detect_format, get_format_description
            try:
                file_format = detect_format(self.config.input_file)
                format_desc = get_format_description(file_format)
                self.console.print(f"[dim]📄 Format: {format_desc}[/dim]")
            except FormatError:
                pass  # Format detection failed, but reading succeeded
            
            return entries, field_names
            
        except FileOperationError as e:
            # If multi-format reading fails, try legacy JSONL reading for backward compatibility
            if "format error" in str(e).lower():
                try:
                    self.console.print(f"[yellow]⚠️  Multi-format reading failed, trying JSONL format...[/yellow]")
                    entries, field_names = read_jsonl_file(
                        self.config.input_file, 
                        return_field_names=True
                    )
                    return entries, field_names
                except FileOperationError:
                    pass  # Fall through to original error
            
            raise  # Re-raise original error
    
    def validate_field(self, entries: List[Dict[str, Any]], field: str) -> bool:
        """
        Validate that target field exists or can be created.
        
        Args:
            entries: List of dataset entries
            field: Target field name to validate
            
        Returns:
            True if field is valid
            
        Raises:
            FieldValidationError: If field doesn't exist and create_new_field is False
            
        Requirements satisfied:
        - 2.1: Accept existing field names
        - 2.2: Reject non-existing fields with available field list
        """
        return validate_target_field(
            entries, 
            field, 
            create_new_field=self.config.create_new_field
        )
    
    def augment_entry(self, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Augment a single entry using the AI model.
        
        Args:
            entry: The dataset entry to augment
            
        Returns:
            Tuple of (augmented_entry, error_or_none)
            - On success: (modified_entry, None)
            - On failure: (original_entry, error_message)
            
        Requirements satisfied:
        - 2.4: Handle new field creation
        - 3.2: Parse response and update target field
        - 3.3: Preserve original entry on failure
        """
        # Create a deep copy to avoid modifying the original
        augmented = copy.deepcopy(entry)
        
        try:
            # Generate prompts
            system_prompt, user_prompt = create_augmentation_prompt(
                entry=entry,
                target_field=self.config.target_field,
                instruction=self.config.instruction,
                context_fields=self.config.context_fields,
                language=self.config.language
            )
            
            # Call Ollama API
            response = ollama.chat(
                model=self.config.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                }
            )
            
            # Extract response content
            if 'message' not in response or 'content' not in response['message']:
                return entry, "Invalid response format from AI model"
            
            augmented_value = response['message']['content'].strip()
            
            # Update the target field
            augmented[self.config.target_field] = augmented_value
            
            return augmented, None
            
        except Exception as e:
            # On any failure, return original entry with error message
            # Log the error for debugging purposes
            error_msg = str(e)
            self._log_error(error_msg, entry)
            return entry, error_msg
    
    def _log_error(self, error_msg: str, entry: Dict[str, Any]) -> None:
        """
        Log an augmentation error.
        
        Args:
            error_msg: The error message
            entry: The entry that failed to augment
            
        Requirements satisfied:
        - 3.3: Log failure
        - 6.4: Support error tracking for continued processing
        """
        # Get a preview of the entry for context (first 50 chars of first field)
        entry_preview = ""
        if entry:
            first_key = next(iter(entry), None)
            if first_key:
                value = str(entry[first_key])[:50]
                entry_preview = f" (entry: {first_key}={value}...)"
        
        self.console.print(
            f"[dim]⚠️  Augmentation failed{entry_preview}: {error_msg}[/dim]"
        )
    
    def augment_dataset(
        self,
        entries: List[Dict[str, Any]],
        concurrency: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> AugmentationResult:
        """
        Augment entire dataset with concurrent processing.
        
        Uses ThreadPoolExecutor for parallel processing of entries, with
        configurable concurrency level. Aggregates results from all entries
        and tracks progress during augmentation.
        
        Args:
            entries: List of dataset entries to augment
            concurrency: Number of parallel workers (default: 5)
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            AugmentationResult with statistics and augmented entries
            
        Requirements satisfied:
        - 3.5: Concurrent processing with configurable parallelism
        - 5.1: Progress tracking during augmentation
        """
        start_time = time.time()
        total_entries = len(entries)
        
        # Initialize result tracking
        augmented_entries: List[Dict[str, Any]] = [None] * total_entries  # type: ignore
        errors: List[str] = []
        success_count = 0
        failure_count = 0
        
        # Set up progress tracking
        progress_tracker = ProgressTracker(self.console)
        progress_tracker.start_progress(total_entries, "Augmenting")
        
        def process_entry(index_entry: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any], Optional[str]]:
            """Process a single entry and return (index, result, error)."""
            index, entry = index_entry
            augmented, error = self.augment_entry(entry)
            return index, augmented, error
        
        try:
            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_entry, (i, entry)): i
                    for i, entry in enumerate(entries)
                }
                
                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        index, augmented, error = future.result()
                        
                        # Store result at correct index to maintain order
                        augmented_entries[index] = augmented
                        
                        if error:
                            failure_count += 1
                            errors.append(error)
                        else:
                            success_count += 1
                        
                        # Update progress
                        progress_tracker.update_progress(1)
                        
                        # Call optional progress callback
                        if progress_callback:
                            completed = success_count + failure_count
                            progress_callback(completed, total_entries)
                            
                    except Exception as e:
                        # Handle unexpected errors from future
                        original_index = futures[future]
                        augmented_entries[original_index] = entries[original_index]
                        failure_count += 1
                        errors.append(f"Unexpected error: {str(e)}")
                        progress_tracker.update_progress(1)
        
        finally:
            # Stop progress tracking
            progress_tracker.stop_progress()
        
        duration = time.time() - start_time
        
        # Store augmented entries for later retrieval
        self._augmented_entries = augmented_entries
        
        return AugmentationResult(
            total_entries=total_entries,
            success_count=success_count,
            failure_count=failure_count,
            output_file=self.config.output_file,
            duration=duration,
            errors=errors
        )
    
    def get_augmented_entries(self) -> List[Dict[str, Any]]:
        """
        Get the augmented entries from the last augment_dataset call.
        
        Returns:
            List of augmented entries, or empty list if no augmentation has been run
        """
        return getattr(self, '_augmented_entries', [])
    
    def preview(self, entries: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Generate preview of augmentation on sample entries.
        
        Processes a configurable number of sample entries (determined by
        config.preview_count) and returns both original and augmented versions
        for comparison. This allows users to verify the AI understands their
        instructions before processing the entire dataset.
        
        Args:
            entries: List of dataset entries to sample from
            
        Returns:
            List of tuples (original_entry, augmented_entry) for each sample.
            The list length is min(len(entries), config.preview_count).
            
        Requirements satisfied:
        - 7.1: Process configurable number of sample entries (default: 3)
        - 7.2: Return original and augmented values for comparison
        """
        # Determine how many entries to preview
        preview_count = min(len(entries), self.config.preview_count)
        
        # Take the first preview_count entries as samples
        sample_entries = entries[:preview_count]
        
        # Process each sample entry and collect results
        results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        
        for entry in sample_entries:
            # Create a deep copy of the original for comparison
            original = copy.deepcopy(entry)
            
            # Augment the entry
            augmented, error = self.augment_entry(entry)
            
            # Add to results (original, augmented)
            # Note: If augmentation failed, augmented will be the same as original
            results.append((original, augmented))
        
        return results
