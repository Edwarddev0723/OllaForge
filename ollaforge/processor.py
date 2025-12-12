"""
Data processing and cleaning utilities for OllaForge.

This module handles the critical task of extracting valid JSON from AI model responses
that may contain markdown formatting, code blocks, or other noise. It provides robust
parsing, validation, and error recovery mechanisms.

Key features:
- Intelligent JSON extraction from markdown-formatted responses
- Common JSON formatting issue repair (trailing commas, quotes, etc.)
- Pydantic-based data validation for structured entries
- Graceful error handling with detailed logging
- Batch processing support for multiple responses

Requirements satisfied:
- 3.1: Extracts valid JSON from responses containing markdown or noise
- 3.3: Skips invalid entries and continues processing on JSON parsing failures
- 3.4: Uses structured validation to ensure data integrity
- 6.4: Continues processing remaining entries when malformed responses occur
"""

import json
import re
from typing import Optional, Dict, Any, List
from ollaforge.models import DataEntry


def clean_json(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract valid JSON from model responses that may contain markdown or noise.
    
    Args:
        response: Raw response from the AI model
        
    Returns:
        Parsed JSON dictionary if valid JSON found, None otherwise
    """
    if not response or not response.strip():
        return None
    
    # Remove common markdown code block markers
    cleaned = response.strip()
    
    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', cleaned, flags=re.DOTALL | re.MULTILINE)
    
    # Remove any leading/trailing whitespace and common prefixes
    cleaned = cleaned.strip()
    
    # Remove common prefixes that models might add
    prefixes_to_remove = [
        "Here's the JSON:",
        "Here is the JSON:",
        "JSON:",
        "Response:",
        "Output:",
        "Result:",
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    # Try to find JSON object in the text
    # Look for content between { and }
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = cleaned
    
    # Try to parse the JSON
    try:
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError:
        # If direct parsing fails, try to fix common issues
        try:
            # Fix common JSON issues like trailing commas, single quotes, etc.
            fixed_json = _fix_common_json_issues(json_str)
            parsed = json.loads(fixed_json)
            return parsed
        except json.JSONDecodeError:
            return None


def _fix_common_json_issues(json_str: str) -> str:
    """
    Attempt to fix common JSON formatting issues.
    
    Args:
        json_str: JSON string that may have formatting issues
        
    Returns:
        Fixed JSON string
    """
    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # Replace single quotes with double quotes (but be careful about apostrophes)
    # This is a simple approach - more sophisticated parsing might be needed
    fixed = re.sub(r"'([^']*)':", r'"\1":', fixed)  # Keys
    fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)  # Values
    
    return fixed


def validate_entry(data: Dict[str, Any]) -> Optional[DataEntry]:
    """
    Validate and convert a dictionary to a DataEntry model.
    
    Args:
        data: Dictionary containing potential DataEntry fields
        
    Returns:
        DataEntry instance if validation succeeds, None otherwise
    """
    try:
        # Check if required fields exist
        if not all(key in data for key in ['instruction', 'input', 'output']):
            return None
        
        # Create and validate DataEntry
        entry = DataEntry(
            instruction=str(data['instruction']),
            input=str(data['input']),
            output=str(data['output'])
        )
        return entry
    except Exception:
        return None


def process_model_response(response: str) -> Optional[DataEntry]:
    """
    Process a raw model response into a validated DataEntry.
    
    This function combines JSON extraction, cleaning, and validation.
    
    Args:
        response: Raw response from the AI model
        
    Returns:
        DataEntry instance if processing succeeds, None otherwise
    """
    # Extract JSON from response
    json_data = clean_json(response)
    if json_data is None:
        return None
    
    # Validate and convert to DataEntry
    entry = validate_entry(json_data)
    return entry


def process_responses(responses: List[str]) -> List[DataEntry]:
    """
    Process multiple model responses, skipping invalid entries.
    
    Args:
        responses: List of raw responses from the AI model
        
    Returns:
        List of valid DataEntry instances
    """
    valid_entries = []
    
    for response in responses:
        entry = process_model_response(response)
        if entry is not None:
            valid_entries.append(entry)
    
    return valid_entries