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
from typing import Optional, Dict, Any, List, Union
from ollaforge.models import (
    DataEntry, PretrainEntry, SFTConversationEntry, DPOEntry, 
    ConversationMessage, DatasetType, DatasetEntry
)


def _clean_response(response: str) -> str:
    """Remove markdown and common prefixes from response."""
    if not response:
        return ""
    
    cleaned = response.strip()
    
    # Remove markdown code blocks - be more careful about nested backticks
    # Only remove if we have proper opening and closing code blocks
    if cleaned.startswith('```') and cleaned.endswith('```'):
        # Find the first newline after opening backticks
        first_newline = cleaned.find('\n')
        if first_newline != -1:
            # Remove opening ```json\n and closing ```
            cleaned = cleaned[first_newline+1:-3].strip()
        else:
            # No newline, just remove the backticks
            cleaned = cleaned[3:-3].strip()
    elif '```json\n' in cleaned and cleaned.endswith('```'):
        # Handle cases where there's text before the code block
        start_idx = cleaned.find('```json\n') + 8
        cleaned = cleaned[start_idx:-3].strip()
    elif '```\n' in cleaned and cleaned.endswith('```'):
        # Handle cases with generic code blocks
        start_idx = cleaned.find('```\n') + 4
        cleaned = cleaned[start_idx:-3].strip()
    
    cleaned = cleaned.strip()
    
    # Remove common prefixes
    for prefix in ["Here's the JSON:", "Here is the JSON:", "JSON:", "Response:", "Output:", "Result:"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    return cleaned


def clean_json(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract valid JSON object from model response.
    
    Returns:
        Parsed JSON dictionary if valid JSON found, None otherwise
    """
    if not response or not response.strip():
        return None
    
    cleaned = _clean_response(response)
    
    # Try to find JSON object
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = cleaned
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return json.loads(_fix_common_json_issues(json_str))
        except json.JSONDecodeError:
            return None


def clean_json_array(response: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract valid JSON array from model response (for batch generation).
    
    Returns:
        List of parsed JSON dictionaries, or None if parsing fails
    """
    if not response or not response.strip():
        return None
    
    cleaned = _clean_response(response)
    
    # Try to find JSON array
    array_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if array_match:
        json_str = array_match.group(0)
    else:
        json_str = cleaned
    
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return parsed
        return None
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_fix_common_json_issues(json_str))
            if isinstance(parsed, list):
                return parsed
            return None
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


def validate_entry(data: Dict[str, Any], dataset_type: DatasetType = DatasetType.SFT) -> Optional[DatasetEntry]:
    """
    Validate and convert a dictionary to the appropriate entry model based on dataset type.
    
    Args:
        data: Dictionary containing potential entry fields
        dataset_type: Type of dataset being generated
        
    Returns:
        Validated entry instance if validation succeeds, None otherwise
    """
    try:
        if dataset_type == DatasetType.SFT:
            # Alpaca/SFT format: instruction, input, output
            if not all(key in data for key in ['instruction', 'input', 'output']):
                return None
            return DataEntry(
                instruction=str(data['instruction']),
                input=str(data['input']),
                output=str(data['output'])
            )
        
        elif dataset_type == DatasetType.PRETRAIN:
            # Pre-training format: text
            if 'text' not in data:
                return None
            return PretrainEntry(text=str(data['text']))
        
        elif dataset_type == DatasetType.SFT_CONVERSATION:
            # Conversation format: conversations array
            if 'conversations' not in data or not isinstance(data['conversations'], list):
                return None
            
            messages = []
            for msg in data['conversations']:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    role = msg['role']
                    if role in ['system', 'user', 'assistant']:
                        messages.append(ConversationMessage(
                            role=role,
                            content=str(msg['content'])
                        ))
            
            if not messages:
                return None
            return SFTConversationEntry(conversations=messages)
        
        elif dataset_type == DatasetType.DPO:
            # DPO format: prompt, chosen, rejected
            if not all(key in data for key in ['prompt', 'chosen', 'rejected']):
                return None
            return DPOEntry(
                prompt=str(data['prompt']),
                chosen=str(data['chosen']),
                rejected=str(data['rejected'])
            )
        
        return None
    except Exception:
        return None


def process_model_response(response: str, is_batch: bool = False, 
                           dataset_type: DatasetType = DatasetType.SFT) -> Union[DatasetEntry, List[DatasetEntry], None]:
    """
    Process a raw model response into validated entry objects.
    
    Args:
        response: Raw response from the AI model
        is_batch: If True, expect JSON array; if False, expect single JSON object
        dataset_type: Type of dataset being generated
        
    Returns:
        For single entries (is_batch=False): DatasetEntry or None if processing fails
        For batch entries (is_batch=True): List of validated entry instances (empty list if processing fails)
    """
    if is_batch:
        # Parse as JSON array
        json_array = clean_json_array(response)
        if json_array is None:
            # Fallback: try parsing as single object
            json_data = clean_json(response)
            if json_data:
                entry = validate_entry(json_data, dataset_type)
                return [entry] if entry else []
            return []
        
        # Validate each entry in the array
        valid_entries = []
        for item in json_array:
            if isinstance(item, dict):
                entry = validate_entry(item, dataset_type)
                if entry:
                    valid_entries.append(entry)
        return valid_entries
    else:
        # Parse as single JSON object
        json_data = clean_json(response)
        if json_data is None:
            return None
        entry = validate_entry(json_data, dataset_type)
        return entry


def process_responses(responses: List[str], 
                      dataset_type: DatasetType = DatasetType.SFT) -> List[DatasetEntry]:
    """
    Process multiple model responses, skipping invalid entries.
    
    Args:
        responses: List of raw responses from the AI model
        dataset_type: Type of dataset being generated
        
    Returns:
        List of valid entry instances
    """
    valid_entries = []
    
    for response in responses:
        entries = process_model_response(response, is_batch=True, dataset_type=dataset_type)
        if isinstance(entries, list):
            valid_entries.extend(entries)
        elif entries is not None:
            valid_entries.append(entries)
    
    return valid_entries