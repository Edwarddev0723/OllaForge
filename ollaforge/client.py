"""
Ollama client for communicating with local Ollama API.

This module provides robust communication with the local Ollama API running on localhost:11434.
It handles connection testing, model validation, prompt engineering for JSON output,
and comprehensive error handling for various failure scenarios.

Key features:
- Automatic connection testing and validation
- Optimized prompt engineering for structured JSON output
- Batch processing support with individual entry generation
- Comprehensive error handling with specific exception types
- Model availability checking and validation

Requirements satisfied:
- 2.1: Establishes connection with local Ollama API on localhost:11434
- 2.2: Uses proper prompt engineering to ensure JSON output format
- 2.3: Supports batch/single-entry generation to prevent context overflow
- 2.4: Displays clear error messages for connection failures
- 2.5: Handles API timeouts and connection errors appropriately
"""

import ollama
import json
import time
from typing import Dict, Any, Optional, List
from rich.console import Console

from .models import DataEntry

console = Console()


class OllamaConnectionError(Exception):
    """Raised when connection to Ollama API fails."""
    pass


class OllamaGenerationError(Exception):
    """Raised when generation request fails."""
    pass


def generate_data(topic: str, model: str = "llama3", count: int = 1) -> List[Dict[str, Any]]:
    """
    Generate structured data entries using Ollama API.
    
    Args:
        topic: Topic description for dataset generation
        model: Ollama model name to use
        count: Number of entries to generate in this batch
        
    Returns:
        List[Dict[str, Any]]: List of generated data entries
        
    Raises:
        OllamaConnectionError: If connection to Ollama API fails
        OllamaGenerationError: If generation request fails
    """
    try:
        # Test connection to Ollama API
        _test_ollama_connection()
        
        # Create system prompt for JSON output format
        system_prompt = _create_system_prompt(topic)
        
        # Generate data entries
        generated_entries = []
        
        for i in range(count):
            try:
                # Create user prompt for this specific entry
                user_prompt = _create_user_prompt(topic, i + 1)
                
                # Make API request to Ollama
                response = ollama.chat(
                    model=model,
                    messages=[
                        {
                            'role': 'system',
                            'content': system_prompt
                        },
                        {
                            'role': 'user', 
                            'content': user_prompt
                        }
                    ],
                    options={
                        'temperature': 0.7,
                        'top_p': 0.9,
                    }
                )
                
                # Extract content from response
                if 'message' in response and 'content' in response['message']:
                    content = response['message']['content']
                    generated_entries.append({'raw_content': content})
                else:
                    console.print(f"[yellow]⚠️  Warning: Invalid response format for entry {i + 1}[/yellow]")
                    continue
                    
            except Exception as e:
                error_msg = str(e).lower()
                # Check for critical errors that should cause immediate failure
                if "model" in error_msg and "not found" in error_msg:
                    raise OllamaGenerationError(
                        f"Model not found: {str(e)}. "
                        "Use 'ollama list' to see available models"
                    )
                elif "context length exceeded" in error_msg:
                    raise OllamaGenerationError(
                        f"Context length exceeded: {str(e)}. "
                        "Try using a shorter topic description"
                    )
                elif "model loading timeout" in error_msg:
                    raise OllamaGenerationError(
                        f"Model loading timeout: {str(e)}. "
                        "The model may be too large or the system is under load"
                    )
                else:
                    # For other errors, log and continue
                    console.print(f"[yellow]⚠️  Warning: Failed to generate entry {i + 1}: {str(e)}[/yellow]")
                    continue
        
        return generated_entries
        
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama API: {str(e)}. "
                "Make sure Ollama is running with 'ollama serve'"
            )
        elif "timeout" in error_msg:
            raise OllamaGenerationError(
                f"Request timed out: {str(e)}. "
                "Try reducing batch size or using a faster model"
            )
        elif "model" in error_msg and "not found" in error_msg:
            raise OllamaGenerationError(
                f"Model not found: {str(e)}. "
                "Use 'ollama list' to see available models"
            )
        else:
            raise OllamaGenerationError(f"Generation failed: {str(e)}")


def _test_ollama_connection() -> None:
    """
    Test connection to Ollama API on localhost:11434.
    
    Raises:
        OllamaConnectionError: If connection fails
    """
    try:
        # Try to list available models to test connection
        models = ollama.list()
        if not isinstance(models, dict):
            raise OllamaConnectionError("Invalid response from Ollama API")
            
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            raise OllamaConnectionError(
                "Cannot connect to Ollama API on localhost:11434. "
                "Please ensure Ollama is running locally with 'ollama serve'"
            )
        elif "timeout" in error_msg:
            raise OllamaConnectionError(
                "Connection to Ollama API timed out. "
                "Check if Ollama is responding properly"
            )
        else:
            raise OllamaConnectionError(f"Ollama API test failed: {str(e)}")


def _create_system_prompt(topic: str) -> str:
    """
    Create system prompt for JSON output format.
    
    Args:
        topic: Topic description for dataset generation
        
    Returns:
        str: System prompt with JSON format instructions
    """
    return f"""You are a helpful assistant that generates structured training data for machine learning.

Your task is to create realistic examples related to: {topic}

You must respond with valid JSON in exactly this format:
{{
    "instruction": "A clear instruction or task description",
    "input": "The input context or data for the task", 
    "output": "The expected output or response"
}}

Requirements:
- Generate realistic, diverse examples
- Ensure all three fields (instruction, input, output) are meaningful and related
- The JSON must be valid and parseable
- Do not include any markdown formatting or extra text
- Focus on practical, useful examples for the given topic"""


def _create_user_prompt(topic: str, entry_number: int) -> str:
    """
    Create user prompt for specific entry generation.
    
    Args:
        topic: Topic description for dataset generation
        entry_number: Current entry number being generated
        
    Returns:
        str: User prompt for this specific entry
    """
    return f"Generate example #{entry_number} for the topic: {topic}. Return only valid JSON."


def get_available_models() -> List[str]:
    """
    Get list of available Ollama models.
    
    Returns:
        List[str]: List of available model names
        
    Raises:
        OllamaConnectionError: If connection to Ollama API fails
    """
    try:
        response = ollama.list()
        if not isinstance(response, dict):
            raise OllamaConnectionError("Invalid response format from Ollama API")
            
        if 'models' not in response:
            raise OllamaConnectionError("Malformed response: missing 'models' key")
            
        models = response['models']
        if not isinstance(models, list):
            raise OllamaConnectionError("Malformed response: 'models' is not a list")
            
        model_names = []
        for model in models:
            if isinstance(model, dict) and 'name' in model:
                model_names.append(model['name'])
            # Skip malformed model entries but don't fail completely
            
        return model_names
            
    except OllamaConnectionError:
        # Re-raise our own exceptions
        raise
    except Exception as e:
        raise OllamaConnectionError(f"Failed to get available models: {str(e)}")