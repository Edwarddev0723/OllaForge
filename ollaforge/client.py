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
- Concurrent request processing for improved performance

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
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional, List, Callable
from rich.console import Console

from .models import DataEntry, DatasetType, OutputLanguage

console = Console()

# Default concurrency level for parallel requests
DEFAULT_CONCURRENCY = 5
# Maximum concurrent batch requests
MAX_CONCURRENT_BATCHES = 10


class OllamaConnectionError(Exception):
    """Raised when connection to Ollama API fails."""
    pass


class OllamaGenerationError(Exception):
    """Raised when generation request fails."""
    pass


def _generate_batch(topic: str, model: str, batch_size: int, batch_number: int, 
                    dataset_type: DatasetType = DatasetType.SFT,
                    language: OutputLanguage = OutputLanguage.EN) -> Dict[str, Any]:
    """
    Generate a batch of entries in a single API call.
    
    Returns:
        Dict with 'raw_content' (JSON array string) or 'error'
    """
    try:
        system_prompt = _create_system_prompt_batch(topic, batch_size, dataset_type, language)
        user_prompt = _create_user_prompt_batch(topic, batch_size, batch_number, dataset_type, language)
        
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                'temperature': 0.8,  # Slightly higher for diversity
                'top_p': 0.9,
            }
        )
        
        if 'message' in response and 'content' in response['message']:
            return {
                'raw_content': response['message']['content'], 
                'batch_number': batch_number, 
                'is_batch': True,
                'dataset_type': dataset_type.value
            }
        else:
            return {'error': 'Invalid response format', 'batch_number': batch_number}
            
    except Exception as e:
        return {'error': str(e), 'batch_number': batch_number}


def _generate_single_entry(topic: str, model: str, entry_number: int,
                           dataset_type: DatasetType = DatasetType.SFT,
                           language: OutputLanguage = OutputLanguage.EN) -> Dict[str, Any]:
    """Generate a single data entry."""
    try:
        system_prompt = _create_system_prompt_single(topic, dataset_type, language)
        user_prompt = _create_user_prompt(topic, entry_number, dataset_type, language)
        
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                'temperature': 0.7,
                'top_p': 0.9,
            }
        )
        
        if 'message' in response and 'content' in response['message']:
            return {
                'raw_content': response['message']['content'], 
                'entry_number': entry_number,
                'dataset_type': dataset_type.value
            }
        else:
            return {'error': 'Invalid response format', 'entry_number': entry_number}
            
    except Exception as e:
        return {'error': str(e), 'entry_number': entry_number}


def generate_data_batch(topic: str, model: str, batch_size: int, batch_number: int,
                        dataset_type: DatasetType = DatasetType.SFT,
                        language: OutputLanguage = OutputLanguage.EN) -> List[Dict[str, Any]]:
    """
    Generate multiple entries in a single API call (batch mode).
    
    Args:
        topic: Topic description
        model: Ollama model name
        batch_size: Number of entries to generate in this batch
        batch_number: Batch identifier for the prompt
        dataset_type: Type of dataset to generate
        language: Output language for generated content
        
    Returns:
        List of dicts with 'raw_content' for each entry
    """
    result = _generate_batch(topic, model, batch_size, batch_number, dataset_type, language)
    
    if 'error' in result:
        error_msg = result['error'].lower()
        if "model" in error_msg and "not found" in error_msg:
            raise OllamaGenerationError(f"Model not found: {result['error']}")
        return []
    
    # Return the batch response for processing
    return [result]


def generate_data(topic: str, model: str = "gpt-oss:20b", count: int = 1, 
                  concurrency: int = DEFAULT_CONCURRENCY,
                  dataset_type: DatasetType = DatasetType.SFT,
                  language: OutputLanguage = OutputLanguage.EN) -> List[Dict[str, Any]]:
    """
    Generate structured data entries using Ollama API.
    Uses batch generation for efficiency.
    
    Args:
        topic: Topic description for dataset generation
        model: Ollama model name to use
        count: Number of entries to generate
        concurrency: Number of concurrent requests (for fallback single mode)
        dataset_type: Type of dataset to generate (SFT, PRETRAIN, SFT_CONVERSATION, DPO)
        language: Output language for generated content (EN, ZH_TW)
        
    Returns:
        List[Dict[str, Any]]: List of generated data entries
    """
    try:
        _test_ollama_connection()
        
        # For small counts, use single generation
        if count <= 3:
            generated_entries = []
            for i in range(count):
                result = _generate_single_entry(topic, model, i + 1, dataset_type, language)
                if 'raw_content' in result:
                    generated_entries.append(result)
            return generated_entries
        
        # For larger counts, use batch generation
        # Each API call generates multiple entries
        return generate_data_batch(topic, model, count, 1, dataset_type, language)
        
    except OllamaGenerationError:
        raise
    except OllamaConnectionError:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            raise OllamaConnectionError(f"Failed to connect: {str(e)}")
        elif "model" in error_msg and "not found" in error_msg:
            raise OllamaGenerationError(f"Model not found: {str(e)}")
        else:
            raise OllamaGenerationError(f"Generation failed: {str(e)}")


def generate_data_concurrent(
    topic: str, 
    model: str, 
    total_count: int,
    batch_size: int = 10,
    max_concurrent: int = MAX_CONCURRENT_BATCHES,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, Any]]:
    """
    Generate data using concurrent batch requests (funnel architecture).
    
    Sends multiple batch requests in parallel to maximize GPU utilization.
    This is the core of the "funnel" approach - over-request and filter.
    
    Args:
        topic: Topic description for dataset generation
        model: Ollama model name to use
        total_count: Total number of entries to generate
        batch_size: Number of entries per batch request
        max_concurrent: Maximum number of concurrent batch requests
        dataset_type: Type of dataset to generate
        language: Output language for generated content
        progress_callback: Optional callback(completed_batches, total_batches)
        
    Returns:
        List of raw response dicts with 'raw_content'
    """
    _test_ollama_connection()
    
    # Calculate number of batches needed
    num_batches = (total_count + batch_size - 1) // batch_size
    
    # Limit concurrent requests
    actual_concurrent = min(max_concurrent, num_batches)
    
    all_responses = []
    completed_batches = 0
    
    def generate_batch_wrapper(batch_num: int) -> Dict[str, Any]:
        """Wrapper for batch generation with batch number."""
        # Calculate entries for this batch
        start_idx = batch_num * batch_size
        remaining = total_count - start_idx
        current_batch_size = min(batch_size, remaining)
        
        return _generate_batch(
            topic, model, current_batch_size, batch_num + 1, 
            dataset_type, language
        )
    
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_concurrent) as executor:
        # Submit all batch requests
        future_to_batch = {
            executor.submit(generate_batch_wrapper, i): i 
            for i in range(num_batches)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                result = future.result()
                if 'raw_content' in result:
                    all_responses.append(result)
            except Exception as e:
                # Log error but continue with other batches
                console.print(f"[dim]Batch {batch_num + 1} failed: {e}[/dim]")
            
            completed_batches += 1
            if progress_callback:
                progress_callback(completed_batches, num_batches)
    
    return all_responses


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


def _get_language_instruction(language: OutputLanguage) -> str:
    """Get language-specific instruction for prompts."""
    if language == OutputLanguage.ZH_TW:
        return """

【語言規範】使用台灣繁體中文，禁止大陸用語：
視頻→影片、軟件→軟體、交互→互動、質量→品質、信息→資訊、數據→資料、網絡→網路、程序→程式、服務器→伺服器、用戶→使用者、優化→最佳化、默認→預設

【正確範例】
❌「這個軟件的質量很好」→ ✅「這個軟體的品質很好」
❌「用戶可以通過網絡下載」→ ✅「使用者可以透過網路下載」"""
    return ""


def _create_system_prompt_single(topic: str, dataset_type: DatasetType = DatasetType.SFT,
                                  language: OutputLanguage = OutputLanguage.EN) -> str:
    """Create system prompt for single JSON output based on dataset type."""
    
    lang_instruction = _get_language_instruction(language)
    
    if dataset_type == DatasetType.SFT:
        return f"""You are a data generator. Generate training data for: {topic}

Output ONLY valid JSON (Alpaca/SFT format):
{{"instruction": "task description", "input": "input context", "output": "expected output"}}

No markdown, no explanation, just JSON.{lang_instruction}"""

    elif dataset_type == DatasetType.PRETRAIN:
        return f"""You are a data generator. Generate pre-training text data for: {topic}

Output ONLY valid JSON:
{{"text": "A complete, coherent paragraph or document about the topic..."}}

The text should be informative, well-written, and suitable for language model pre-training.
No markdown, no explanation, just JSON.{lang_instruction}"""

    elif dataset_type == DatasetType.SFT_CONVERSATION:
        return f"""You are a data generator creating DIVERSE multi-turn conversations for: {topic}

Output ONLY valid JSON (ShareGPT/ChatML format):
{{"conversations": [
  {{"role": "system", "content": "system prompt"}},
  {{"role": "user", "content": "user message"}},
  {{"role": "assistant", "content": "assistant response"}}
]}}

【CRITICAL REQUIREMENTS】
1. DIVERSITY: Each conversation must use DIFFERENT sentence patterns, vocabulary, and scenarios
2. CONSISTENT SYSTEM PROMPT: Use the SAME system prompt for all conversations in this batch
3. REALISTIC DIALOGUE: Users speak naturally with typos, incomplete sentences, or casual tone
4. MULTI-TURN DEPTH (2-4 turns): Include scenarios like:
   - Clarification questions (user is vague, assistant asks for details)
   - Error handling (item unavailable, invalid request)
   - Modifications (user changes mind mid-conversation)
   - Edge cases (unusual requests, boundary conditions)
5. LOGICAL ACCURACY: Assistant responses must be mathematically and logically correct
6. NO ARTIFACTS: Do not include any XML tags, source markers, or metadata in content

No markdown, no explanation, just JSON.{lang_instruction}"""

    elif dataset_type == DatasetType.DPO:
        return f"""You are a data generator. Generate preference data for DPO training on: {topic}

Output ONLY valid JSON (DPO format):
{{"prompt": "the question or instruction", "chosen": "the better/preferred response", "rejected": "the worse/less preferred response"}}

The "chosen" response should be clearly better than "rejected" in terms of:
- Accuracy, helpfulness, safety, or quality
No markdown, no explanation, just JSON.{lang_instruction}"""

    return _create_system_prompt_single(topic, DatasetType.SFT, language)


def _create_system_prompt_batch(topic: str, batch_size: int, 
                                 dataset_type: DatasetType = DatasetType.SFT,
                                 language: OutputLanguage = OutputLanguage.EN) -> str:
    """Create system prompt for batch JSON output based on dataset type."""
    
    lang_instruction = _get_language_instruction(language)
    
    if dataset_type == DatasetType.SFT:
        return f"""You are a data generator. Generate {batch_size} DIFFERENT training examples for: {topic}

Output a JSON array with {batch_size} objects (Alpaca/SFT format).
Each object has: instruction, input, output.

Example format:
[
  {{"instruction": "task1", "input": "context1", "output": "output1"}},
  {{"instruction": "task2", "input": "context2", "output": "output2"}}
]

IMPORTANT:
- Output ONLY the JSON array, no markdown, no explanation
- Each example must be UNIQUE and DIFFERENT
- Generate exactly {batch_size} examples{lang_instruction}"""

    elif dataset_type == DatasetType.PRETRAIN:
        return f"""You are a data generator. Generate {batch_size} DIFFERENT pre-training text samples for: {topic}

Output a JSON array with {batch_size} objects.
Each object has: text (a complete paragraph or document).

Example format:
[
  {{"text": "First informative paragraph about the topic..."}},
  {{"text": "Second different paragraph about another aspect..."}}
]

IMPORTANT:
- Output ONLY the JSON array, no markdown, no explanation
- Each text must be UNIQUE, informative, and well-written
- Generate exactly {batch_size} examples{lang_instruction}"""

    elif dataset_type == DatasetType.SFT_CONVERSATION:
        return f"""You are a data generator. Generate {batch_size} DIVERSE multi-turn conversations for: {topic}

Output a JSON array with {batch_size} objects (ShareGPT/ChatML format).

【CRITICAL REQUIREMENTS - READ CAREFULLY】

1. UNIFIED SYSTEM PROMPT: Use this EXACT system prompt for ALL conversations:
   "你是一位專業的助理，負責協助使用者完成「{topic}」相關的任務。請用自然、友善的語氣回應。"

2. SENTENCE DIVERSITY: Each user input must use DIFFERENT patterns:
   ✗ AVOID: "我要一杯...", "請給我..." (repeating same pattern)
   ✓ USE VARIED PATTERNS: "來杯...", "幫我弄個...", "可以給我...", "想要...", "有沒有...", "欸，那個..."

3. REALISTIC USER BEHAVIOR: Users are NOT perfect:
   - Casual/incomplete: "呃...那個拿鐵" instead of "請給我一杯拿鐵"
   - Typos/colloquial: "ㄋㄟㄋㄟ" (milk), "冰ㄉ" (iced)
   - Vague requests: "我要咖啡" (assistant should ask: 美式還是拿鐵？)

4. MULTI-TURN SCENARIOS (distribute across {batch_size} examples):
   - 30% Clarification: User is vague → Assistant asks for details
   - 20% Modification: User changes order mid-conversation
   - 20% Error handling: Item unavailable, invalid combo
   - 20% Complex orders: Multiple items, special requests
   - 10% Edge cases: Cancel, complaints, unusual requests

5. LOGICAL ACCURACY: 
   ✗ WRONG: User orders 1 item → Assistant says "總共兩杯"
   ✓ CORRECT: Quantities must match exactly

6. NO ARTIFACTS: Never include <source>, XML tags, or metadata in JSON content

Example format:
[
  {{"conversations": [
    {{"role": "system", "content": "你是一位專業的助理..."}},
    {{"role": "user", "content": "欸，有賣咖啡嗎"}},
    {{"role": "assistant", "content": "有的！我們有美式和拿鐵，請問您想要哪一種呢？"}},
    {{"role": "user", "content": "拿鐵好了，冰的"}},
    {{"role": "assistant", "content": "好的，一杯冰拿鐵，還需要其他的嗎？"}}
  ]}}
]

Output ONLY the JSON array. Generate exactly {batch_size} UNIQUE examples.{lang_instruction}"""

    elif dataset_type == DatasetType.DPO:
        return f"""You are a data generator. Generate {batch_size} DIFFERENT preference pairs for DPO training on: {topic}

Output a JSON array with {batch_size} objects (DPO format).
Each object has: prompt, chosen (better response), rejected (worse response).

Example format:
[
  {{"prompt": "question1", "chosen": "good answer1", "rejected": "bad answer1"}},
  {{"prompt": "question2", "chosen": "good answer2", "rejected": "bad answer2"}}
]

IMPORTANT:
- Output ONLY the JSON array, no markdown, no explanation
- "chosen" must be clearly BETTER than "rejected"
- Generate exactly {batch_size} examples{lang_instruction}"""

    return _create_system_prompt_batch(topic, batch_size, DatasetType.SFT, language)


def _create_user_prompt(topic: str, entry_number: int, 
                        dataset_type: DatasetType = DatasetType.SFT,
                        language: OutputLanguage = OutputLanguage.EN) -> str:
    """Create user prompt for single entry generation."""
    type_hint = {
        DatasetType.SFT: "instruction/input/output",
        DatasetType.PRETRAIN: "text",
        DatasetType.SFT_CONVERSATION: "conversation",
        DatasetType.DPO: "prompt/chosen/rejected"
    }.get(dataset_type, "")
    
    lang_hint = " 請用繁體中文（台灣用語）回答。" if language == OutputLanguage.ZH_TW else ""
    return f"Generate {type_hint} example #{entry_number} for: {topic}. JSON only.{lang_hint}"


def _create_user_prompt_batch(topic: str, batch_size: int, batch_number: int,
                               dataset_type: DatasetType = DatasetType.SFT,
                               language: OutputLanguage = OutputLanguage.EN) -> str:
    """Create user prompt for batch generation."""
    type_hint = {
        DatasetType.SFT: "instruction/input/output",
        DatasetType.PRETRAIN: "text",
        DatasetType.SFT_CONVERSATION: "conversation",
        DatasetType.DPO: "prompt/chosen/rejected"
    }.get(dataset_type, "")
    
    lang_hint = " 請用繁體中文（台灣用語）回答。" if language == OutputLanguage.ZH_TW else ""
    return f"Generate {batch_size} unique {type_hint} examples (batch {batch_number}) for: {topic}. JSON array only.{lang_hint}"


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