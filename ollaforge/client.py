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

import concurrent.futures
from typing import Any, Callable, Optional

import ollama
from rich.console import Console

from .models import DatasetType, OutputLanguage

console = Console()

# Default concurrency level for parallel requests
DEFAULT_CONCURRENCY = 5
# Maximum concurrent batch requests
MAX_CONCURRENT_BATCHES = 10
# Optimized batch size for Mac (smaller = better quality, less attention decay)
# With prompt caching, smaller batches don't hurt throughput much
DEFAULT_BATCH_SIZE = 5


# ============================================================================
# JSON Schema Definitions for Structured Output (Ollama format parameter)
# Forces model to output valid JSON, eliminating parsing errors
# ============================================================================


def _get_json_schema_for_type(dataset_type: "DatasetType", batch_size: int = 1) -> dict:
    """
    Get JSON schema for structured output based on dataset type.

    Using Ollama's 'format' parameter with JSON schema forces the model
    to only output valid JSON tokens, achieving 0% format error rate.

    Args:
        dataset_type: Type of dataset being generated
        batch_size: Number of items (for array schema)

    Returns:
        JSON schema dict for Ollama format parameter
    """
    from .models import DatasetType

    # Base schemas for each entry type
    sft_entry_schema = {
        "type": "object",
        "properties": {
            "instruction": {"type": "string"},
            "input": {"type": "string"},
            "output": {"type": "string"},
        },
        "required": ["instruction", "input", "output"],
    }

    pretrain_entry_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    conversation_message_schema = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["system", "user", "assistant"]},
            "content": {"type": "string"},
        },
        "required": ["role", "content"],
    }

    sft_conversation_entry_schema = {
        "type": "object",
        "properties": {
            "conversations": {"type": "array", "items": conversation_message_schema}
        },
        "required": ["conversations"],
    }

    dpo_entry_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "chosen": {"type": "string"},
            "rejected": {"type": "string"},
        },
        "required": ["prompt", "chosen", "rejected"],
    }

    # Select schema based on dataset type
    entry_schema = {
        DatasetType.SFT: sft_entry_schema,
        DatasetType.PRETRAIN: pretrain_entry_schema,
        DatasetType.SFT_CONVERSATION: sft_conversation_entry_schema,
        DatasetType.DPO: dpo_entry_schema,
    }.get(dataset_type, sft_entry_schema)

    # For batch generation, wrap in array
    if batch_size > 1:
        return {"type": "array", "items": entry_schema}

    return entry_schema


class OllamaConnectionError(Exception):
    """Raised when connection to Ollama API fails."""

    pass


class OllamaGenerationError(Exception):
    """Raised when generation request fails."""

    pass


def _generate_batch(
    topic: str,
    model: str,
    batch_size: int,
    batch_number: int,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
    use_structured_output: bool = True,
) -> dict[str, Any]:
    """
    Generate a batch of entries in a single API call.

    Args:
        topic: Topic for generation
        model: Ollama model name
        batch_size: Number of entries per batch
        batch_number: Batch identifier
        dataset_type: Type of dataset
        language: Output language
        use_structured_output: Use JSON schema to force valid output (recommended)

    Returns:
        Dict with 'raw_content' (JSON array string) or 'error'
    """
    try:
        system_prompt = _create_system_prompt_batch(
            topic, batch_size, dataset_type, language
        )
        user_prompt = _create_user_prompt_batch(
            topic, batch_size, batch_number, dataset_type, language
        )

        # Build request parameters
        chat_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0.8,  # Slightly higher for diversity
                "top_p": 0.9,
            },
        }

        # Add structured output (JSON schema) if enabled
        # This forces the model to only output valid JSON tokens
        if use_structured_output:
            json_schema = _get_json_schema_for_type(dataset_type, batch_size)
            chat_params["format"] = json_schema

        response = ollama.chat(**chat_params)

        if "message" in response and "content" in response["message"]:
            return {
                "raw_content": response["message"]["content"],
                "batch_number": batch_number,
                "is_batch": True,
                "dataset_type": dataset_type.value,
            }
        else:
            return {"error": "Invalid response format", "batch_number": batch_number}

    except Exception as e:
        return {"error": str(e), "batch_number": batch_number}


def _generate_single_entry(
    topic: str,
    model: str,
    entry_number: int,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
) -> dict[str, Any]:
    """Generate a single data entry."""
    try:
        system_prompt = _create_system_prompt_single(topic, dataset_type, language)
        user_prompt = _create_user_prompt(topic, entry_number, dataset_type, language)

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

        if "message" in response and "content" in response["message"]:
            return {
                "raw_content": response["message"]["content"],
                "entry_number": entry_number,
                "dataset_type": dataset_type.value,
            }
        else:
            return {"error": "Invalid response format", "entry_number": entry_number}

    except Exception as e:
        error_msg = str(e).lower()
        if "model" in error_msg and "not found" in error_msg:
            raise OllamaGenerationError("Model not found") from e
        elif "model loading timeout" in error_msg:
            raise OllamaGenerationError("Model loading timeout") from e
        elif "context length exceeded" in error_msg:
            raise OllamaGenerationError("Context length exceeded") from e
        elif "connection" in error_msg or "refused" in error_msg:
            raise OllamaConnectionError(f"Failed to connect: {str(e)}") from e
        else:
            return {"error": str(e), "entry_number": entry_number}


def generate_data_batch(
    topic: str,
    model: str,
    batch_size: int,
    batch_number: int,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
) -> list[dict[str, Any]]:
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
    result = _generate_batch(
        topic, model, batch_size, batch_number, dataset_type, language
    )

    if "error" in result:
        error_msg = result["error"].lower()
        if "model" in error_msg and "not found" in error_msg:
            raise OllamaGenerationError(f"Model not found: {result['error']}")
        return []

    # Return the batch response for processing
    return [result]


def generate_data(
    topic: str,
    model: str = "gpt-oss:20b",
    count: int = 1,
    concurrency: int = DEFAULT_CONCURRENCY,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
) -> list[dict[str, Any]]:
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
                result = _generate_single_entry(
                    topic, model, i + 1, dataset_type, language
                )
                if "raw_content" in result:
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
            raise OllamaConnectionError(f"Failed to connect: {str(e)}") from e
        elif "model" in error_msg and "not found" in error_msg:
            raise OllamaGenerationError("Model not found") from e
        elif "model loading timeout" in error_msg:
            raise OllamaGenerationError("Model loading timeout") from e
        elif "context length exceeded" in error_msg:
            raise OllamaGenerationError("Context length exceeded") from e
        else:
            raise OllamaGenerationError(f"Generation failed: {str(e)}") from e


def generate_data_concurrent(
    topic: str,
    model: str,
    total_count: int,
    batch_size: int = DEFAULT_BATCH_SIZE,  # Optimized: 5 instead of 10
    max_concurrent: int = MAX_CONCURRENT_BATCHES,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_structured_output: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate data using concurrent batch requests (funnel architecture).

    Sends multiple batch requests in parallel to maximize GPU utilization.
    This is the core of the "funnel" approach - over-request and filter.

    Performance optimizations for Mac:
    - Smaller batch_size (5) reduces attention decay, improves quality
    - JSON schema structured output eliminates format errors (0% error rate)
    - BERT QC runs on CPU to keep MPS free for LLM

    Args:
        topic: Topic description for dataset generation
        model: Ollama model name to use
        total_count: Total number of entries to generate
        batch_size: Number of entries per batch (default 5 for quality)
        max_concurrent: Maximum number of concurrent batch requests
        dataset_type: Type of dataset to generate
        language: Output language for generated content
        progress_callback: Optional callback(completed_batches, total_batches)
        use_structured_output: Use JSON schema for guaranteed valid output

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

    def generate_batch_wrapper(batch_num: int) -> dict[str, Any]:
        """Wrapper for batch generation with batch number."""
        # Calculate entries for this batch
        start_idx = batch_num * batch_size
        remaining = total_count - start_idx
        current_batch_size = min(batch_size, remaining)

        return _generate_batch(
            topic,
            model,
            current_batch_size,
            batch_num + 1,
            dataset_type,
            language,
            use_structured_output,
        )

    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=actual_concurrent
    ) as executor:
        # Submit all batch requests
        future_to_batch = {
            executor.submit(generate_batch_wrapper, i): i for i in range(num_batches)
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                result = future.result()
                if "raw_content" in result:
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
        # Handle both dict (older versions) and ListResponse (newer versions)
        if not (isinstance(models, dict) or hasattr(models, 'models')):
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
            raise OllamaConnectionError(f"Ollama API test failed: {str(e)}") from e


def _get_language_instruction(language: OutputLanguage) -> str:
    """Get language-specific instruction for prompts."""
    if language == OutputLanguage.ZH_TW:
        return """

【語言規範】使用台灣繁體中文，禁止大陸用語：
視頻→影片、軟件→軟體、交互→互動、質量→品質、信息→資訊、數據→資料、網絡→網路、程序→程式、服務器→伺服器、用戶→使用者、優化→最佳化、默認→預設

【正確範例】
❌「這個軟件的質量很好」→ ✅「這個軟體的品質很好」
❌「用戶可以通過網絡下載」→ ✅「使用者可以透過網路下載」"""

    elif language == OutputLanguage.ZH_CN:
        return """

【语言规范】使用中国大陆简体中文，符合大陆用语习惯：
影片→视频、軟體→软件、互動→交互、品質→质量、資訊→信息、資料→数据、網路→网络、程式→程序、伺服器→服务器、使用者→用户、最佳化→优化、預設→默认

【文化与价值观规范】
1. 弘扬社会主义核心价值观：富强、民主、文明、和谐、自由、平等、公正、法治、爱国、敬业、诚信、友善
2. 尊重中华优秀传统文化，体现文化自信
3. 使用符合中国国情的表述方式
4. 涉及历史、政治、领土等敏感话题时，须符合中国官方立场
5. 货币使用人民币（元/￥），计量单位使用公制

【正确范例】
❌「這個軟體的品質很好」→ ✅「这个软件的质量很好」
❌「使用者可以透過網路下載」→ ✅「用户可以通过网络下载」
❌「台灣是一個國家」→ ✅「台湾是中国的一部分」"""
    return ""


def _create_system_prompt_single(
    topic: str,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
) -> str:
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


def _create_system_prompt_batch(
    topic: str,
    batch_size: int,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
) -> str:
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

【CRITICAL REQUIREMENTS】

1. UNIFIED SYSTEM PROMPT: Use this EXACT system prompt for ALL conversations:
   "你是一位專業的助理，負責協助使用者完成「{topic}」相關的任務。請用自然、友善的語氣回應。"

2. USER INPUT DIVERSITY: Each user must speak DIFFERENTLY:
   ✗ AVOID repeating: "我要一杯...", "請給我..."
   ✓ VARY patterns: "來杯...", "幫我弄個...", "欸那個...", "有沒有...", "想點..."

3. 【ASSISTANT RESPONSE DIVERSITY - CRITICAL】
   Assistant must NOT always use the same confirmation pattern!

   ✗ ROBOTIC (avoid): "好的，已為您準備一杯冰拿鐵。" (every time)

   ✓ NATURAL VARIATIONS (mix these styles):
   - Casual: "沒問題～" / "OK！" / "好喔～"
   - Friendly: "馬上來！" / "稍等一下喔～"
   - Efficient: "冰拿鐵一杯。" (short, no fluff)
   - Playful: "冰拿鐵 get！還要什麼嗎？"
   - Professional: "為您準備冰拿鐵，請稍候。"

   ✗ VERBOSE (avoid): "好的，您點的是一杯冰的拿鐵咖啡，不加濃縮咖啡，總共一杯，請問還需要其他的嗎？"
   ✓ CONCISE: "冰拿鐵～還要別的嗎？"

4. FLEXIBLE RULES: The system should handle creative requests gracefully:
   - Unknown items: "抱歉我們沒有這個，要不要試試...？"
   - Special requests: Accept reasonable customizations, politely decline impossible ones
   - Don't rigidly reject - offer alternatives

5. MULTI-TURN SCENARIOS (distribute across {batch_size} examples):
   - 30% Clarification: User vague → Assistant asks briefly
   - 20% Modification: User changes mind
   - 20% Error/alternatives: Item unavailable → suggest similar
   - 20% Complex: Multiple items, combos
   - 10% Edge: Cancel, complaints, chitchat

6. LOGICAL ACCURACY: Quantities must match. No hallucinated items.

7. NO ARTIFACTS: Never include XML tags or metadata in content.

Example (note the VARIED assistant responses):
[
  {{"conversations": [
    {{"role": "system", "content": "你是一位專業的助理..."}},
    {{"role": "user", "content": "欸有賣咖啡嗎"}},
    {{"role": "assistant", "content": "有喔！美式跟拿鐵，要哪個？"}},
    {{"role": "user", "content": "拿鐵 冰的"}},
    {{"role": "assistant", "content": "冰拿鐵～還要嗎？"}}
  ]}},
  {{"conversations": [
    {{"role": "system", "content": "你是一位專業的助理..."}},
    {{"role": "user", "content": "一杯熱美式"}},
    {{"role": "assistant", "content": "熱美式 get！"}}
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


def _create_user_prompt(
    topic: str,
    entry_number: int,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
) -> str:
    """Create user prompt for single entry generation."""
    type_hint = {
        DatasetType.SFT: "instruction/input/output",
        DatasetType.PRETRAIN: "text",
        DatasetType.SFT_CONVERSATION: "conversation",
        DatasetType.DPO: "prompt/chosen/rejected",
    }.get(dataset_type, "")

    lang_hint = (
        " 請用繁體中文（台灣用語）回答。" if language == OutputLanguage.ZH_TW else ""
    )
    return f"Generate {type_hint} example #{entry_number} for: {topic}. JSON only.{lang_hint}"


def _create_user_prompt_batch(
    topic: str,
    batch_size: int,
    batch_number: int,
    dataset_type: DatasetType = DatasetType.SFT,
    language: OutputLanguage = OutputLanguage.EN,
) -> str:
    """Create user prompt for batch generation."""
    type_hint = {
        DatasetType.SFT: "instruction/input/output",
        DatasetType.PRETRAIN: "text",
        DatasetType.SFT_CONVERSATION: "conversation",
        DatasetType.DPO: "prompt/chosen/rejected",
    }.get(dataset_type, "")

    lang_hint = (
        " 請用繁體中文（台灣用語）回答。" if language == OutputLanguage.ZH_TW else ""
    )
    return f"Generate {batch_size} unique {type_hint} examples (batch {batch_number}) for: {topic}. JSON array only.{lang_hint}"


def get_available_models() -> list[str]:
    """
    Get list of available Ollama models.

    Returns:
        List[str]: List of available model names

    Raises:
        OllamaConnectionError: If connection to Ollama API fails
    """
    try:
        response = ollama.list()
        
        # Handle both dict (older versions) and ListResponse (newer versions)
        if hasattr(response, 'models'):
            # Newer version returns ListResponse object
            models = response.models
            model_names = []
            for model in models:
                if hasattr(model, 'model'):
                    model_names.append(model.model)
                elif isinstance(model, dict) and "name" in model:
                    model_names.append(model["name"])
            return model_names
        elif isinstance(response, dict):
            # Older version returns dict
            if "models" not in response:
                raise OllamaConnectionError("Malformed response: missing 'models' key")

            models = response["models"]
            if not isinstance(models, list):
                raise OllamaConnectionError("Malformed response: 'models' is not a list")

            model_names = []
            for model in models:
                if isinstance(model, dict) and "name" in model:
                    model_names.append(model["name"])
                # Skip malformed model entries but don't fail completely

            return model_names
        else:
            raise OllamaConnectionError("Invalid response format from Ollama API")

    except OllamaConnectionError:
        # Re-raise our own exceptions
        raise
    except Exception as e:
        raise OllamaConnectionError(f"Failed to get available models: {str(e)}") from e
