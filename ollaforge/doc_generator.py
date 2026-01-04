"""
Document dataset generator module for OllaForge.

This module provides functionality to generate fine-tuning datasets from
document chunks using Ollama LLM. It supports multiple output formats
including SFT, Pre-training, Conversation, and DPO.

Key features:
- Generate training data from document chunks
- Support for multiple dataset formats (SFT, Pre-training, Conversation, DPO)
- Integration with Ollama API for content generation
- Output validation for each format
- Quality control filtering

Requirements satisfied:
- 3.1: SFT format with instruction, input, output fields
- 3.2: Pre-training format with text field
- 3.3: Conversation format with conversations array
- 3.4: DPO format with prompt, chosen, rejected fields
- 3.5: Use Ollama model to generate content based on document context
- 3.6: Serialization round-trip property
- 7.1: Validate generated entries match expected format schema
- 7.2: Ensure SFT instruction, input, output are non-empty
- 7.3: Ensure Conversation has user and assistant messages
- 7.4: Ensure DPO chosen and rejected are different
- 7.5: QC flag for quality control filtering
- 7.6: Taiwan Chinese validation when QC enabled for zh-tw
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

import ollama

from .chunk_splitter import TextChunk
from .models import (
    ConversationMessage,
    DataEntry,
    DatasetEntry,
    DatasetType,
    DPOEntry,
    OutputLanguage,
    PretrainEntry,
    SFTConversationEntry,
)
from .qc import QualityController


@dataclass
class DocGenerationConfig:
    """
    Configuration for document-to-dataset generation.

    Attributes:
        dataset_type: Type of dataset to generate (SFT, PRETRAIN, SFT_CONVERSATION, DPO)
        model: Ollama model name to use for generation
        language: Output language for generated content
        entries_per_chunk: Number of dataset entries to generate per document chunk
        qc_enabled: Whether to enable quality control filtering
        qc_confidence: Confidence threshold for QC filtering (0.0-1.0)

    Requirements satisfied:
    - 3.5: Configuration for Ollama model usage
    """
    dataset_type: DatasetType = DatasetType.SFT
    model: str = "llama3.2"
    language: OutputLanguage = OutputLanguage.EN
    entries_per_chunk: int = 3
    qc_enabled: bool = True
    qc_confidence: float = 0.9

    def __post_init__(self):
        """Validate configuration values."""
        if self.entries_per_chunk < 1:
            raise ValueError("entries_per_chunk must be at least 1")
        if self.entries_per_chunk > 10:
            raise ValueError("entries_per_chunk cannot exceed 10")
        if self.qc_confidence < 0.0 or self.qc_confidence > 1.0:
            raise ValueError("qc_confidence must be between 0.0 and 1.0")


class DocumentDatasetGenerator:
    """
    Generator for creating fine-tuning datasets from document chunks.

    This class handles the generation of training data from document chunks
    using Ollama LLM. It supports multiple output formats and includes
    validation for generated entries.

    Usage:
        config = DocGenerationConfig(dataset_type=DatasetType.SFT)
        generator = DocumentDatasetGenerator(config)
        entries = generator.generate_from_chunks(chunks)

    Requirements satisfied:
    - 3.1: SFT format generation
    - 3.2: Pre-training format generation
    - 3.3: Conversation format generation
    - 3.4: DPO format generation
    - 3.5: Ollama model integration
    - 7.1-7.4: Output validation
    - 7.5: QC flag for quality control filtering
    - 7.6: Taiwan Chinese validation when QC enabled for zh-tw
    """

    def __init__(self, config: DocGenerationConfig):
        """
        Initialize the document dataset generator.

        Args:
            config: Generation configuration
        """
        self.config = config

        # Initialize QC controller if enabled and language is zh-tw
        self._qc_controller: Optional[QualityController] = None
        if config.qc_enabled and config.language == OutputLanguage.ZH_TW:
            self._qc_controller = QualityController(
                enabled=True,
                confidence_threshold=config.qc_confidence
            )

    @property
    def qc_controller(self) -> Optional[QualityController]:
        """Get the QC controller instance."""
        return self._qc_controller

    def generate_from_chunks(
        self,
        chunks: list[TextChunk],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[DatasetEntry]:
        """
        Generate dataset entries from document chunks.

        Args:
            chunks: List of text chunks to generate from
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            List of generated dataset entries

        Requirements satisfied:
        - 3.5: Use Ollama model to generate content based on document context
        - 7.5: QC flag for quality control filtering
        - 7.6: Taiwan Chinese validation when QC enabled for zh-tw
        """
        all_entries: list[DatasetEntry] = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            try:
                # Generate entries for this chunk
                chunk_entries = self._generate_entries_for_chunk(chunk)

                # Validate and filter entries
                valid_entries = [
                    entry for entry in chunk_entries
                    if self._validate_entry(entry)
                ]

                # Apply QC filtering if enabled for zh-tw
                if self._qc_controller is not None:
                    valid_entries = self._apply_qc_filter(valid_entries)

                all_entries.extend(valid_entries)

            except Exception:
                # Log error but continue with other chunks
                pass

            # Report progress
            if progress_callback:
                progress_callback(i + 1, total_chunks)

        return all_entries

    def _apply_qc_filter(self, entries: list[DatasetEntry]) -> list[DatasetEntry]:
        """
        Apply QC filtering to entries for Taiwan Chinese validation.

        Args:
            entries: List of entries to filter

        Returns:
            List of entries that pass QC validation

        Requirements satisfied:
        - 7.5: QC flag for quality control filtering
        - 7.6: Taiwan Chinese validation when QC enabled for zh-tw
        """
        if self._qc_controller is None:
            return entries

        filtered_entries: list[DatasetEntry] = []

        for entry in entries:
            # Convert entry to dict for QC validation
            entry_dict = entry_to_dict(entry)

            # Check if entry passes QC
            passed, failed_fields = self._qc_controller.check_entry(entry_dict)

            if passed:
                filtered_entries.append(entry)

        return filtered_entries

    def get_qc_stats(self) -> Optional[dict[str, Any]]:
        """
        Get QC statistics if QC is enabled.

        Returns:
            QC statistics dict or None if QC is not enabled
        """
        if self._qc_controller is not None:
            return self._qc_controller.get_stats()
        return None

    def _generate_entries_for_chunk(self, chunk: TextChunk) -> list[DatasetEntry]:
        """
        Generate dataset entries for a single chunk.

        Args:
            chunk: The text chunk to generate from

        Returns:
            List of generated entries
        """
        entries: list[DatasetEntry] = []

        # Create prompt for the chunk
        prompt = self._create_prompt_for_chunk(chunk)

        # Get JSON schema for structured output
        json_schema = self._get_json_schema()

        try:
            # Call Ollama API
            response = ollama.chat(
                model=self.config.model,
                messages=[
                    {'role': 'system', 'content': self._get_system_prompt()},
                    {'role': 'user', 'content': prompt}
                ],
                format=json_schema,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                }
            )

            if 'message' in response and 'content' in response['message']:
                raw_content = response['message']['content']
                entries = self._parse_response(raw_content)

        except Exception:
            # Return empty list on error
            pass

        return entries

    def _create_prompt_for_chunk(self, chunk: TextChunk) -> str:
        """
        Create generation prompt for a chunk based on dataset type.

        Args:
            chunk: The text chunk to create prompt for

        Returns:
            The prompt string

        Requirements satisfied:
        - 3.1, 3.2, 3.3, 3.4: Format-specific prompts
        """
        lang_hint = " 請用繁體中文（台灣用語）回答。" if self.config.language == OutputLanguage.ZH_TW else ""
        count = self.config.entries_per_chunk

        base_prompt = f"""Based on the following document content, generate {count} training examples.

Document Content:
---
{chunk.content}
---

"""

        if self.config.dataset_type == DatasetType.SFT:
            return base_prompt + f"""Generate {count} SFT (Supervised Fine-Tuning) examples in JSON array format.
Each example should have:
- instruction: A clear task or question based on the document
- input: Relevant context from the document
- output: The expected answer or completion

Output ONLY a JSON array with {count} objects.{lang_hint}"""

        elif self.config.dataset_type == DatasetType.PRETRAIN:
            return base_prompt + f"""Generate {count} pre-training text samples in JSON array format.
Each example should have:
- text: A coherent, informative paragraph derived from or inspired by the document

Output ONLY a JSON array with {count} objects.{lang_hint}"""

        elif self.config.dataset_type == DatasetType.SFT_CONVERSATION:
            return base_prompt + f"""Generate {count} multi-turn conversation examples in JSON array format.
Each example should have:
- conversations: An array of messages with role (system/user/assistant) and content

Requirements:
- Each conversation must have at least one "user" message and one "assistant" message
- Conversations should be based on the document content
- Include realistic dialogue patterns

Output ONLY a JSON array with {count} objects.{lang_hint}"""

        elif self.config.dataset_type == DatasetType.DPO:
            return base_prompt + f"""Generate {count} DPO (Direct Preference Optimization) examples in JSON array format.
Each example should have:
- prompt: A question or instruction based on the document
- chosen: A high-quality, preferred response
- rejected: A lower-quality, less preferred response

Requirements:
- The "chosen" response must be clearly better than "rejected"
- Both responses should be different from each other

Output ONLY a JSON array with {count} objects.{lang_hint}"""

        return base_prompt

    def _get_system_prompt(self) -> str:
        """Get the system prompt based on dataset type and language."""
        lang_instruction = self._get_language_instruction()

        base = "You are a training data generator. Generate high-quality training examples based on document content."

        return f"{base}{lang_instruction}"

    def _get_language_instruction(self) -> str:
        """Get language-specific instruction for prompts."""
        if self.config.language == OutputLanguage.ZH_TW:
            return """

【語言規範】使用台灣繁體中文，禁止大陸用語：
視頻→影片、軟件→軟體、交互→互動、質量→品質、信息→資訊、數據→資料、網絡→網路、程序→程式、服務器→伺服器、用戶→使用者、優化→最佳化、默認→預設"""

        elif self.config.language == OutputLanguage.ZH_CN:
            return """

【语言规范】使用中国大陆简体中文，符合大陆用语习惯：
影片→视频、軟體→软件、互動→交互、品質→质量、資訊→信息、資料→数据、網路→网络、程式→程序、伺服器→服务器、使用者→用户、最佳化→优化、預設→默认

【文化与价值观规范】
1. 弘扬社会主义核心价值观
2. 尊重中华优秀传统文化，体现文化自信
3. 使用符合中国国情的表述方式
4. 货币使用人民币（元/￥），计量单位使用公制"""
        return ""

    def _get_json_schema(self) -> dict:
        """
        Get JSON schema for structured output based on dataset type.

        Returns:
            JSON schema dict for Ollama format parameter
        """
        if self.config.dataset_type == DatasetType.SFT:
            entry_schema = {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "input": {"type": "string"},
                    "output": {"type": "string"}
                },
                "required": ["instruction", "input", "output"]
            }

        elif self.config.dataset_type == DatasetType.PRETRAIN:
            entry_schema = {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }

        elif self.config.dataset_type == DatasetType.SFT_CONVERSATION:
            message_schema = {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                    "content": {"type": "string"}
                },
                "required": ["role", "content"]
            }
            entry_schema = {
                "type": "object",
                "properties": {
                    "conversations": {
                        "type": "array",
                        "items": message_schema
                    }
                },
                "required": ["conversations"]
            }

        elif self.config.dataset_type == DatasetType.DPO:
            entry_schema = {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "chosen": {"type": "string"},
                    "rejected": {"type": "string"}
                },
                "required": ["prompt", "chosen", "rejected"]
            }

        else:
            # Default to SFT schema
            entry_schema = {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "input": {"type": "string"},
                    "output": {"type": "string"}
                },
                "required": ["instruction", "input", "output"]
            }

        # Return array schema for batch generation
        return {
            "type": "array",
            "items": entry_schema
        }

    def _parse_response(self, raw_content: str) -> list[DatasetEntry]:
        """
        Parse the raw response content into dataset entries.

        Args:
            raw_content: Raw JSON string from Ollama

        Returns:
            List of parsed dataset entries
        """
        entries: list[DatasetEntry] = []

        try:
            data = json.loads(raw_content)

            # Handle both single object and array responses
            if isinstance(data, dict):
                data = [data]

            if not isinstance(data, list):
                return entries

            for item in data:
                if not isinstance(item, dict):
                    continue

                entry = self._dict_to_entry(item)
                if entry is not None:
                    entries.append(entry)

        except json.JSONDecodeError:
            pass

        return entries

    def _dict_to_entry(self, data: dict) -> Optional[DatasetEntry]:
        """
        Convert a dictionary to the appropriate dataset entry type.

        Args:
            data: Dictionary with entry data

        Returns:
            Dataset entry or None if conversion fails
        """
        try:
            if self.config.dataset_type == DatasetType.SFT:
                return DataEntry(
                    instruction=str(data.get('instruction', '')),
                    input=str(data.get('input', '')),
                    output=str(data.get('output', ''))
                )

            elif self.config.dataset_type == DatasetType.PRETRAIN:
                return PretrainEntry(
                    text=str(data.get('text', ''))
                )

            elif self.config.dataset_type == DatasetType.SFT_CONVERSATION:
                conversations_data = data.get('conversations', [])
                if not isinstance(conversations_data, list):
                    return None

                conversations = []
                for msg in conversations_data:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        role = msg['role']
                        if role in ('system', 'user', 'assistant'):
                            conversations.append(ConversationMessage(
                                role=role,
                                content=str(msg['content'])
                            ))

                if conversations:
                    return SFTConversationEntry(conversations=conversations)
                return None

            elif self.config.dataset_type == DatasetType.DPO:
                return DPOEntry(
                    prompt=str(data.get('prompt', '')),
                    chosen=str(data.get('chosen', '')),
                    rejected=str(data.get('rejected', ''))
                )

        except Exception:
            pass

        return None

    def _validate_entry(self, entry: DatasetEntry) -> bool:
        """
        Validate a generated dataset entry.

        Args:
            entry: The entry to validate

        Returns:
            True if entry is valid, False otherwise

        Requirements satisfied:
        - 7.1: Validate entries match expected format schema
        - 7.2: Ensure SFT fields are non-empty
        - 7.3: Ensure Conversation has user and assistant messages
        - 7.4: Ensure DPO chosen and rejected are different
        """
        return validate_entry(entry, self.config.dataset_type)


# ============================================================================
# Validation Functions (exported for property testing)
# ============================================================================

def validate_entry(entry: DatasetEntry, dataset_type: DatasetType) -> bool:
    """
    Validate a dataset entry against its format requirements.

    Args:
        entry: The entry to validate
        dataset_type: The expected dataset type

    Returns:
        True if entry is valid, False otherwise

    Requirements satisfied:
    - 7.1: Validate entries match expected format schema
    - 7.2: Ensure SFT fields are non-empty
    - 7.3: Ensure Conversation has user and assistant messages
    - 7.4: Ensure DPO chosen and rejected are different
    """
    if dataset_type == DatasetType.SFT:
        return validate_sft_entry(entry)
    elif dataset_type == DatasetType.PRETRAIN:
        return validate_pretrain_entry(entry)
    elif dataset_type == DatasetType.SFT_CONVERSATION:
        return validate_conversation_entry(entry)
    elif dataset_type == DatasetType.DPO:
        return validate_dpo_entry(entry)

    return False


def validate_sft_entry(entry: DatasetEntry) -> bool:
    """
    Validate an SFT entry.

    Requirements:
    - Must be a DataEntry instance
    - instruction, input, output must all be non-empty strings

    Requirements satisfied:
    - 7.1: Schema validation
    - 7.2: Non-empty field validation
    """
    if not isinstance(entry, DataEntry):
        return False

    # All fields must be non-empty
    if not entry.instruction or not entry.instruction.strip():
        return False
    if not entry.input or not entry.input.strip():
        return False
    if not entry.output or not entry.output.strip():
        return False

    return True


def validate_pretrain_entry(entry: DatasetEntry) -> bool:
    """
    Validate a pre-training entry.

    Requirements:
    - Must be a PretrainEntry instance
    - text must be non-empty

    Requirements satisfied:
    - 7.1: Schema validation
    """
    if not isinstance(entry, PretrainEntry):
        return False

    # Text must be non-empty
    if not entry.text or not entry.text.strip():
        return False

    return True


def validate_conversation_entry(entry: DatasetEntry) -> bool:
    """
    Validate a conversation entry.

    Requirements:
    - Must be a SFTConversationEntry instance
    - conversations must be non-empty
    - Must have at least one user message
    - Must have at least one assistant message

    Requirements satisfied:
    - 7.1: Schema validation
    - 7.3: User and assistant message validation
    """
    if not isinstance(entry, SFTConversationEntry):
        return False

    # Must have conversations
    if not entry.conversations:
        return False

    # Check for required roles
    has_user = False
    has_assistant = False

    for msg in entry.conversations:
        if not isinstance(msg, ConversationMessage):
            return False
        if not msg.content or not msg.content.strip():
            return False

        if msg.role == 'user':
            has_user = True
        elif msg.role == 'assistant':
            has_assistant = True

    # Must have at least one user and one assistant message
    return has_user and has_assistant


def validate_dpo_entry(entry: DatasetEntry) -> bool:
    """
    Validate a DPO entry.

    Requirements:
    - Must be a DPOEntry instance
    - prompt, chosen, rejected must all be non-empty
    - chosen and rejected must be different

    Requirements satisfied:
    - 7.1: Schema validation
    - 7.4: Chosen/rejected differentiation
    """
    if not isinstance(entry, DPOEntry):
        return False

    # All fields must be non-empty
    if not entry.prompt or not entry.prompt.strip():
        return False
    if not entry.chosen or not entry.chosen.strip():
        return False
    if not entry.rejected or not entry.rejected.strip():
        return False

    # Chosen and rejected must be different
    if entry.chosen.strip() == entry.rejected.strip():
        return False

    return True


def entry_to_dict(entry: DatasetEntry) -> dict:
    """
    Convert a dataset entry to a dictionary for serialization.

    Args:
        entry: The dataset entry to convert

    Returns:
        Dictionary representation of the entry

    Requirements satisfied:
    - 3.6: Serialization support
    """
    if isinstance(entry, DataEntry):
        return {
            'instruction': entry.instruction,
            'input': entry.input,
            'output': entry.output
        }
    elif isinstance(entry, PretrainEntry):
        return {
            'text': entry.text
        }
    elif isinstance(entry, SFTConversationEntry):
        return {
            'conversations': [
                {'role': msg.role, 'content': msg.content}
                for msg in entry.conversations
            ]
        }
    elif isinstance(entry, DPOEntry):
        return {
            'prompt': entry.prompt,
            'chosen': entry.chosen,
            'rejected': entry.rejected
        }

    return {}


def dict_to_entry(data: dict, dataset_type: DatasetType) -> Optional[DatasetEntry]:
    """
    Convert a dictionary to a dataset entry.

    Args:
        data: Dictionary with entry data
        dataset_type: The type of entry to create

    Returns:
        Dataset entry or None if conversion fails

    Requirements satisfied:
    - 3.6: Deserialization support
    """
    try:
        if dataset_type == DatasetType.SFT:
            return DataEntry(
                instruction=str(data.get('instruction', '')),
                input=str(data.get('input', '')),
                output=str(data.get('output', ''))
            )

        elif dataset_type == DatasetType.PRETRAIN:
            return PretrainEntry(
                text=str(data.get('text', ''))
            )

        elif dataset_type == DatasetType.SFT_CONVERSATION:
            conversations_data = data.get('conversations', [])
            if not isinstance(conversations_data, list):
                return None

            conversations = []
            for msg in conversations_data:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    role = msg['role']
                    if role in ('system', 'user', 'assistant'):
                        conversations.append(ConversationMessage(
                            role=role,
                            content=str(msg['content'])
                        ))

            if conversations:
                return SFTConversationEntry(conversations=conversations)
            return None

        elif dataset_type == DatasetType.DPO:
            return DPOEntry(
                prompt=str(data.get('prompt', '')),
                chosen=str(data.get('chosen', '')),
                rejected=str(data.get('rejected', ''))
            )

    except Exception:
        pass

    return None
