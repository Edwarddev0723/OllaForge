# Requirements Document

## Introduction

本功能擴展 OllaForge CLI，新增從各種文件格式自動生成 fine-tuning 資料集的能力。使用者可以上傳 Markdown、PDF、HTML、TXT、JSON 或程式碼檔案，系統會透過 Ollama LLM 分析文件內容並生成符合 SFT、Pre-training、Conversation (ShareGPT) 或 DPO 格式的訓練資料。

## Glossary

- **Document_Parser**: 負責解析各種文件格式並提取純文字內容的模組
- **Chunk_Splitter**: 將長文件分割成適合 LLM 處理的文字區塊的元件
- **Dataset_Generator**: 根據文件內容和指定格式生成訓練資料的核心引擎
- **CLI**: OllaForge 命令列介面
- **SFT**: Supervised Fine-Tuning，監督式微調資料格式 (instruction/input/output)
- **Pre-training**: 預訓練資料格式 (純文字)
- **Conversation**: ShareGPT/ChatML 多輪對話格式
- **DPO**: Direct Preference Optimization，偏好優化資料格式 (prompt/chosen/rejected)

## Requirements

### Requirement 1: 文件格式支援

**User Story:** As a developer, I want to convert various document formats into training datasets, so that I can leverage existing documentation for fine-tuning LLMs.

#### Acceptance Criteria

1. WHEN a Markdown file (.md) is provided, THE Document_Parser SHALL extract text content preserving heading structure
2. WHEN a PDF file (.pdf) is provided, THE Document_Parser SHALL extract text content from all pages
3. WHEN an HTML file (.html, .htm) is provided, THE Document_Parser SHALL extract text content removing HTML tags
4. WHEN a plain text file (.txt) is provided, THE Document_Parser SHALL read the content directly
5. WHEN a JSON file (.json) is provided, THE Document_Parser SHALL extract text values from the JSON structure
6. WHEN a code file (.py, .js, .ts, .java, .go, .rs, .c, .cpp, .rb) is provided, THE Document_Parser SHALL extract code content with language detection
7. IF an unsupported file format is provided, THEN THE Document_Parser SHALL return a descriptive error message listing supported formats

### Requirement 2: 文件分塊處理

**User Story:** As a developer, I want long documents to be automatically split into manageable chunks, so that the LLM can process them effectively.

#### Acceptance Criteria

1. WHEN a document exceeds the chunk size limit, THE Chunk_Splitter SHALL divide it into smaller segments
2. THE Chunk_Splitter SHALL preserve semantic boundaries (paragraphs, sections, code blocks) when splitting
3. WHEN splitting Markdown documents, THE Chunk_Splitter SHALL respect heading hierarchy
4. WHEN splitting code files, THE Chunk_Splitter SHALL preserve function/class boundaries where possible
5. THE Chunk_Splitter SHALL allow configurable chunk size (default: 2000 characters)
6. THE Chunk_Splitter SHALL allow configurable overlap between chunks (default: 200 characters)

### Requirement 3: 資料集格式生成

**User Story:** As a developer, I want to generate datasets in multiple training formats, so that I can use them with different fine-tuning approaches.

#### Acceptance Criteria

1. WHEN SFT format is selected, THE Dataset_Generator SHALL produce entries with instruction, input, and output fields
2. WHEN Pre-training format is selected, THE Dataset_Generator SHALL produce entries with text field only
3. WHEN Conversation format is selected, THE Dataset_Generator SHALL produce entries with conversations array containing role/content pairs
4. WHEN DPO format is selected, THE Dataset_Generator SHALL produce entries with prompt, chosen, and rejected fields
5. THE Dataset_Generator SHALL use the Ollama model to generate appropriate content based on document context
6. FOR ALL generated entries, serializing then deserializing SHALL produce equivalent data (round-trip property)

### Requirement 4: CLI 命令整合

**User Story:** As a developer, I want a simple CLI command to convert documents to datasets, so that I can easily integrate this into my workflow.

#### Acceptance Criteria

1. THE CLI SHALL provide a `doc2dataset` subcommand for document conversion
2. WHEN the command is invoked, THE CLI SHALL accept a document path as required argument
3. THE CLI SHALL accept `--type` option to specify output dataset format (sft, pretrain, sft_conv, dpo)
4. THE CLI SHALL accept `--model` option to specify the Ollama model (default: llama3.2)
5. THE CLI SHALL accept `--output` option to specify output file path
6. THE CLI SHALL accept `--chunk-size` option to configure chunk size
7. THE CLI SHALL accept `--chunk-overlap` option to configure overlap
8. THE CLI SHALL accept `--count` option to specify number of entries to generate per chunk
9. THE CLI SHALL accept `--lang` option to specify output language (en, zh-tw)
10. THE CLI SHALL display progress during generation with entry count and estimated time

### Requirement 5: 錯誤處理與驗證

**User Story:** As a developer, I want clear error messages when something goes wrong, so that I can quickly identify and fix issues.

#### Acceptance Criteria

1. IF the input file does not exist, THEN THE CLI SHALL display an error with the file path
2. IF the input file cannot be read, THEN THE CLI SHALL display a permission error message
3. IF the Ollama model is not available, THEN THE CLI SHALL display connection instructions
4. IF document parsing fails, THEN THE Document_Parser SHALL return a descriptive error with the failure reason
5. WHEN generation is interrupted, THE CLI SHALL save partial results to the output file
6. THE CLI SHALL validate all parameters before starting generation

### Requirement 6: 多文件批次處理

**User Story:** As a developer, I want to process multiple documents at once, so that I can efficiently create large datasets.

#### Acceptance Criteria

1. THE CLI SHALL accept a directory path to process all supported files within
2. WHEN a directory is provided, THE CLI SHALL recursively find all supported document files
3. THE CLI SHALL accept `--pattern` option to filter files by glob pattern (e.g., "*.md")
4. WHEN processing multiple files, THE CLI SHALL display per-file progress
5. THE CLI SHALL combine results from all files into a single output dataset
6. IF any file fails to process, THEN THE CLI SHALL continue with remaining files and report failures at the end

### Requirement 7: 生成品質控制

**User Story:** As a developer, I want to ensure generated data quality, so that my fine-tuned models perform well.

#### Acceptance Criteria

1. THE Dataset_Generator SHALL validate generated entries match the expected format schema
2. WHEN generating SFT data, THE Dataset_Generator SHALL ensure instruction, input, and output are non-empty
3. WHEN generating Conversation data, THE Dataset_Generator SHALL ensure at least one user and one assistant message exist
4. WHEN generating DPO data, THE Dataset_Generator SHALL ensure chosen and rejected responses are meaningfully different
5. THE CLI SHALL accept `--qc` flag to enable quality control filtering
6. WHEN QC is enabled for zh-tw language, THE Dataset_Generator SHALL apply Taiwan Chinese validation
