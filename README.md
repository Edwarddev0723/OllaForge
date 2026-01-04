<p align="center">
  <img src="img/banner.png" alt="OllaForge Banner" width="100%">
</p>

<h1 align="center">OllaForge 🔥</h1>

<p align="center">
  <strong>AI-Powered Dataset Generator & Augmentor for LLM Fine-tuning</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-web-interface">Web Interface</a> •
  <a href="#-document-to-dataset">Doc2Dataset</a> •
  <a href="#-dataset-augmentation">Augmentation</a> •
  <a href="#-dataset-formats">Formats</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-TW.md">繁體中文</a>
</p>

<p align="center">
  <a href="https://github.com/ollaforge/ollaforge/actions"><img src="https://img.shields.io/github/actions/workflow/status/ollaforge/ollaforge/ci.yml?branch=main&label=CI&logo=github" alt="CI Status"></a>
  <a href="https://pypi.org/project/ollaforge/"><img src="https://img.shields.io/pypi/v/ollaforge?color=blue&logo=pypi&logoColor=white" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/ollaforge/"><img src="https://img.shields.io/pypi/pyversions/ollaforge?logo=python&logoColor=white" alt="Python Versions"></a>
  <a href="https://github.com/ollaforge/ollaforge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ollaforge/ollaforge?color=green" alt="License"></a>
  <a href="https://github.com/ollaforge/ollaforge/stargazers"><img src="https://img.shields.io/github/stars/ollaforge/ollaforge?style=social" alt="GitHub Stars"></a>
</p>

---

## 🎯 What is OllaForge?

**OllaForge** is a high-performance CLI tool that leverages local Ollama models to **generate** and **augment** training datasets for LLM fine-tuning. With structured JSON output, concurrent batch processing, and built-in quality control, it's optimized for both quality and speed.

### Why OllaForge?

- 🔒 **100% Local & Private** - Your data never leaves your machine
- ⚡ **Blazing Fast** - Concurrent batch processing with structured output
- 🎨 **Flexible** - Generate new datasets or augment existing ones
- 🌐 **Multilingual** - English & Traditional Chinese with QC validation
- 🔧 **Production Ready** - HuggingFace & LLaMA-Factory compatible

---

## ✨ Features

### 🆕 Dataset Generation

| Feature | Description |
|---------|-------------|
| 🎯 **Natural Language Topics** | Describe your dataset needs in plain language |
| 🤖 **Any Ollama Model** | Works with Llama 3, Mistral, Qwen, DeepSeek, Gemma, and more |
| 📊 **4 Dataset Formats** | SFT, Pre-training, Conversation (ShareGPT), DPO |
| ⚡ **Concurrent Batching** | Generate hundreds of entries in minutes |

### 🔄 Dataset Augmentation

| Feature | Description |
|---------|-------------|
| 📝 **Field Modification** | Enhance existing fields with AI-powered transformations |
| ➕ **New Field Creation** | Add computed fields based on existing data |
| 👀 **Preview Mode** | Test augmentation on samples before full processing |
| 🛡️ **Failure Recovery** | Preserves original data on AI failures |

### 📄 Document to Dataset

| Feature | Description |
|---------|-------------|
| 📑 **Multi-Format Parsing** | Markdown, PDF, HTML, TXT, JSON, and code files |
| ✂️ **Smart Chunking** | Semantic boundary-aware text splitting |
| 🎯 **4 Output Formats** | SFT, Pre-training, Conversation, DPO |
| 📁 **Batch Processing** | Process entire directories with glob patterns |
| 🔍 **Quality Control** | Built-in validation and QC filtering |

### 📁 Multi-Format Support

| Feature | Description |
|---------|-------------|
| 📄 **JSONL** | JSON Lines format (default) - one JSON object per line |
| 📋 **JSON** | Single JSON array of objects |
| 📊 **CSV** | Comma-separated values with automatic header detection |
| 📑 **TSV** | Tab-separated values for structured data |
| 🗃️ **Parquet** | Columnar storage format (requires pandas) |

### 🌐 Quality & Localization

| Feature | Description |
|---------|-------------|
| 🔍 **BERT-based QC** | Filters Mainland Chinese expressions for Taiwan datasets |
| 🌏 **Multi-language** | English, Traditional Chinese (Taiwan), and Simplified Chinese (China) |
| ✅ **Structured Output** | JSON schema enforcement for 0% format errors |
| 📈 **Progress Tracking** | Real-time progress with Rich-powered UI |

### 🤗 HuggingFace Integration

| Feature | Description |
|---------|-------------|
| 📥 **Direct Loading** | Load datasets directly from HuggingFace Hub |
| 🔄 **Augment HF Datasets** | Augment any HuggingFace dataset without downloading |
| ⚙️ **Split & Config Support** | Specify dataset splits and configurations |
| 📊 **Large Dataset Handling** | Limit entries for efficient processing |

### 🖥️ Web Interface (🚧 In Development)

| Feature | Description |
|---------|-------------|
| 🌐 **Browser-based UI** | No CLI knowledge required - use through your browser |
| 📊 **Real-time Progress** | WebSocket-powered live progress updates |
| 🌍 **Bilingual Support** | English and Traditional Chinese interface |
| 🐳 **Docker Ready** | One-command deployment with Docker Compose |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running

### Installation

```bash
# Install from PyPI (recommended)
pip install ollaforge

# Or install from source
git clone https://github.com/ollaforge/ollaforge.git
cd ollaforge
pip install -e .

# With QC support for Traditional Chinese
pip install ollaforge[qc]

# With multi-format support (CSV, Parquet, etc.)
pip install ollaforge[formats]

# With HuggingFace datasets support
pip install ollaforge[hf]

# With all features
pip install ollaforge[all]
```

### Upgrade Existing Installation

```bash
# Upgrade to latest version
pip install --upgrade ollaforge

# Upgrade with all features
pip install --upgrade ollaforge[all]

# Force reinstall (if having issues)
pip install --force-reinstall ollaforge
```

### Your First Dataset

```bash
# Interactive mode (recommended for beginners)
ollaforge -i

# Generate SFT dataset
ollaforge generate "Python programming tutorials" --count 100 --output python_sft.jsonl

# Traditional Chinese conversation dataset
ollaforge generate "咖啡點餐對話" --type sft_conv --lang zh-tw --count 100
```

### Augment Existing Dataset

```bash
# Preview augmentation before processing
ollaforge augment data.jsonl --field output --instruction "Add more detail" --preview

# Augment with new field
ollaforge augment data.jsonl --field difficulty --new-field --instruction "Rate difficulty: easy/medium/hard"

# Work with CSV files
ollaforge augment data.csv --field sentiment --new-field --instruction "Analyze sentiment: positive/negative/neutral"

# Augment HuggingFace dataset directly
ollaforge augment renhehuang/govQA-database-zhtw --field answer --instruction "Translate to English" --output translated.jsonl

# Convert between formats
ollaforge convert data.csv data.jsonl
```

### Convert Documents to Datasets

```bash
# Convert Markdown documentation to SFT dataset
ollaforge doc2dataset README.md --type sft --output readme_dataset.jsonl

# Process all Python files in a directory
ollaforge doc2dataset ./src --pattern "*.py" --type pretrain

# Convert PDF with Traditional Chinese output
ollaforge doc2dataset manual.pdf --lang zh-tw --qc
```

---

## 📖 Usage

### Generate Command

```bash
ollaforge generate <topic> [options]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--count` | `-c` | 10 | Number of entries (1-10,000) |
| `--model` | `-m` | llama3.2 | Ollama model name |
| `--output` | `-o` | dataset.jsonl | Output filename |
| `--type` | `-t` | sft | Format: `sft`, `pretrain`, `sft_conv`, `dpo` |
| `--lang` | `-l` | en | Language: `en`, `zh-tw`, `zh-cn` |
| `--concurrency` | `-j` | 5 | Parallel requests (1-20) |
| `--qc/--no-qc` | | --qc | Taiwan Chinese QC filter |
| `--interactive` | `-i` | | Launch wizard mode |

### Augment Command

```bash
ollaforge augment <input_file_or_hf_dataset> [options]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--field` | `-f` | required | Target field to augment |
| `--instruction` | `-I` | required | AI instruction for augmentation |
| `--output` | `-o` | auto | Output file (default: input_augmented.jsonl) |
| `--model` | `-m` | llama3.2 | Ollama model name |
| `--lang` | `-l` | en | Language: `en`, `zh-tw`, `zh-cn` |
| `--new-field` | | false | Create new field instead of modifying |
| `--context` | `-c` | | Additional context fields |
| `--preview` | `-p` | | Preview before full processing |
| `--preview-count` | | 3 | Number of entries to preview (1-10) |
| `--concurrency` | `-j` | 5 | Parallel requests |
| `--input-format` | | auto | Input format: jsonl, json, csv, tsv, parquet |
| `--output-format` | | auto | Output format: jsonl, json, csv, tsv, parquet |
| `--hf-split` | | train | HuggingFace dataset split to use |
| `--hf-config` | | | HuggingFace dataset configuration name |
| `--max-entries` | | | Maximum entries to load (for large datasets) |
| `--force` | `-y` | | Overwrite output without confirmation |

### Doc2Dataset Command

Convert documents (Markdown, PDF, HTML, TXT, JSON, code files) into fine-tuning datasets.

```bash
ollaforge doc2dataset <source> [options]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | dataset.jsonl | Output file path |
| `--type` | `-t` | sft | Format: `sft`, `pretrain`, `sft_conv`, `dpo` |
| `--model` | `-m` | llama3.2 | Ollama model name |
| `--chunk-size` | | 2000 | Chunk size in characters (500-10000) |
| `--chunk-overlap` | | 200 | Overlap between chunks (0-1000) |
| `--count` | `-c` | 3 | Entries to generate per chunk (1-10) |
| `--lang` | `-l` | en | Language: `en`, `zh-tw`, `zh-cn` |
| `--pattern` | `-p` | | File pattern for directories (e.g., `*.md`) |
| `--recursive/--no-recursive` | | --recursive | Recursively process directories |
| `--qc/--no-qc` | | --qc | Enable quality control |

#### Supported File Formats

| Format | Extensions | Description |
|--------|------------|-------------|
| Markdown | `.md` | Preserves heading structure |
| PDF | `.pdf` | Extracts text from all pages |
| HTML | `.html`, `.htm` | Removes tags, preserves text |
| Text | `.txt` | Direct text reading |
| JSON | `.json` | Extracts string values |
| Code | `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, `.rb` | Language detection |

#### Examples

```bash
# Convert a single Markdown file to SFT dataset
ollaforge doc2dataset README.md --type sft --output readme_sft.jsonl

# Convert all Python files in a directory
ollaforge doc2dataset ./src --pattern "*.py" --type pretrain

# Generate conversation data from documentation
ollaforge doc2dataset docs/ --type sft_conv --lang zh-tw

# Convert PDF with custom chunk settings
ollaforge doc2dataset manual.pdf --chunk-size 3000 --chunk-overlap 300 --count 5

# Process HTML files with quality control
ollaforge doc2dataset ./html_docs --pattern "*.html" --qc --lang zh-tw
```

---

## 🖥️ Web Interface

> ⚠️ **Note: The web interface is currently under active development.** Some features may be incomplete or unstable.

OllaForge provides a modern web interface for users who prefer a graphical UI over command line.

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/ollaforge/ollaforge.git
cd ollaforge

# Start with Docker Compose (requires Ollama running on host)
docker-compose up -d

# Access the web interface
open http://localhost
```

### Manual Setup

**Backend:**
```bash
pip install ollaforge[web]
python -m ollaforge.web.server
# Server runs at http://localhost:8000
```

**Frontend:**
```bash
cd ollaforge-web
npm install
npm run dev
# Frontend runs at http://localhost:5173
```

### Pages

| Page | Description |
|------|-------------|
| `/generate` | Create new datasets from topic descriptions |
| `/augment` | Upload and enhance existing datasets |
| `/config` | Manage saved configurations |

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📄 Document to Dataset

OllaForge can convert various document formats into fine-tuning datasets using AI-powered content analysis.

### How It Works

1. **Parse**: Documents are parsed to extract text content
2. **Chunk**: Long documents are split into manageable chunks respecting semantic boundaries
3. **Generate**: AI analyzes each chunk and generates training data entries
4. **Validate**: Output is validated against format schemas with optional QC filtering

### Use Cases

- **Documentation → Q&A**: Convert technical docs into question-answer pairs
- **Code → Tutorials**: Transform code files into instructional content
- **Articles → Conversations**: Create dialogue datasets from articles
- **Manuals → Training Data**: Generate fine-tuning data from product manuals

### Examples

```bash
# Convert project README to SFT dataset
ollaforge doc2dataset README.md --type sft --count 5

# Process all Markdown docs in a folder
ollaforge doc2dataset ./docs --pattern "*.md" --type sft_conv

# Convert PDF manual with Traditional Chinese output
ollaforge doc2dataset manual.pdf --lang zh-tw --qc

# Generate pre-training data from source code
ollaforge doc2dataset ./src --pattern "*.py" --type pretrain --chunk-size 1500

# Batch process HTML documentation
ollaforge doc2dataset ./html_docs --pattern "*.html" --type sft --output training_data.jsonl
```

### Installation

Document parsing requires additional dependencies:

```bash
# Install with document parsing support
pip install ollaforge[docs]

# Or install all features
pip install ollaforge[all]
```

---

## 🔄 Dataset Augmentation

OllaForge can enhance existing datasets by modifying or adding fields using AI.

### Use Cases

- **Translation**: Translate fields to different languages
- **Enrichment**: Add metadata like difficulty, category, or sentiment
- **Expansion**: Expand brief answers into detailed explanations
- **Transformation**: Convert formats or styles

### Examples

```bash
# Translate output field to Chinese
ollaforge augment qa.jsonl -f output -I "Translate to Traditional Chinese (Taiwan)"

# Add difficulty rating
ollaforge augment problems.csv -f difficulty --new-field -I "Rate: easy/medium/hard"

# Expand brief answers
ollaforge augment faq.json -f answer -I "Expand with more detail and examples"

# Add category using context
ollaforge augment articles.tsv -f category --new-field -c title -c content -I "Categorize: tech/science/business/other"
```

### Preview Mode

Always preview before processing large datasets:

```bash
ollaforge augment large_dataset.jsonl -f output -I "Improve clarity" --preview
```

---

## 📋 Dataset Formats

### SFT (Alpaca Format)
```json
{"instruction": "Explain recursion", "input": "", "output": "Recursion is..."}
```

### Pre-training
```json
{"text": "Machine learning is a subset of artificial intelligence..."}
```

### SFT Conversation (ShareGPT/ChatML)
```json
{
  "conversations": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I reverse a string?"},
    {"role": "assistant", "content": "Use slicing: `s[::-1]`"}
  ]
}
```

### DPO (Preference Pairs)
```json
{"prompt": "Write factorial", "chosen": "def factorial(n)...", "rejected": "def f(n):..."}
```

---

## ⚡ Performance

| Optimization | Benefit |
|--------------|---------|
| **Structured JSON Output** | 0% format errors via Ollama's schema enforcement |
| **Small Batch Size (5)** | Reduces attention decay, improves quality |
| **Concurrent Requests** | Up to 10 parallel batch requests |
| **BERT on CPU** | Keeps GPU/MPS free for LLM generation |

### Benchmarks

| Task | Entries | Time | Throughput |
|------|---------|------|------------|
| SFT Generation | 1,000 | ~5 min | 200/min |
| Conversation | 500 | ~8 min | 60/min |
| Augmentation | 1,000 | ~3 min | 330/min |

*Tested on M2 Max with llama3.2:8b*

---

## 🔍 Traditional Chinese QC

When using `--lang zh-tw`, OllaForge automatically filters Mainland Chinese expressions:

| ❌ Filtered | ✅ Accepted |
|-------------|-------------|
| 軟件 | 軟體 |
| 視頻 | 影片 |
| 程序 | 程式 |
| 網絡 | 網路 |
| 信息 | 資訊 |

---

## 🤖 Recommended Models

| Model | Best For | VRAM |
|-------|----------|------|
| `llama3.2` | General purpose (default) | 8GB |
| `qwen2.5:14b` | Multilingual, Chinese | 16GB |
| `deepseek-r1:14b` | Reasoning tasks | 16GB |
| `gemma2:9b` | Efficient, balanced | 12GB |
| `mistral:7b` | Fast inference | 8GB |

---

## 🧪 Development

```bash
# Clone and setup
git clone https://github.com/ollaforge/ollaforge.git
cd ollaforge
pip install -e ".[dev]"

# Run tests
make test

# Lint & format
make lint
make format

# All checks
make check
```

### Project Structure

```
ollaforge/
├── ollaforge/           # Core Python package
│   ├── web/             # Web API server
│   └── ...
├── ollaforge-web/       # React frontend (Web UI)
├── tests/               # Test suite
├── docs/                # User documentation
├── examples/            # Example datasets and configs
└── .kiro/specs/         # Internal design specifications
```

### Architecture & Specs

Internal design documents and specifications are located in `.kiro/specs/`:

| Spec | Description |
|------|-------------|
| `dataset-augmentation/` | Dataset augmentation feature design |
| `document-to-dataset/` | Doc2Dataset conversion pipeline |
| `web-interface/` | Web UI architecture and API design |

Each spec directory contains:
- `requirements.md` - User stories and acceptance criteria
- `design.md` - Technical design and architecture
- `tasks.md` - Implementation tasks and progress

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Write tests for your changes
4. Ensure all tests pass (`make check`)
5. Submit a Pull Request

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM inference
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal UI
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Pydantic](https://pydantic.dev/) - Data validation

---

<p align="center">
  <strong>Made with ❤️ by the OllaForge Team</strong>
</p>

<p align="center">
  <a href="https://github.com/ollaforge/ollaforge/issues/new?template=bug_report.md">Report Bug</a> •
  <a href="https://github.com/ollaforge/ollaforge/issues/new?template=feature_request.md">Request Feature</a> •
  <a href="https://github.com/ollaforge/ollaforge/discussions">Discussions</a>
</p>
