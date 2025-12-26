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
  <a href="#-dataset-augmentation">Augmentation</a> •
  <a href="#-dataset-formats">Formats</a> •
  <a href="#-performance">Performance</a> •
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

<p align="center">
  <a href="https://codecov.io/gh/ollaforge/ollaforge"><img src="https://img.shields.io/codecov/c/github/ollaforge/ollaforge?logo=codecov" alt="Coverage"></a>
  <a href="https://github.com/ollaforge/ollaforge/issues"><img src="https://img.shields.io/github/issues/ollaforge/ollaforge" alt="Issues"></a>
  <a href="https://github.com/ollaforge/ollaforge/pulls"><img src="https://img.shields.io/github/issues-pr/ollaforge/ollaforge" alt="Pull Requests"></a>
  <a href="https://ollama.ai/"><img src="https://img.shields.io/badge/Ollama-Local%20LLM-orange?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+" alt="Ollama"></a>
</p>

---

## � What iis OllaForge?

**OllaForge** is a high-performance CLI tool that leverages local Ollama models to **generate** and **augment** training datasets for LLM fine-tuning. With structured JSON output, concurrent batch processing, and built-in quality control, it's optimized for both quality and speed.

### Why OllaForge?

- � **u100% Local & Private** - Your data never leaves your machine
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
| 🌏 **Multi-language** | English and Traditional Chinese (Taiwan) support |
| ✅ **Structured Output** | JSON schema enforcement for 0% format errors |
| 📈 **Progress Tracking** | Real-time progress with Rich-powered UI |

### 🖥️ Web Interface
| Feature | Description |
|---------|-------------|
| 🌐 **Browser-based UI** | No CLI knowledge required - use through your browser |
| 📊 **Real-time Progress** | WebSocket-powered live progress updates |
| 🌍 **Bilingual Support** | English and Traditional Chinese interface |
| 💾 **Configuration Save** | Save and load your favorite settings |
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

# With all features
pip install ollaforge[all]
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

# Convert between formats
ollaforge convert data.csv data.jsonl

# Interactive augmentation wizard
ollaforge augment data.jsonl -i
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
| `--lang` | `-l` | en | Language: `en`, `zh-tw` |
| `--concurrency` | `-j` | 5 | Parallel requests (1-20) |
| `--qc/--no-qc` | | --qc | Taiwan Chinese QC filter |
| `--interactive` | `-i` | | Launch wizard mode |

### Augment Command

```bash
ollaforge augment <input_file> [options]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--field` | `-f` | required | Target field to augment |
| `--instruction` | `-I` | required | AI instruction for augmentation |
| `--output` | `-o` | auto | Output file (default: input_augmented.jsonl) |
| `--model` | `-m` | llama3.2 | Ollama model name |
| `--new-field` | | false | Create new field instead of modifying |
| `--context` | `-c` | | Additional context fields |
| `--preview` | `-p` | | Preview before full processing |
| `--concurrency` | `-j` | 5 | Parallel requests |
| `--interactive` | `-i` | | Interactive mode |

---

## 🖥️ Web Interface

OllaForge provides a modern web interface for users who prefer a graphical UI over command line.

### Features

- **Dataset Generation Page** - Generate datasets with a visual form
- **Dataset Augmentation Page** - Upload, preview, and augment datasets
- **Configuration Management** - Save and load your favorite settings
- **Real-time Progress** - WebSocket-powered live progress updates
- **Bilingual UI** - English and Traditional Chinese (Taiwan) support
- **Multi-format Support** - Upload/download in JSONL, JSON, CSV, TSV, Parquet

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

**Backend (Python):**
```bash
# Install web dependencies
pip install ollaforge[web]

# Start the API server
python -m ollaforge.web.server
# Server runs at http://localhost:8000
```

**Frontend (Node.js):**
```bash
# Navigate to frontend directory
cd ollaforge-web

# Install dependencies
npm install

# Start development server
npm run dev
# Frontend runs at http://localhost:5173
```

### Web Interface Pages

#### 1. Generate Page (`/generate`)
Create new datasets from topic descriptions:
- Enter topic in natural language
- Select model, count, dataset type, and language
- View real-time generation progress
- Preview generated entries before download
- Download in multiple formats

#### 2. Augment Page (`/augment`)
Enhance existing datasets:
- Drag-and-drop file upload (JSONL, JSON, CSV, TSV, Parquet)
- Select target field and provide AI instructions
- Preview augmentation on sample entries
- Process full dataset with progress tracking
- Download augmented dataset

#### 3. Configuration Page (`/config`)
Manage saved configurations:
- View all saved generation/augmentation configs
- Load configurations to pre-fill forms
- Delete unused configurations

### API Documentation

The backend provides OpenAPI documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Allowed CORS origins |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `PORT` | `8000` | Backend server port |
| `DEBUG` | `false` | Enable debug mode |

---

## 🔄 Dataset Augmentation

OllaForge can enhance existing JSONL datasets by modifying or adding fields using AI.

### Use Cases

- **Translation**: Translate fields to different languages
- **Enrichment**: Add metadata like difficulty, category, or sentiment
- **Expansion**: Expand brief answers into detailed explanations
- **Transformation**: Convert formats or styles

### Examples

```bash
# Translate output field to Chinese (JSONL)
ollaforge augment qa.jsonl -f output -I "Translate to Traditional Chinese (Taiwan)"

# Add difficulty rating (CSV)
ollaforge augment problems.csv -f difficulty --new-field -I "Rate: easy/medium/hard based on complexity"

# Expand brief answers (JSON)
ollaforge augment faq.json -f answer -I "Expand this answer with more detail and examples"

# Add category field using context (TSV)
ollaforge augment articles.tsv -f category --new-field -c title -c content -I "Categorize: tech/science/business/other"

# Convert formats
ollaforge convert data.csv data.parquet  # CSV to Parquet
ollaforge convert data.jsonl data.json   # JSONL to JSON array
```

### Preview Mode

Always preview before processing large datasets:

```bash
ollaforge augment large_dataset.jsonl -f output -I "Improve clarity" --preview
```

This processes 3 sample entries and shows before/after comparison.

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

OllaForge is optimized for local LLM inference:

| Optimization | Benefit |
|--------------|---------|
| **Structured JSON Output** | 0% format errors via Ollama's schema enforcement |
| **Small Batch Size (5)** | Reduces attention decay, improves quality |
| **Concurrent Requests** | Up to 10 parallel batch requests |
| **BERT on CPU** | Keeps GPU/MPS free for LLM generation |
| **Funnel Architecture** | Over-request → Filter → Keep valid entries |

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

```bash
# Enable QC (default for zh-tw)
ollaforge generate "對話" --lang zh-tw --qc

# Stricter threshold
ollaforge generate "對話" --lang zh-tw --qc-confidence 0.95
```

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

## 🏗️ Project Structure

```
ollaforge/
├── ollaforge/              # Core package
│   ├── __init__.py         # Package exports
│   ├── cli.py              # CLI commands (generate, augment)
│   ├── client.py           # Ollama API client
│   ├── augmentor.py        # Dataset augmentation engine
│   ├── processor.py        # Response parsing & validation
│   ├── models.py           # Pydantic data models
│   ├── qc.py               # Taiwan Chinese QC (BERT)
│   ├── progress.py         # Progress tracking
│   ├── file_manager.py     # File I/O operations
│   └── interactive.py      # Interactive wizard
├── tests/                  # Test suite (pytest + hypothesis)
├── docs/                   # Documentation
├── examples/               # Example datasets
├── pyproject.toml          # Project configuration
├── Makefile                # Development commands
└── README.md               # This file
```

---

## 🧪 Development

```bash
# Clone and setup
git clone https://github.com/ollaforge/ollaforge.git
cd ollaforge
pip install -e ".[dev]"

# Run tests
make test

# Run with coverage
make test-cov

# Lint & format
make lint
make format

# Type check
make typecheck

# All checks
make check
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Property-based tests only
pytest tests/ -v -k "property"

# With coverage report
pytest tests/ --cov=ollaforge --cov-report=html
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Start

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Write tests for your changes
4. Ensure all tests pass (`make check`)
5. Submit a Pull Request

### Areas for Contribution

- 🌐 Additional language support
- 📊 New dataset formats
- 🔧 Performance optimizations
- 📚 Documentation improvements
- 🐛 Bug fixes

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM inference
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal UI
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Pydantic](https://pydantic.dev/) - Data validation
- [Hypothesis](https://hypothesis.readthedocs.io/) - Property-based testing

---

## 📊 Star History

<p align="center">
  <a href="https://star-history.com/#ollaforge/ollaforge&Date">
    <img src="https://api.star-history.com/svg?repos=ollaforge/ollaforge&type=Date" alt="Star History Chart">
  </a>
</p>

---

<p align="center">
  <strong>Made with ❤️ by the OllaForge Team</strong>
</p>

<p align="center">
  <a href="https://github.com/ollaforge/ollaforge/issues/new?template=bug_report.md">Report Bug</a> •
  <a href="https://github.com/ollaforge/ollaforge/issues/new?template=feature_request.md">Request Feature</a> •
  <a href="https://github.com/ollaforge/ollaforge/discussions">Discussions</a>
</p>
