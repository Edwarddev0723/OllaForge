<p align="center">
  <img src="img/banner.png" alt="OllaForge Banner" width="100%">
</p>

<h1 align="center">OllaForge 🔥</h1>

<p align="center">
  <strong>AI-Powered Dataset Generator for LLM Fine-tuning</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#dataset-formats">Formats</a> •
  <a href="#performance">Performance</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-TW.md">繁體中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/ollama-local-orange.svg" alt="Ollama">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</p>

---

**OllaForge** is a high-performance CLI tool that leverages local Ollama models to generate training datasets for LLM fine-tuning. With structured JSON output, concurrent batch processing, and built-in quality control for Traditional Chinese, it's optimized for both quality and speed.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Natural Language Topics** | Describe your dataset needs in plain language |
| 🤖 **Any Ollama Model** | Works with Llama 3, Mistral, Qwen, DeepSeek, Gemma, and more |
| 📊 **4 Dataset Formats** | SFT, Pre-training, Conversation (ShareGPT), DPO |
| 🌐 **Multi-language** | English and Traditional Chinese (Taiwan) with QC |
| ⚡ **High Performance** | Structured output + concurrent batching |
| 🔍 **Quality Control** | BERT-based filtering for Taiwan Chinese terminology |
| 🎨 **Beautiful CLI** | Interactive wizard with Rich-powered UI |
| 🔄 **HuggingFace Ready** | Compatible with HuggingFace & LLaMA-Factory |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ollaforge.git
cd ollaforge

# Install (basic)
pip install -e .

# Install with QC support for Traditional Chinese
pip install -e ".[qc]"

# Install with dev tools
pip install -e ".[dev]"
```

### Your First Dataset

```bash
# Interactive mode (recommended for beginners)
ollaforge -i

# Or generate directly
ollaforge "Python programming tutorials" --count 100 --output python_sft.jsonl

# Traditional Chinese conversation dataset
ollaforge "咖啡點餐對話" --type sft_conv --lang zh-tw --count 100
```

## 📖 Usage

### Command Line

```bash
ollaforge <topic> [options]
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
| `--qc-confidence` | | 0.9 | QC threshold (0.0-1.0) |
| `--interactive` | `-i` | | Launch wizard mode |

### Examples

```bash
# SFT instruction-following data
ollaforge "customer service conversations" --count 500 --type sft

# Pre-training corpus
ollaforge "machine learning concepts" --type pretrain --count 1000

# Multi-turn conversations (ShareGPT format)
ollaforge "technical support dialogues" --type sft_conv -o conversations.jsonl

# DPO preference pairs
ollaforge "code review feedback" --type dpo --count 200

# Traditional Chinese with QC
ollaforge "客服對話範例" --lang zh-tw --count 100 --qc-confidence 0.85

# Use specific model with high concurrency
ollaforge "medical Q&A" --model qwen2.5:14b --count 500 -j 10
```

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

## ⚡ Performance Optimizations

OllaForge is optimized for Mac (Apple Silicon) and local LLM inference:

| Optimization | Benefit |
|--------------|---------|
| **Structured JSON Output** | 0% format errors via Ollama's schema enforcement |
| **Small Batch Size (5)** | Reduces attention decay, improves quality |
| **Concurrent Requests** | Up to 10 parallel batch requests |
| **BERT on CPU** | Keeps GPU/MPS free for LLM generation |
| **Funnel Architecture** | Over-request → Filter → Keep valid entries |

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Prompt    │────▶│  Ollama API  │────▶│   JSON      │
│  Engineering│     │  (Parallel)  │     │   Schema    │
└─────────────┘     └──────────────┘     └──────────────┘
                                                │
                    ┌──────────────┐            ▼
                    │   QC Filter  │◀────┌─────────────┐
                    │  (CPU BERT)  │     │  Processor  │
                    └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   JSONL     │
                    │   Output    │
                    └─────────────┘
```

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
# Enable QC (default)
ollaforge "對話" --lang zh-tw --qc

# Stricter threshold
ollaforge "對話" --lang zh-tw --qc-confidence 0.95

# Disable QC
ollaforge "對話" --lang zh-tw --no-qc
```

## 🤖 Recommended Models

| Model | Best For |
|-------|----------|
| `llama3.2` | General purpose (default) |
| `qwen2.5:14b` | Multilingual, Chinese |
| `deepseek-r1:14b` | Reasoning tasks |
| `gemma2:9b` | Efficient, single GPU |
| `mistral:7b` | Fast inference |

## 🏗️ Project Structure

```
ollaforge/
├── ollaforge/
│   ├── __init__.py      # Package exports
│   ├── cli.py           # CLI implementation
│   ├── client.py        # Ollama API + JSON schema
│   ├── processor.py     # Response parsing
│   ├── models.py        # Pydantic models
│   ├── qc.py            # Taiwan Chinese QC
│   ├── progress.py      # Progress tracking
│   └── file_manager.py  # File I/O
├── tests/               # Test suite
├── pyproject.toml       # Project config
└── Makefile             # Dev commands
```

## 🧪 Development

```bash
# Install dev dependencies
make install-dev

# Run tests
make test

# Lint & format
make lint
make format

# Type check
make typecheck

# All checks
make check
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM inference
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal UI
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Pydantic](https://pydantic.dev/) - Data validation

---

<p align="center">
  Made with ❤️ by the OllaForge Team
</p>
