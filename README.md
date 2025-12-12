<p align="center">
  <img src="img/banner.png" alt="OllaForge Banner" width="100%">
</p>

<h1 align="center">OllaForge 🔥</h1>

<p align="center">
  <strong>AI-Powered Dataset Generator for LLM Training</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#dataset-formats">Dataset Formats</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-TW.md">繁體中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/ollama-local-orange.svg" alt="Ollama">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</p>

---

**OllaForge** is a powerful CLI tool that leverages local Ollama models to automatically generate high-quality, topic-specific datasets for LLM training. Generate SFT, Pre-training, Conversation, and DPO datasets with a single command.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Natural Language Topics** | Describe your dataset needs in plain language |
| 🤖 **Multiple LLM Support** | Works with Llama 3, Mistral, Qwen, DeepSeek, and more |
| 📊 **4 Dataset Formats** | SFT, Pre-training, Conversation (ShareGPT), DPO |
| 🌐 **Multi-language Output** | Generate datasets in English or Traditional Chinese |
| 🎨 **Beautiful CLI** | Interactive wizard with Rich-powered UI |
| ⚡ **Batch Processing** | Efficient generation with configurable concurrency |
| ✅ **Auto Validation** | Built-in JSON validation and error recovery |
| 🔄 **HuggingFace Compatible** | Output formats work with HuggingFace & LLaMA-Factory |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ollaforge.git
cd ollaforge

# Install dependencies
pip install -r requirements.txt

# Verify Ollama is running
ollama list
```

### Your First Dataset

```bash
# Interactive mode (recommended for beginners)
python main.py -i

# Or generate directly
python main.py "Python programming tutorials" --count 100 --output python_sft.jsonl
```

## 📖 Usage

### Interactive Mode

Launch the step-by-step wizard:

```bash
python main.py -i
```

The wizard guides you through:
1. 📝 Topic description
2. 📊 Dataset type selection
3. 🌐 Output language
4. 🔢 Number of entries
5. 🤖 Model selection
6. 📄 Output settings

### Command Line Mode

```bash
python main.py <topic> [options]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--count` | `-c` | 10 | Number of entries to generate |
| `--model` | `-m` | gpt-oss:20b | Ollama model to use |
| `--output` | `-o` | dataset.jsonl | Output filename |
| `--type` | `-t` | sft | Dataset type (sft/pretrain/sft_conv/dpo) |
| `--lang` | `-l` | en | Output language (en/zh-tw) |
| `--qc/--no-qc` | - | --qc | Enable/disable Taiwan Chinese QC |
| `--qc-confidence` | - | 0.9 | QC confidence threshold (0.0-1.0) |
| `--concurrency` | `-j` | 5 | Parallel requests (1-20) |
| `--interactive` | `-i` | - | Launch interactive mode |

#### Examples


```bash
# Generate SFT training data
python main.py "customer service conversations" --count 500 --type sft

# Generate pre-training corpus
python main.py "machine learning research papers" --type pretrain --count 1000

# Generate multi-turn conversations
python main.py "technical support dialogues" --type sft_conv --output conversations.jsonl

# Generate DPO preference pairs
python main.py "code review feedback" --type dpo --count 200

# Generate in Traditional Chinese
python main.py "客服對話範例" --lang zh-tw --count 100 --output zh_dataset.jsonl

# Use a specific model
python main.py "medical Q&A" --model deepseek-r1:14b --count 50
```

## 📋 Dataset Formats

OllaForge generates datasets compatible with **HuggingFace** and **LLaMA-Factory**.

### SFT (Supervised Fine-tuning)

Alpaca-style format for instruction tuning:

```json
{
  "instruction": "Explain the concept of recursion",
  "input": "I'm learning programming",
  "output": "Recursion is a programming technique where a function calls itself..."
}
```

### Pre-training

Raw text format for continued pre-training:

```json
{
  "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience..."
}
```

### SFT Conversation (ShareGPT/ChatML)

Multi-turn dialogue format:

```json
{
  "conversations": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I reverse a string in Python?"},
    {"role": "assistant", "content": "You can reverse a string using slicing: `reversed_string = original[::-1]`"},
    {"role": "user", "content": "What about using a loop?"},
    {"role": "assistant", "content": "Here's how to do it with a loop:\n```python\ndef reverse_string(s):\n    result = ''\n    for char in s:\n        result = char + result\n    return result\n```"}
  ]
}
```

### DPO (Direct Preference Optimization)

Preference pairs for RLHF training:

```json
{
  "prompt": "Write a function to calculate factorial",
  "chosen": "Here's an efficient recursive implementation with proper error handling:\n```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError('Factorial is not defined for negative numbers')\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```",
  "rejected": "def f(n): return n*f(n-1) if n else 1"
}
```

## 🌐 Supported Languages

| Code | Language | Example |
|------|----------|---------|
| `en` | English | `--lang en` |
| `zh-tw` | 繁體中文（台灣） | `--lang zh-tw` |

## 🔍 Quality Control (QC) for Traditional Chinese

When generating datasets in Traditional Chinese (`--lang zh-tw`), OllaForge includes an optional **Quality Control** system that automatically filters out Mainland Chinese expressions.

### How It Works

- Uses a BERT-based classifier ([renhehuang/bert-traditional-chinese-classifier](https://huggingface.co/renhehuang/bert-traditional-chinese-classifier))
- Classifies text as "Taiwan Traditional" or "Mainland Traditional"
- Entries with Mainland expressions are automatically regenerated
- Default confidence threshold: 90%

### Usage

```bash
# Enable QC (default when using zh-tw)
python main.py "客服對話" --lang zh-tw --qc

# Disable QC
python main.py "客服對話" --lang zh-tw --no-qc

# Adjust confidence threshold (stricter)
python main.py "客服對話" --lang zh-tw --qc-confidence 0.95
```

### Examples of Filtered Expressions

| Mainland (Filtered) | Taiwan (Accepted) |
|---------------------|-------------------|
| 軟件 | 軟體 |
| 程序 | 程式 |
| 計算機 | 電腦 |
| 網絡 | 網路 |
| 界面 | 介面 |

## 🤖 Recommended Models

| Model | Size | Best For |
|-------|------|----------|
| `gpt-oss:20b` | 20B | General purpose (default) |
| `deepseek-r1:14b` | 14B | Reasoning & complex tasks |
| `qwen3:14b` | 14B | Multilingual support |
| `ministral-3:14b` | 14B | Edge deployment |
| `gemma3:12b` | 12B | Single GPU efficiency |

## 🏗️ Architecture

```
ollaforge/
├── main.py              # CLI entry point
├── ollaforge/
│   ├── client.py        # Ollama API communication
│   ├── processor.py     # Response parsing & validation
│   ├── models.py        # Pydantic data models
│   ├── interactive.py   # Rich-based interactive UI
│   ├── progress.py      # Progress tracking
│   └── file_manager.py  # File I/O operations
└── tests/               # Comprehensive test suite
```

## 🧪 Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ollaforge

# Type checking
mypy ollaforge/
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔃 Open a Pull Request

### Areas We Need Help

- [ ] Additional language support (Japanese, Korean, etc.)
- [ ] More dataset format templates
- [ ] Performance optimizations
- [ ] Documentation improvements
- [ ] Test coverage expansion

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for making local LLMs accessible
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [Typer](https://typer.tiangolo.com/) for elegant CLI creation
- [Pydantic](https://pydantic.dev/) for data validation

---

<p align="center">
  Made with ❤️ by the OllaForge Team
</p>

<p align="center">
  <a href="https://github.com/yourusername/ollaforge/stargazers">⭐ Star us on GitHub</a>
</p>
