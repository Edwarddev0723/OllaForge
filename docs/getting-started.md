# Getting Started with OllaForge

This guide will help you get up and running with OllaForge in minutes.

## Prerequisites

Before installing OllaForge, ensure you have:

- **Python 3.9+** - [Download Python](https://www.python.org/downloads/)
- **Ollama** - [Install Ollama](https://ollama.ai/)

### Verify Ollama is Running

```bash
# Check Ollama is installed
ollama --version

# Start Ollama (if not running)
ollama serve

# Pull a model (if you haven't already)
ollama pull llama3.2
```

## Installation

### From PyPI (Recommended)

```bash
# Basic installation
pip install ollaforge

# With Traditional Chinese QC support
pip install ollaforge[qc]

# With all features (QC + dev tools)
pip install ollaforge[all]
```

### From Source

```bash
git clone https://github.com/ollaforge/ollaforge.git
cd ollaforge
pip install -e .
```

### Verify Installation

```bash
ollaforge --version
ollaforge --help
```

## Your First Dataset

### Interactive Mode (Recommended for Beginners)

```bash
ollaforge -i
```

This launches a wizard that guides you through:
1. Choosing generation or augmentation
2. Selecting dataset format
3. Entering topic/instructions
4. Configuring options

### Command Line Mode

```bash
# Generate 10 SFT entries about Python
ollaforge generate "Python programming tutorials" --count 10

# Generate 100 entries with specific model
ollaforge generate "Customer service conversations" \
    --count 100 \
    --model qwen2.5:14b \
    --output customer_service.jsonl
```

## Dataset Formats

OllaForge supports four dataset formats:

| Format | Flag | Use Case |
|--------|------|----------|
| SFT | `--type sft` | Instruction fine-tuning (Alpaca format) |
| Pre-training | `--type pretrain` | Continued pre-training |
| Conversation | `--type sft_conv` | Multi-turn chat (ShareGPT format) |
| DPO | `--type dpo` | Preference optimization |

### Examples

```bash
# SFT dataset
ollaforge generate "Math word problems" --type sft

# Pre-training corpus
ollaforge generate "Scientific articles about climate change" --type pretrain

# Conversation dataset
ollaforge generate "Technical support dialogues" --type sft_conv

# DPO preference pairs
ollaforge generate "Code review feedback" --type dpo
```

## Augmenting Existing Datasets

OllaForge can enhance existing JSONL datasets:

```bash
# Modify existing field
ollaforge augment data.jsonl --field output --instruction "Make more detailed"

# Add new field
ollaforge augment data.jsonl --field category --new-field --instruction "Categorize: tech/science/other"

# Preview before processing
ollaforge augment data.jsonl --field output --instruction "Translate to Chinese" --preview
```

## Next Steps

- [CLI Reference](cli-reference.md) - Full command documentation
- [Dataset Formats](dataset-formats.md) - Detailed format specifications
- [Augmentation Guide](augmentation-guide.md) - Advanced augmentation techniques
- [FAQ](faq.md) - Common questions and troubleshooting
