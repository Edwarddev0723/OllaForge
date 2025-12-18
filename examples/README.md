# OllaForge Examples

This directory contains example datasets and scripts demonstrating OllaForge capabilities.

## 📁 Directory Structure

```
examples/
├── datasets/           # Sample generated datasets
│   ├── sft_english.jsonl
│   ├── sft_chinese.jsonl
│   ├── conversation.jsonl
│   └── dpo_pairs.jsonl
├── scripts/            # Example scripts
│   ├── batch_generate.sh
│   └── augment_workflow.sh
└── README.md           # This file
```

## 🚀 Quick Examples

### Generate SFT Dataset

```bash
# English programming tutorials
ollaforge generate "Python programming tutorials for beginners" \
    --count 100 \
    --type sft \
    --output examples/datasets/python_sft.jsonl

# Traditional Chinese customer service
ollaforge generate "客服對話範例" \
    --count 50 \
    --type sft \
    --lang zh-tw \
    --output examples/datasets/customer_service_zhtw.jsonl
```

### Generate Conversation Dataset

```bash
# Multi-turn technical support
ollaforge generate "Technical support conversations for software products" \
    --count 100 \
    --type sft_conv \
    --output examples/datasets/tech_support.jsonl
```

### Generate DPO Dataset

```bash
# Code review preference pairs
ollaforge generate "Code review feedback with good and bad examples" \
    --count 50 \
    --type dpo \
    --output examples/datasets/code_review_dpo.jsonl
```

### Augment Existing Dataset

```bash
# Add difficulty ratings
ollaforge augment examples/datasets/python_sft.jsonl \
    --field difficulty \
    --new-field \
    --instruction "Rate difficulty: beginner/intermediate/advanced" \
    --output examples/datasets/python_sft_rated.jsonl

# Translate to Chinese
ollaforge augment examples/datasets/python_sft.jsonl \
    --field output \
    --instruction "Translate to Traditional Chinese (Taiwan)" \
    --output examples/datasets/python_sft_chinese.jsonl
```

## 📊 Sample Dataset Formats

### SFT Format
```json
{"instruction": "Explain what a variable is in Python", "input": "", "output": "A variable in Python is a named container..."}
```

### Conversation Format
```json
{"conversations": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How do I install Python?"}, {"role": "assistant", "content": "You can install Python by..."}]}
```

### DPO Format
```json
{"prompt": "Write a function to calculate factorial", "chosen": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)", "rejected": "def f(x): return x*f(x-1)"}
```

## 🔧 Batch Processing

See `scripts/batch_generate.sh` for examples of generating multiple datasets in batch.

## 📚 More Resources

- [Main Documentation](../README.md)
- [API Reference](../docs/api.md)
- [Contributing Guide](../CONTRIBUTING.md)
