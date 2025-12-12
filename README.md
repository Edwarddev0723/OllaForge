# OllaForge 🔥

A Python-based CLI application that leverages local Ollama models (Llama 3, Mistral, etc.) to automatically generate topic-specific datasets and output them in JSONL format.

## Features

- 🎯 **Topic-based Generation**: Describe your dataset needs in natural language
- 🤖 **Multiple Models**: Support for various Ollama models (Llama 3, Mistral, etc.)
- 📊 **Progress Tracking**: Beautiful progress bars and real-time feedback
- 🔧 **Data Validation**: Automatic JSON validation and cleaning
- 📁 **JSONL Output**: Industry-standard format for machine learning datasets
- 🎨 **Rich CLI**: Modern, colorful command-line interface
- 📚 **Multiple Dataset Types**: Support for SFT, Pre-training, Conversation, and DPO formats

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ollaforge
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running locally on port 11434

## Usage

Basic usage:
```bash
python main.py "customer service conversations about refunds"
```

With custom parameters:
```bash
python main.py "technical documentation examples" --count 50 --model gpt-oss:20b --output tech_docs.jsonl
```

### Parameters

- `topic` (required): Description of the dataset content to generate
- `--count, -c`: Number of data entries to generate (default: 10)
- `--model, -m`: Ollama model to use (default: llama3)
- `--output, -o`: Output filename (default: dataset.jsonl)
- `--type, -t`: Dataset type to generate (default: sft)

### Dataset Types

OllaForge supports multiple dataset formats compatible with HuggingFace and LLaMA-Factory:

| Type | Description | Format |
|------|-------------|--------|
| `sft` | Supervised Fine-tuning (Alpaca) | `{"instruction": "...", "input": "...", "output": "..."}` |
| `pretrain` | Pre-training | `{"text": "..."}` |
| `sft_conv` | SFT Conversation (ShareGPT/ChatML) | `{"conversations": [{"role": "user", "content": "..."}, ...]}` |
| `dpo` | Direct Preference Optimization | `{"prompt": "...", "chosen": "...", "rejected": "..."}` |

### Examples

Generate SFT data (default):
```bash
python main.py "customer service conversations" --count 50
```

Generate pre-training data:
```bash
python main.py "machine learning concepts" --type pretrain --count 100
```

Generate multi-turn conversation data:
```bash
python main.py "technical support dialogues" --type sft_conv --output conversations.jsonl
```

Generate DPO preference pairs:
```bash
python main.py "code review feedback" --type dpo --count 50 --output dpo_data.jsonl
```

## Development

Run tests:
```bash
pytest tests/
```

## Requirements

- Python 3.8+
- Local Ollama installation
- Required Python packages (see requirements.txt)

## License

MIT License