# Changelog

All notable changes to OllaForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-16

### Added
- Initial release of OllaForge
- Support for multiple dataset formats:
  - SFT (Supervised Fine-tuning) - Alpaca format
  - Pre-training - Raw text format
  - SFT Conversation - ShareGPT/ChatML format
  - DPO (Direct Preference Optimization)
- Traditional Chinese (Taiwan) support with QC validation
- Structured JSON output using Ollama's format parameter
- Concurrent batch generation for performance
- Interactive wizard mode
- Rich CLI interface with progress tracking

### Performance Optimizations
- JSON Schema structured output (0% format error rate)
- Optimized batch size (5) for better quality on Mac
- BERT QC model forced to CPU to keep GPU free for LLM
- Funnel architecture: over-request and filter

### Quality Features
- Few-shot examples for Taiwan Chinese terminology
- Diverse response patterns to avoid robotic outputs
- Multi-turn conversation scenarios
- Logical accuracy validation
