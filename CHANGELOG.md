# Changelog

All notable changes to OllaForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2024-12-18

### Added

- **Dataset Augmentation** - New `augment` command for enhancing existing datasets
  - Modify existing fields with AI-powered transformations
  - Create new computed fields based on existing data
  - Preview mode to test augmentation on samples before full processing
  - Context fields support for multi-field reasoning
  - Concurrent processing for high throughput
  - Graceful failure handling that preserves original data

- **Property-Based Testing** - Comprehensive test suite using Hypothesis
  - 11 correctness properties for augmentation feature
  - Round-trip consistency tests for JSON serialization
  - Field validation tests
  - Concurrent processing correctness tests

- **Improved Documentation**
  - New augmentation guide
  - Updated README with augmentation examples
  - API reference documentation

### Changed

- CLI restructured with subcommands (`generate`, `augment`)
- Improved error messages with line numbers for JSONL parsing errors
- Enhanced progress tracking for augmentation operations

### Fixed

- JSONL parsing now reports exact line numbers for errors
- Better handling of Unicode content in datasets

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

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.1.0 | 2024-12-18 | Dataset augmentation, property-based testing |
| 1.0.0 | 2024-12-16 | Initial release |

[Unreleased]: https://github.com/ollaforge/ollaforge/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ollaforge/ollaforge/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ollaforge/ollaforge/releases/tag/v1.0.0
