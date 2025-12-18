# OllaForge Project Structure

This document describes the complete project structure of OllaForge after optimization for international open-source standards.

## 📁 Directory Structure

```
ollaforge/
├── .github/                    # GitHub-specific files
│   ├── ISSUE_TEMPLATE/         # Issue templates
│   │   ├── bug_report.md       # Bug report template
│   │   └── feature_request.md  # Feature request template
│   ├── workflows/              # GitHub Actions workflows
│   │   ├── ci.yml              # Continuous Integration
│   │   └── release.yml         # Automated releases to PyPI
│   └── PULL_REQUEST_TEMPLATE.md # Pull request template
├── docs/                       # Documentation
│   ├── README.md               # Documentation index
│   └── getting-started.md      # Getting started guide
├── examples/                   # Example datasets and scripts
│   ├── datasets/               # Sample generated datasets
│   │   ├── coffee_order_zhtw.jsonl
│   │   ├── dnd.jsonl
│   │   ├── dnd_test.jsonl
│   │   └── test.jsonl
│   ├── eng_demodataset.jsonl   # English demo dataset
│   ├── zhtw_demodataset.jsonl  # Chinese demo dataset
│   └── README.md               # Examples documentation
├── img/                        # Images and assets
│   └── banner.png              # Project banner
├── ollaforge/                  # Main package
│   ├── __init__.py             # Package initialization and exports
│   ├── __main__.py             # Entry point for python -m ollaforge
│   ├── augmentor.py            # Dataset augmentation engine
│   ├── cli.py                  # CLI implementation (generate, augment)
│   ├── client.py               # Ollama API client
│   ├── file_manager.py         # File I/O operations
│   ├── interactive.py          # Interactive wizard
│   ├── models.py               # Pydantic data models
│   ├── processor.py            # Response parsing & validation
│   ├── progress.py             # Progress tracking
│   └── qc.py                   # Quality control (Taiwan Chinese)
├── tests/                      # Test suite
│   ├── __init__.py             # Test package initialization
│   ├── test_augmentor.py       # Augmentation tests
│   ├── test_cli.py             # CLI tests
│   ├── test_client.py          # Client tests
│   ├── test_file_manager.py    # File operations tests
│   ├── test_models.py          # Model validation tests
│   ├── test_processor.py       # Processing tests
│   └── test_progress.py        # Progress tracking tests
├── .gitignore                  # Git ignore rules
├── CHANGELOG.md                # Version history
├── CODE_OF_CONDUCT.md          # Community guidelines
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
├── Makefile                    # Development commands
├── pyproject.toml              # Project configuration
├── README.md                   # Main documentation (English)
├── README_zh-TW.md             # Chinese documentation
├── requirements.txt            # Python dependencies
└── SECURITY.md                 # Security policy
```

## 🏗️ Architecture Overview

### Core Components

1. **CLI Layer** (`cli.py`)
   - Main entry point with subcommands
   - Parameter validation
   - Interactive mode routing

2. **Generation Engine** (`client.py`)
   - Ollama API integration
   - Concurrent batch processing
   - Structured JSON output

3. **Augmentation Engine** (`augmentor.py`)
   - Dataset enhancement
   - Field modification/creation
   - Preview functionality

4. **Processing Layer** (`processor.py`)
   - Response parsing
   - JSON validation
   - Error recovery

5. **File Management** (`file_manager.py`)
   - JSONL operations
   - Disk space checking
   - Interruption handling

6. **Quality Control** (`qc.py`)
   - Taiwan Chinese validation
   - BERT-based filtering

### Data Models (`models.py`)

- `GenerationConfig` - Generation parameters
- `AugmentationConfig` - Augmentation parameters
- `DataEntry` - SFT format entries
- `PretrainEntry` - Pre-training format
- `SFTConversationEntry` - Conversation format
- `DPOEntry` - DPO format

## 🧪 Testing Strategy

### Test Coverage

- **Property-Based Tests** (Hypothesis) - 11 correctness properties
- **Unit Tests** - Individual component testing
- **Integration Tests** - End-to-end workflows
- **CLI Tests** - Command-line interface validation

### Test Categories

1. **Dataset Augmentation** (`test_augmentor.py`)
   - Property 5: Prompt Contains Context and Instruction
   - Property 6: New Field Creation
   - Property 7: Successful Response Updates Target Field
   - Property 8: Failure Preserves Original Entry
   - Property 9: Concurrent Processing Correctness
   - Property 11: Preview Count Correctness

2. **File Operations** (`test_file_manager.py`)
   - Property 1: JSON Round-Trip Consistency
   - Property 2: Invalid JSONL Error Reporting
   - Disk space validation
   - Unicode handling
   - Interruption recovery

3. **Model Validation** (`test_models.py`)
   - Property 3: Field Validation - Existing Field Accepted
   - Property 4: Field Validation - Non-Existing Field Rejected
   - Property 10: Statistics Accuracy

## 🚀 Development Workflow

### Setup

```bash
git clone https://github.com/ollaforge/ollaforge.git
cd ollaforge
pip install -e ".[dev]"
```

### Commands

```bash
make test          # Run all tests
make test-cov      # Run with coverage
make lint          # Code linting
make format        # Code formatting
make check         # All quality checks
make build         # Build package
```

### CI/CD Pipeline

1. **Pull Request Checks**
   - Linting (Ruff)
   - Type checking (MyPy)
   - Code formatting (Black)
   - Test suite (pytest + Hypothesis)
   - Coverage reporting (Codecov)

2. **Release Process**
   - Automated PyPI publishing
   - GitHub release creation
   - Version tagging

## 📦 Distribution

### PyPI Package

- **Name**: `ollaforge`
- **Entry Point**: `ollaforge = ollaforge.cli:app`
- **Optional Dependencies**:
  - `[qc]` - Traditional Chinese QC support
  - `[dev]` - Development tools
  - `[all]` - All features

### Installation Options

```bash
pip install ollaforge           # Basic
pip install ollaforge[qc]       # With QC
pip install ollaforge[all]      # Everything
```

## 🌐 Internationalization

### Supported Languages

- **English** - Primary documentation and interface
- **Traditional Chinese (Taiwan)** - Full localization with QC

### Documentation

- `README.md` - English
- `README_zh-TW.md` - Traditional Chinese
- Inline help text in both languages

## 📊 Quality Metrics

### Code Quality

- **Type Coverage**: 95%+ with MyPy
- **Test Coverage**: 90%+ with pytest-cov
- **Linting**: Ruff with strict rules
- **Formatting**: Black with 88-character line length

### Testing Metrics

- **Property Tests**: 11 correctness properties
- **Unit Tests**: 52 test functions
- **Integration Tests**: CLI and end-to-end workflows
- **Hypothesis Examples**: 100+ per property test

## 🔧 Configuration Files

### `pyproject.toml`
- Project metadata
- Dependencies
- Tool configurations (Black, Ruff, MyPy, pytest)
- Build system setup

### `Makefile`
- Development commands
- Quality checks
- Build and release automation

### `.github/workflows/`
- CI/CD pipeline definitions
- Automated testing and deployment

This structure follows modern Python packaging standards and international open-source best practices.