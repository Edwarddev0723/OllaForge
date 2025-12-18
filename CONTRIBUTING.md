# Contributing to OllaForge

First off, thank you for considering contributing to OllaForge! 🎉

This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, Ollama version)
- **Error messages** or logs if applicable
- **Sample data** if relevant (anonymized)

### 💡 Suggesting Features

Feature suggestions are welcome! Please include:

- **Clear description** of the feature
- **Use case** explaining why this would be useful
- **Possible implementation** if you have ideas
- **Examples** of similar features in other tools

### 🔧 Pull Requests

1. Fork the repo and create your branch from `main`
2. Add tests if you've added code
3. Ensure the test suite passes
4. Update documentation as needed
5. Submit your PR!

## Development Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) (for integration testing)
- Git

### Setup Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/ollaforge.git
cd ollaforge

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install

# 5. Verify setup
make check
```

### Using Make Commands

```bash
make install-dev  # Install dev dependencies
make test         # Run tests
make test-cov     # Run tests with coverage
make lint         # Run linting
make format       # Format code
make typecheck    # Run type checking
make check        # Run all checks
make clean        # Clean build artifacts
```

## Project Structure

```
ollaforge/
├── ollaforge/              # Main package
│   ├── __init__.py         # Package exports
│   ├── cli.py              # CLI implementation
│   ├── client.py           # Ollama API client
│   ├── augmentor.py        # Dataset augmentation
│   ├── processor.py        # Response processing
│   ├── models.py           # Pydantic models
│   ├── qc.py               # Quality control
│   ├── progress.py         # Progress tracking
│   ├── file_manager.py     # File operations
│   └── interactive.py      # Interactive wizard
├── tests/                  # Test suite
│   ├── test_*.py           # Test modules
│   └── conftest.py         # Pytest fixtures
├── docs/                   # Documentation
├── examples/               # Example datasets
├── .github/                # GitHub templates & workflows
├── pyproject.toml          # Project configuration
├── Makefile                # Development commands
└── README.md               # Main documentation
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_augmentor.py -v

# Run tests matching pattern
pytest tests/ -v -k "augment"

# Run with coverage
pytest tests/ --cov=ollaforge --cov-report=html
open htmlcov/index.html
```

### Property-Based Testing

We use [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. When adding new features:

1. Identify correctness properties that should hold
2. Write property tests that verify these properties
3. Tag tests with the feature and property number

Example:
```python
@given(data=st.dictionaries(st.text(), st.text()))
@settings(max_examples=100)
def test_json_round_trip(data):
    """
    **Feature: my-feature, Property 1: Round-trip consistency**
    **Validates: Requirements 1.1**
    """
    serialized = json.dumps(data)
    deserialized = json.loads(serialized)
    assert data == deserialized
```

### Test Categories

- **Unit tests**: Test individual functions/methods
- **Property tests**: Verify properties hold across many inputs
- **Integration tests**: Test component interactions

## Pull Request Process

### Before Submitting

1. **Update tests**: Add tests for new functionality
2. **Run checks**: `make check` should pass
3. **Update docs**: Update README, docstrings, etc.
4. **Update CHANGELOG**: Add entry under `[Unreleased]`

### PR Guidelines

- Use a clear, descriptive title
- Reference related issues (`Fixes #123`)
- Describe what changes were made and why
- Include screenshots for UI changes
- Keep PRs focused - one feature/fix per PR

### Review Process

1. Automated checks must pass
2. At least one maintainer approval required
3. All conversations must be resolved
4. Squash and merge preferred

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with these tools:

- **Black**: Code formatting (line length 88)
- **Ruff**: Linting
- **MyPy**: Type checking

```bash
# Format code
black ollaforge/ tests/

# Lint code
ruff check ollaforge/ tests/

# Type check
mypy ollaforge/
```

### Code Conventions

- Use type hints for all public functions
- Write docstrings for public functions and classes
- Keep functions focused and small
- Prefer explicit over implicit
- Handle errors gracefully

### Docstring Format

```python
def process_entry(entry: Dict[str, Any], config: Config) -> Result:
    """
    Process a single dataset entry.
    
    Args:
        entry: The entry to process
        config: Processing configuration
        
    Returns:
        Processing result with status and data
        
    Raises:
        ValidationError: If entry is invalid
    """
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs when relevant

Example:
```
Add dataset augmentation preview mode

- Add --preview flag to augment command
- Process configurable number of sample entries
- Display before/after comparison

Closes #42
```

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create GitHub release with tag
4. CI automatically publishes to PyPI

## 🙏 Thank You!

Your contributions make OllaForge better for everyone. We appreciate your time and effort!

### Contributors

Thanks to all our contributors! 

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

Questions? Open a [Discussion](https://github.com/ollaforge/ollaforge/discussions) or reach out to the maintainers.
