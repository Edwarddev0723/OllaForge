# Contributing to OllaForge

First off, thank you for considering contributing to OllaForge! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, Ollama version)
- **Error messages** or logs if applicable

### 💡 Suggesting Features

Feature suggestions are welcome! Please include:

- **Clear description** of the feature
- **Use case** explaining why this would be useful
- **Possible implementation** if you have ideas

### 🔧 Pull Requests

1. Fork the repo and create your branch from `main`
2. Add tests if you've added code
3. Ensure the test suite passes
4. Update documentation as needed
5. Submit your PR!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ollaforge.git
cd ollaforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
pytest tests/ -v
```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the version numbers following [SemVer](https://semver.org/)
3. Your PR will be merged once you have approval from maintainers

## Style Guidelines

### Python Code Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints where possible
- Write docstrings for public functions
- Keep functions focused and small

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs when relevant

### Example

```
Add Traditional Chinese support for dataset generation

- Add OutputLanguage enum with zh-tw option
- Update prompts to include language instructions
- Add language selection to interactive wizard

Closes #123
```

## 🙏 Thank You!

Your contributions make OllaForge better for everyone. We appreciate your time and effort!
