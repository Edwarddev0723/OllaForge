.PHONY: help install install-dev install-all test lint format clean build publish

# Default target
help:
	@echo "OllaForge Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install package (basic)"
	@echo "  make install-dev  Install with dev dependencies"
	@echo "  make install-all  Install with all dependencies (including QC)"
	@echo ""
	@echo "Development:"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make format       Format code (black)"
	@echo "  make typecheck    Run type checker (mypy)"
	@echo ""
	@echo "Build:"
	@echo "  make build        Build package"
	@echo "  make clean        Clean build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  make demo         Run demo generation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

install-qc:
	pip install -e ".[qc]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=ollaforge --cov-report=html

# Code Quality
lint:
	ruff check ollaforge/ tests/

lint-fix:
	ruff check ollaforge/ tests/ --fix

format:
	black ollaforge/ tests/

format-check:
	black ollaforge/ tests/ --check

typecheck:
	mypy ollaforge/

# All quality checks
check: lint format-check typecheck test

# Build
build: clean
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Demo
demo:
	ollaforge "Python programming tutorials" --count 5 --type sft -o demo_output.jsonl

demo-zh:
	ollaforge "咖啡點餐對話" --count 5 --type sft_conv --lang zh-tw -o demo_zh_output.jsonl

# Development server (for testing)
dev:
	python -m ollaforge.cli --help
