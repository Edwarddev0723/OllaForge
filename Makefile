.PHONY: help install install-dev install-all test test-cov lint format typecheck check clean build publish demo

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

# Default target
help:
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║$(RESET)           $(GREEN)OllaForge Development Commands$(RESET)                    $(BLUE)║$(RESET)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(YELLOW)Setup:$(RESET)"
	@echo "  make install       Install package (basic)"
	@echo "  make install-dev   Install with dev dependencies"
	@echo "  make install-all   Install with all dependencies (including QC)"
	@echo ""
	@echo "$(YELLOW)Development:$(RESET)"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make test-fast     Run tests without slow tests"
	@echo "  make lint          Run linter (ruff)"
	@echo "  make lint-fix      Run linter with auto-fix"
	@echo "  make format        Format code (black)"
	@echo "  make typecheck     Run type checker (mypy)"
	@echo "  make check         Run all quality checks"
	@echo ""
	@echo "$(YELLOW)Build & Release:$(RESET)"
	@echo "  make build         Build package"
	@echo "  make clean         Clean build artifacts"
	@echo "  make publish-test  Publish to TestPyPI"
	@echo "  make publish       Publish to PyPI"
	@echo ""
	@echo "$(YELLOW)Demo:$(RESET)"
	@echo "  make demo          Run demo generation (English)"
	@echo "  make demo-zh       Run demo generation (Chinese)"
	@echo "  make demo-augment  Run demo augmentation"

# ============================================================================
# Installation
# ============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

install-qc:
	pip install -e ".[qc]"

# ============================================================================
# Testing
# ============================================================================

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=ollaforge --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(RESET)"

test-fast:
	pytest tests/ -v --tb=short -x -q

test-property:
	pytest tests/ -v -k "property or Property"

# ============================================================================
# Code Quality
# ============================================================================

lint:
	ruff check ollaforge/ tests/

lint-fix:
	ruff check ollaforge/ tests/ --fix

format:
	black ollaforge/ tests/

format-check:
	black ollaforge/ tests/ --check

typecheck:
	mypy ollaforge/ --ignore-missing-imports

# All quality checks
check: format-check lint typecheck test
	@echo "$(GREEN)✓ All checks passed!$(RESET)"

# ============================================================================
# Build & Release
# ============================================================================

build: clean
	python -m build
	@echo "$(GREEN)✓ Build complete! Check dist/ directory$(RESET)"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleaned build artifacts$(RESET)"

publish-test: build
	twine upload --repository testpypi dist/*

publish: build
	twine upload dist/*

# ============================================================================
# Demo Commands
# ============================================================================

demo:
	@echo "$(BLUE)Generating English SFT dataset...$(RESET)"
	ollaforge generate "Python programming tutorials" --count 5 --type sft -o demo_output.jsonl
	@echo "$(GREEN)✓ Generated demo_output.jsonl$(RESET)"

demo-zh:
	@echo "$(BLUE)Generating Chinese conversation dataset...$(RESET)"
	ollaforge generate "咖啡點餐對話" --count 5 --type sft_conv --lang zh-tw -o demo_zh_output.jsonl
	@echo "$(GREEN)✓ Generated demo_zh_output.jsonl$(RESET)"

demo-augment:
	@echo "$(BLUE)Running augmentation demo...$(RESET)"
	@if [ -f demo_output.jsonl ]; then \
		ollaforge augment demo_output.jsonl --field output --instruction "Add more detail" --preview; \
	else \
		echo "$(YELLOW)Run 'make demo' first to generate demo_output.jsonl$(RESET)"; \
	fi

# ============================================================================
# Development Utilities
# ============================================================================

# Show current version
version:
	@python -c "import ollaforge; print(ollaforge.__version__)"

# Update dependencies
update-deps:
	pip install --upgrade pip
	pip install -e ".[dev]" --upgrade

# Generate requirements.txt from pyproject.toml
requirements:
	pip-compile pyproject.toml -o requirements.txt

# Run the CLI help
cli-help:
	ollaforge --help
	@echo ""
	ollaforge generate --help
	@echo ""
	ollaforge augment --help
