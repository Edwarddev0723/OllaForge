"""
Setup script for OllaForge CLI tool.

OllaForge is a Python CLI application for generating high-quality training
datasets using local Ollama models.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "typer[all]>=0.9.0",
        "rich>=13.7.0",
        "ollama>=0.1.7",
        "pydantic>=2.5.0",
    ]

# Optional dependencies
extras_require = {
    "qc": [
        "transformers>=4.36.0",
        "torch>=2.0.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "hypothesis>=6.90.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "ruff>=0.1.0",
        "mypy>=1.0.0",
    ],
}
extras_require["all"] = list(set(
    dep for deps in extras_require.values() for dep in deps
))

setup(
    name="ollaforge",
    version="1.0.0",
    author="OllaForge Team",
    author_email="",
    description="CLI tool for generating datasets using local Ollama models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ollaforge/ollaforge",
    project_urls={
        "Bug Tracker": "https://github.com/ollaforge/ollaforge/issues",
        "Documentation": "https://github.com/ollaforge/ollaforge#readme",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "ollaforge=ollaforge.cli:main",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "ollama",
        "llm",
        "dataset",
        "fine-tuning",
        "sft",
        "dpo",
        "machine-learning",
        "nlp",
        "traditional-chinese",
    ],
    include_package_data=True,
    zip_safe=False,
)
