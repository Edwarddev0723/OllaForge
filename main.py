#!/usr/bin/env python3
"""
OllaForge - CLI tool for generating datasets using local Ollama models

This file is kept for backward compatibility.
The main CLI implementation has been moved to ollaforge/cli.py

Usage:
    python main.py [OPTIONS] TOPIC
    
Or install and use directly:
    pip install -e .
    ollaforge [OPTIONS] TOPIC
"""

from ollaforge.cli import app, main

if __name__ == "__main__":
    main()
