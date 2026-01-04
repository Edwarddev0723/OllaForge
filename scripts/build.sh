#!/bin/bash
# Build script for OllaForge

set -e

echo "🔨 Building OllaForge package..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install --upgrade build twine

# Build the package
echo "🏗️  Building package..."
python -m build

# Check the built package
echo "🔍 Checking built package..."
python -m twine check dist/*

echo "✅ Build completed successfully!"
echo "📁 Built files:"
ls -la dist/

echo ""
echo "To upload to PyPI:"
echo "  Test PyPI: python -m twine upload --repository testpypi dist/*"
echo "  Production: python -m twine upload dist/*"