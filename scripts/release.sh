#!/bin/bash
# Release script for OllaForge

set -e

VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")

echo "🚀 Releasing OllaForge v$VERSION"

# Ensure we're on main branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo "❌ Must be on main branch to release. Current branch: $BRANCH"
    exit 1
fi

# Ensure working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ Working directory is not clean. Please commit or stash changes."
    git status --short
    exit 1
fi

# Run tests
echo "🧪 Running tests..."
make test

# Build package
echo "🔨 Building package..."
./scripts/build.sh

# Create git tag
echo "🏷️  Creating git tag v$VERSION..."
git tag -a "v$VERSION" -m "Release v$VERSION"

# Push tag
echo "📤 Pushing tag to GitHub..."
git push origin "v$VERSION"

echo "✅ Release v$VERSION completed!"
echo ""
echo "Next steps:"
echo "1. Upload to PyPI: python -m twine upload dist/*"
echo "2. Create GitHub release at: https://github.com/ollaforge/ollaforge/releases/new?tag=v$VERSION"