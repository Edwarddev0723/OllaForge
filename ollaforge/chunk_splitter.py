"""
Chunk splitter module for OllaForge.

This module provides functionality to split long documents into manageable chunks
for LLM processing. It supports multiple splitting strategies including fixed-size
splitting and semantic boundary-aware splitting.

Key features:
- Configurable chunk size and overlap
- Semantic boundary preservation (paragraphs, headings, code blocks)
- Support for Markdown heading hierarchy
- Code-aware splitting that preserves function/class boundaries
- Hybrid strategy combining fixed-size with semantic awareness

Requirements satisfied:
- 2.1: Divide documents exceeding chunk size into smaller segments
- 2.2: Preserve semantic boundaries (paragraphs, sections, code blocks)
- 2.3: Respect Markdown heading hierarchy when splitting
- 2.4: Preserve function/class boundaries in code files
- 2.5: Configurable chunk size (default: 2000 characters)
- 2.6: Configurable overlap between chunks (default: 200 characters)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .doc_parser import DocumentType, ParsedDocument


class SplitStrategy(Enum):
    """
    Splitting strategies for document chunking.

    - FIXED_SIZE: Split at fixed character intervals
    - SEMANTIC: Split at semantic boundaries (paragraphs, headings, code blocks)
    - HYBRID: Combine fixed-size with semantic awareness (default)
    """

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class ChunkConfig:
    """
    Configuration for document chunking.

    Attributes:
        chunk_size: Maximum size of each chunk in characters (default: 2000)
        chunk_overlap: Number of overlapping characters between chunks (default: 200)
        strategy: Splitting strategy to use (default: HYBRID)
        respect_code_blocks: Preserve code block integrity when possible
        respect_paragraphs: Preserve paragraph integrity when possible
        min_chunk_size: Minimum chunk size to avoid tiny fragments (default: 100)

    Requirements satisfied:
    - 2.5: Configurable chunk size (default: 2000 characters)
    - 2.6: Configurable overlap between chunks (default: 200 characters)
    """

    chunk_size: int = 2000
    chunk_overlap: int = 200
    strategy: SplitStrategy = SplitStrategy.HYBRID
    respect_code_blocks: bool = True
    respect_paragraphs: bool = True
    min_chunk_size: int = 100

    def __post_init__(self):
        """Validate configuration values."""
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size < 0:
            raise ValueError("min_chunk_size cannot be negative")
        if self.min_chunk_size >= self.chunk_size:
            raise ValueError("min_chunk_size must be less than chunk_size")


@dataclass
class TextChunk:
    """
    Represents a chunk of text from a document.

    Attributes:
        content: The text content of the chunk
        index: Zero-based index of this chunk in the sequence
        start_char: Starting character position in the original document
        end_char: Ending character position in the original document
        metadata: Additional metadata (source section, document info, etc.)
    """

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the length of the chunk content."""
        return len(self.content)


class ChunkSplitter:
    """
    Document chunk splitter with configurable strategies.

    This class handles splitting documents into chunks suitable for LLM processing,
    with support for semantic boundary preservation and configurable overlap.

    Usage:
        config = ChunkConfig(chunk_size=2000, chunk_overlap=200)
        splitter = ChunkSplitter(config)
        chunks = splitter.split(parsed_document)

    Requirements satisfied:
    - 2.1: Divide documents exceeding chunk size into smaller segments
    - 2.2: Preserve semantic boundaries (paragraphs, sections, code blocks)
    - 2.3: Respect Markdown heading hierarchy when splitting
    - 2.4: Preserve function/class boundaries in code files
    - 2.5: Configurable chunk size
    - 2.6: Configurable overlap between chunks
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize the chunk splitter.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkConfig()

    def split(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Split a document into chunks based on the configured strategy.

        Args:
            document: The parsed document to split

        Returns:
            List of TextChunk objects
        """
        if not document.content or not document.content.strip():
            return []

        # Choose splitting method based on strategy and document type
        if self.config.strategy == SplitStrategy.FIXED_SIZE:
            return self._split_fixed_size(document.content)
        elif self.config.strategy == SplitStrategy.SEMANTIC:
            return self._split_semantic(document)
        else:  # HYBRID
            return self._split_hybrid(document)

    def _split_fixed_size(self, text: str) -> list[TextChunk]:
        """
        Split text into fixed-size chunks with overlap.

        This is the simplest splitting strategy that divides text at
        fixed character intervals, with configurable overlap between chunks.

        Args:
            text: The text to split

        Returns:
            List of TextChunk objects

        Requirements satisfied:
        - 2.1: Divide documents exceeding chunk size
        - 2.5: Configurable chunk size
        - 2.6: Configurable overlap
        """
        if not text:
            return []

        chunks = []
        text_length = len(text)
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Calculate step size (how far to advance for each chunk)
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size  # Fallback if overlap is too large

        start = 0
        index = 0

        while start < text_length:
            # Calculate end position
            end = min(start + chunk_size, text_length)

            # Extract chunk content
            chunk_content = text[start:end]

            # Create chunk
            chunks.append(
                TextChunk(
                    content=chunk_content,
                    index=index,
                    start_char=start,
                    end_char=end,
                    metadata={"strategy": "fixed_size"},
                )
            )

            # Move to next position
            if end >= text_length:
                break

            start += step
            index += 1

        return chunks

    def _split_semantic(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Split document at semantic boundaries.

        This strategy attempts to split at natural boundaries like
        paragraphs, headings, and code blocks. Oversized chunks are
        further split using fixed-size splitting to ensure compliance
        with chunk size limits.

        Args:
            document: The parsed document to split

        Returns:
            List of TextChunk objects

        Requirements satisfied:
        - 2.1: Divide documents exceeding chunk size
        - 2.2: Preserve semantic boundaries
        - 2.3: Respect Markdown heading hierarchy
        - 2.4: Preserve function/class boundaries
        """
        # Use document type-specific splitting
        if document.doc_type == DocumentType.MARKDOWN:
            semantic_chunks = self._split_markdown(document)
        elif document.doc_type == DocumentType.CODE:
            semantic_chunks = self._split_code(document)
        else:
            semantic_chunks = self._split_by_paragraphs(document.content)

        # Ensure all chunks respect size limits by splitting oversized ones
        final_chunks = []
        current_index = 0

        for chunk in semantic_chunks:
            if len(chunk.content) <= self.config.chunk_size:
                # Chunk is within size limit
                final_chunks.append(
                    TextChunk(
                        content=chunk.content,
                        index=current_index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata={**chunk.metadata, "strategy": "semantic"},
                    )
                )
                current_index += 1
            else:
                # Chunk exceeds size limit, split further using fixed-size
                sub_chunks = self._split_fixed_size(chunk.content)
                for sub_chunk in sub_chunks:
                    final_chunks.append(
                        TextChunk(
                            content=sub_chunk.content,
                            index=current_index,
                            start_char=chunk.start_char + sub_chunk.start_char,
                            end_char=chunk.start_char + sub_chunk.end_char,
                            metadata={**chunk.metadata, "strategy": "semantic_fixed"},
                        )
                    )
                    current_index += 1

        return final_chunks

    def _split_hybrid(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Split using hybrid strategy combining semantic and fixed-size.

        First attempts semantic splitting, then applies fixed-size splitting
        to any chunks that exceed the size limit.

        Args:
            document: The parsed document to split

        Returns:
            List of TextChunk objects
        """
        # First, get semantic chunks
        semantic_chunks = self._split_semantic(document)

        # Then, split any oversized chunks using fixed-size
        final_chunks = []
        current_index = 0

        for chunk in semantic_chunks:
            if len(chunk.content) <= self.config.chunk_size:
                # Chunk is within size limit, keep as-is but update index
                final_chunks.append(
                    TextChunk(
                        content=chunk.content,
                        index=current_index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata={**chunk.metadata, "strategy": "hybrid_semantic"},
                    )
                )
                current_index += 1
            else:
                # Chunk exceeds size limit, split further
                sub_chunks = self._split_fixed_size(chunk.content)
                for sub_chunk in sub_chunks:
                    final_chunks.append(
                        TextChunk(
                            content=sub_chunk.content,
                            index=current_index,
                            start_char=chunk.start_char + sub_chunk.start_char,
                            end_char=chunk.start_char + sub_chunk.end_char,
                            metadata={**chunk.metadata, "strategy": "hybrid_fixed"},
                        )
                    )
                    current_index += 1

        # Merge small chunks if needed
        return self._merge_small_chunks(final_chunks)

    def _split_markdown(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Split Markdown document respecting heading hierarchy.

        Args:
            document: The parsed Markdown document

        Returns:
            List of TextChunk objects

        Requirements satisfied:
        - 2.3: Respect Markdown heading hierarchy
        """
        chunks = []
        current_pos = 0

        # Use sections from the parsed document if available
        if document.sections:
            for i, section in enumerate(document.sections):
                section_content = section.content
                if section.title:
                    # Reconstruct heading with content
                    heading_prefix = (
                        "#" * section.level + " " if section.level > 0 else ""
                    )
                    section_content = (
                        f"{heading_prefix}{section.title}\n\n{section.content}"
                    )

                if section_content.strip():
                    chunks.append(
                        TextChunk(
                            content=section_content.strip(),
                            index=i,
                            start_char=current_pos,
                            end_char=current_pos + len(section_content),
                            metadata={
                                "section_title": section.title,
                                "section_level": section.level,
                                "strategy": "markdown_section",
                            },
                        )
                    )
                current_pos += len(section_content) + 1  # +1 for separator
        else:
            # Fallback to paragraph-based splitting
            return self._split_by_paragraphs(document.content)

        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.index = i

        return chunks

    def _split_code(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Split code document preserving function/class boundaries.

        Args:
            document: The parsed code document

        Returns:
            List of TextChunk objects

        Requirements satisfied:
        - 2.4: Preserve function/class boundaries
        """
        chunks = []
        current_pos = 0

        # Use sections from the parsed document (functions/classes)
        if document.sections:
            for i, section in enumerate(document.sections):
                if section.content.strip():
                    chunks.append(
                        TextChunk(
                            content=section.content,
                            index=i,
                            start_char=current_pos,
                            end_char=current_pos + len(section.content),
                            metadata={
                                "definition": section.title,
                                "language": document.metadata.get(
                                    "language", "unknown"
                                ),
                                "strategy": "code_section",
                            },
                        )
                    )
                current_pos += len(section.content) + 1
        else:
            # Fallback to paragraph-based splitting
            return self._split_by_paragraphs(document.content)

        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.index = i

        return chunks

    def _split_by_paragraphs(self, text: str) -> list[TextChunk]:
        """
        Split text by paragraph boundaries.

        Paragraphs are detected by double newlines or other common separators.

        Args:
            text: The text to split

        Returns:
            List of TextChunk objects

        Requirements satisfied:
        - 2.2: Preserve semantic boundaries (paragraphs)
        """
        if not text:
            return []

        # Split by double newlines (paragraph boundaries)
        # Also handle various newline styles
        paragraphs = re.split(r"\n\s*\n", text)

        chunks = []
        current_pos = 0

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                # Find actual position in original text
                try:
                    actual_start = text.index(para, current_pos)
                except ValueError:
                    actual_start = current_pos

                chunks.append(
                    TextChunk(
                        content=para,
                        index=i,
                        start_char=actual_start,
                        end_char=actual_start + len(para),
                        metadata={"strategy": "paragraph"},
                    )
                )
                current_pos = actual_start + len(para)

        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.index = i

        return chunks

    def _merge_small_chunks(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """
        Merge chunks that are smaller than the minimum size.

        This prevents creating too many tiny chunks that would be
        inefficient for LLM processing.

        Args:
            chunks: List of chunks to potentially merge

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        max_size = self.config.chunk_size

        merged = []
        current_content = ""
        current_start = 0
        current_metadata = {}

        for chunk in chunks:
            # If adding this chunk would exceed max size, save current and start new
            if (
                current_content
                and len(current_content) + len(chunk.content) + 2 > max_size
            ):
                if current_content.strip():
                    merged.append(
                        TextChunk(
                            content=current_content.strip(),
                            index=len(merged),
                            start_char=current_start,
                            end_char=current_start + len(current_content),
                            metadata=current_metadata,
                        )
                    )
                current_content = chunk.content
                current_start = chunk.start_char
                current_metadata = chunk.metadata.copy()
            else:
                # Merge with current
                if current_content:
                    current_content += "\n\n" + chunk.content
                else:
                    current_content = chunk.content
                    current_start = chunk.start_char
                    current_metadata = chunk.metadata.copy()

        # Add remaining content
        if current_content.strip():
            merged.append(
                TextChunk(
                    content=current_content.strip(),
                    index=len(merged),
                    start_char=current_start,
                    end_char=current_start + len(current_content),
                    metadata=current_metadata,
                )
            )

        return merged

    def _find_semantic_boundaries(self, text: str) -> list[int]:
        """
        Find positions of semantic boundaries in text.

        Identifies positions where it's safe to split without breaking
        semantic units like paragraphs, sentences, or code blocks.

        Args:
            text: The text to analyze

        Returns:
            List of character positions that are safe split points
        """
        boundaries = []

        # Paragraph boundaries (double newlines)
        for match in re.finditer(r"\n\s*\n", text):
            boundaries.append(match.end())

        # Markdown heading boundaries
        for match in re.finditer(r"\n#{1,6}\s", text):
            boundaries.append(match.start() + 1)  # After the newline

        # Code block boundaries (```)
        for match in re.finditer(r"\n```", text):
            boundaries.append(match.start() + 1)

        # Sort and deduplicate
        boundaries = sorted(set(boundaries))

        return boundaries

    def _split_at_boundary(
        self, text: str, target_pos: int, boundaries: list[int]
    ) -> int:
        """
        Find the best split position near the target position.

        Looks for the nearest semantic boundary to the target position,
        preferring boundaries that are before the target.

        Args:
            text: The text being split
            target_pos: The ideal split position
            boundaries: List of valid boundary positions

        Returns:
            The best split position
        """
        if not boundaries:
            return target_pos

        # Find boundaries within a reasonable range of target
        tolerance = self.config.chunk_size // 4  # 25% tolerance

        candidates = [
            b
            for b in boundaries
            if target_pos - tolerance <= b <= target_pos + tolerance
        ]

        if not candidates:
            return target_pos

        # Prefer boundary closest to but not exceeding target
        before_target = [b for b in candidates if b <= target_pos]
        if before_target:
            return max(before_target)

        return min(candidates)
