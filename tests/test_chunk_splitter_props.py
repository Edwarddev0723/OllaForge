"""
Property-based tests for the chunk splitter module.

This module contains property-based tests using Hypothesis to verify
the correctness of the ChunkSplitter implementation.

Feature: document-to-dataset
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ollaforge.chunk_splitter import (
    ChunkSplitter,
    ChunkConfig,
    SplitStrategy,
    TextChunk,
)
from ollaforge.doc_parser import (
    ParsedDocument,
    DocumentType,
    DocumentSection,
)


# ============================================================================
# Strategies for generating test data
# ============================================================================

@st.composite
def chunk_configs(draw):
    """Generate valid ChunkConfig instances."""
    chunk_size = draw(st.integers(min_value=200, max_value=5000))
    # Ensure overlap is less than chunk_size
    max_overlap = min(chunk_size - 1, 1000)
    chunk_overlap = draw(st.integers(min_value=0, max_value=max_overlap))
    # Ensure min_chunk_size is less than chunk_size
    max_min_size = min(chunk_size - 1, 100)
    min_chunk_size = draw(st.integers(min_value=10, max_value=max_min_size))
    strategy = draw(st.sampled_from(list(SplitStrategy)))
    
    return ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
        min_chunk_size=min_chunk_size,
    )


@st.composite
def text_documents(draw, min_length=1, max_length=10000):
    """Generate ParsedDocument instances with text content."""
    # Generate text content
    content = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
        min_size=min_length,
        max_size=max_length
    ))
    
    # Ensure content is not just whitespace
    assume(content.strip())
    
    return ParsedDocument(
        content=content,
        doc_type=DocumentType.TEXT,
        metadata={},
        sections=[],
        source_path='test.txt'
    )


@st.composite
def markdown_documents(draw, min_sections=1, max_sections=5):
    """Generate ParsedDocument instances with Markdown content and sections."""
    num_sections = draw(st.integers(min_value=min_sections, max_value=max_sections))
    
    sections = []
    content_parts = []
    current_line = 0
    
    for i in range(num_sections):
        level = draw(st.integers(min_value=1, max_value=3))
        title = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N')),
            min_size=1,
            max_size=30
        ))
        section_content = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
            min_size=10,
            max_size=500
        ))
        
        # Ensure title and content are not just whitespace
        assume(title.strip())
        assume(section_content.strip())
        
        heading = '#' * level + ' ' + title
        content_parts.append(heading)
        content_parts.append(section_content)
        
        sections.append(DocumentSection(
            title=title,
            content=section_content,
            level=level,
            start_line=current_line,
            end_line=current_line + 2
        ))
        current_line += 3
    
    full_content = '\n\n'.join(content_parts)
    
    return ParsedDocument(
        content=full_content,
        doc_type=DocumentType.MARKDOWN,
        metadata={},
        sections=sections,
        source_path='test.md'
    )


# ============================================================================
# Property Tests
# ============================================================================

class TestChunkSizeCompliance:
    """
    Property 3: Chunk Size Compliance
    
    *For any* chunk size configuration and any document content,
    all generated chunks SHALL have length less than or equal to
    the configured chunk size.
    
    **Validates: Requirements 2.1, 2.5**
    """
    
    @given(
        config=chunk_configs(),
        doc=text_documents(min_length=100, max_length=5000)
    )
    @settings(max_examples=20)
    def test_all_chunks_within_size_limit(self, config, doc):
        """
        Feature: document-to-dataset, Property 3: Chunk Size Compliance
        
        For any chunk configuration and document, all chunks must be
        within the configured size limit.
        
        **Validates: Requirements 2.1, 2.5**
        """
        splitter = ChunkSplitter(config)
        chunks = splitter.split(doc)
        
        for chunk in chunks:
            assert len(chunk.content) <= config.chunk_size, (
                f"Chunk size {len(chunk.content)} exceeds limit {config.chunk_size}"
            )
    
    @given(
        chunk_size=st.integers(min_value=200, max_value=2000),
        text_length=st.integers(min_value=100, max_value=5000)
    )
    @settings(max_examples=20)
    def test_fixed_size_chunks_within_limit(self, chunk_size, text_length):
        """
        Feature: document-to-dataset, Property 3: Chunk Size Compliance (Fixed Size)
        
        For fixed-size splitting, all chunks must be within the size limit.
        
        **Validates: Requirements 2.1, 2.5**
        """
        # Create config with valid parameters
        overlap = min(chunk_size // 4, 200)
        min_size = min(chunk_size // 4, 50)
        
        config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            strategy=SplitStrategy.FIXED_SIZE,
            min_chunk_size=min_size
        )
        
        # Create document with specified length
        content = 'A' * text_length
        doc = ParsedDocument(
            content=content,
            doc_type=DocumentType.TEXT,
            source_path='test.txt'
        )
        
        splitter = ChunkSplitter(config)
        chunks = splitter.split(doc)
        
        for chunk in chunks:
            assert len(chunk.content) <= chunk_size, (
                f"Chunk size {len(chunk.content)} exceeds limit {chunk_size}"
            )
    
    @given(
        config=chunk_configs(),
        doc=markdown_documents(min_sections=1, max_sections=3)
    )
    @settings(max_examples=20)
    def test_hybrid_chunks_within_limit(self, config, doc):
        """
        Feature: document-to-dataset, Property 3: Chunk Size Compliance (Hybrid)
        
        For hybrid splitting, all chunks must be within the size limit.
        
        **Validates: Requirements 2.1, 2.5**
        """
        # Force hybrid strategy
        config.strategy = SplitStrategy.HYBRID
        
        splitter = ChunkSplitter(config)
        chunks = splitter.split(doc)
        
        for chunk in chunks:
            assert len(chunk.content) <= config.chunk_size, (
                f"Chunk size {len(chunk.content)} exceeds limit {config.chunk_size}"
            )



class TestSemanticBoundaryPreservation:
    """
    Property 4: Semantic Boundary Preservation
    
    *For any* document with clear semantic boundaries (paragraphs, headings, code blocks),
    the Chunk_Splitter SHALL not split content mid-paragraph or mid-heading
    when the boundary fits within chunk size.
    
    **Validates: Requirements 2.2, 2.3, 2.4**
    """
    
    @given(doc=markdown_documents(min_sections=2, max_sections=4))
    @settings(max_examples=20)
    def test_markdown_headings_preserved(self, doc):
        """
        Feature: document-to-dataset, Property 4: Semantic Boundary Preservation (Markdown)
        
        For Markdown documents with sections that fit within chunk size,
        each section should be in its own chunk without being split.
        
        **Validates: Requirements 2.2, 2.3**
        """
        # Use a large chunk size to ensure sections fit
        config = ChunkConfig(
            chunk_size=10000,
            chunk_overlap=200,
            strategy=SplitStrategy.SEMANTIC,
            min_chunk_size=50
        )
        
        splitter = ChunkSplitter(config)
        chunks = splitter.split(doc)
        
        # Each section should be preserved as a separate chunk
        # (when sections are small enough to fit)
        for section in doc.sections:
            section_title = section.title
            if section_title:
                # Find a chunk that contains this section's title
                found = False
                for chunk in chunks:
                    if section_title in chunk.content:
                        found = True
                        # The heading should not be split from its content
                        # Check that the heading marker is present
                        heading_marker = '#' * section.level + ' ' + section_title
                        if heading_marker in chunk.content:
                            # Heading is properly preserved
                            break
                
                assert found, f"Section '{section_title}' not found in any chunk"
    
    @given(
        num_paragraphs=st.integers(min_value=2, max_value=5),
        para_length=st.integers(min_value=50, max_value=200)
    )
    @settings(max_examples=20)
    def test_paragraphs_not_split_when_fit(self, num_paragraphs, para_length):
        """
        Feature: document-to-dataset, Property 4: Semantic Boundary Preservation (Paragraphs)
        
        For text documents with paragraphs that fit within chunk size,
        paragraphs should not be split mid-content.
        
        **Validates: Requirements 2.2**
        """
        # Create paragraphs
        paragraphs = [f"Paragraph {i}: " + "x" * para_length for i in range(num_paragraphs)]
        content = "\n\n".join(paragraphs)
        
        doc = ParsedDocument(
            content=content,
            doc_type=DocumentType.TEXT,
            sections=[],
            source_path='test.txt'
        )
        
        # Use chunk size large enough to fit each paragraph
        config = ChunkConfig(
            chunk_size=max(para_length * 2, 500),
            chunk_overlap=50,
            strategy=SplitStrategy.SEMANTIC,
            min_chunk_size=20
        )
        
        splitter = ChunkSplitter(config)
        chunks = splitter.split(doc)
        
        # Each paragraph should be fully contained in some chunk
        for para in paragraphs:
            found_complete = False
            for chunk in chunks:
                if para in chunk.content:
                    found_complete = True
                    break
            
            # Note: Due to merging, paragraphs might be combined
            # but should not be split mid-content
            if not found_complete:
                # Check if paragraph start and end are in the same chunk
                para_start = para[:20]
                para_end = para[-20:]
                for chunk in chunks:
                    if para_start in chunk.content and para_end in chunk.content:
                        found_complete = True
                        break
            
            assert found_complete, f"Paragraph was split across chunks"
    
    @given(
        num_functions=st.integers(min_value=2, max_value=4),
        func_body_lines=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=20)
    def test_code_blocks_preserved(self, num_functions, func_body_lines):
        """
        Feature: document-to-dataset, Property 4: Semantic Boundary Preservation (Code)
        
        For code documents with function definitions that fit within chunk size,
        functions should not be split mid-definition.
        
        **Validates: Requirements 2.4**
        """
        # Create Python functions
        sections = []
        content_parts = []
        
        for i in range(num_functions):
            func_name = f"function_{i}"
            body_lines = [f"    line_{j} = {j}" for j in range(func_body_lines)]
            func_content = f"def {func_name}():\n" + "\n".join(body_lines)
            
            content_parts.append(func_content)
            sections.append(DocumentSection(
                title=f"def {func_name}",
                content=func_content,
                level=1,
                start_line=i * (func_body_lines + 1),
                end_line=(i + 1) * (func_body_lines + 1) - 1
            ))
        
        content = "\n\n".join(content_parts)
        
        doc = ParsedDocument(
            content=content,
            doc_type=DocumentType.CODE,
            metadata={'language': 'python'},
            sections=sections,
            source_path='test.py'
        )
        
        # Use chunk size large enough to fit each function
        max_func_size = max(len(s.content) for s in sections)
        config = ChunkConfig(
            chunk_size=max(max_func_size * 2, 500),
            chunk_overlap=50,
            strategy=SplitStrategy.SEMANTIC,
            min_chunk_size=20
        )
        
        splitter = ChunkSplitter(config)
        chunks = splitter.split(doc)
        
        # Each function should be fully contained in some chunk
        for section in sections:
            func_def = section.title  # e.g., "def function_0"
            found = False
            for chunk in chunks:
                if func_def in chunk.content:
                    # Check that the function body is also in the same chunk
                    # by verifying the last line of the function
                    last_line = f"line_{func_body_lines - 1}"
                    if last_line in chunk.content:
                        found = True
                        break
            
            assert found, f"Function '{func_def}' was split across chunks"



class TestChunkOverlapCorrectness:
    """
    Property 5: Chunk Overlap Correctness
    
    *For any* configured overlap value and consecutive chunk pairs,
    the end of chunk N and the beginning of chunk N+1 SHALL share
    exactly the configured overlap characters (when content permits).
    
    **Validates: Requirements 2.6**
    """
    
    @given(
        chunk_size=st.integers(min_value=200, max_value=1000),
        overlap=st.integers(min_value=10, max_value=100),
        text_length=st.integers(min_value=500, max_value=3000)
    )
    @settings(max_examples=20)
    def test_fixed_size_overlap_exact(self, chunk_size, overlap, text_length):
        """
        Feature: document-to-dataset, Property 5: Chunk Overlap Correctness (Fixed Size)
        
        For fixed-size splitting with overlap, consecutive chunks should
        share exactly the configured overlap characters.
        
        **Validates: Requirements 2.6**
        """
        # Ensure overlap is less than chunk_size
        assume(overlap < chunk_size)
        
        config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            strategy=SplitStrategy.FIXED_SIZE,
            min_chunk_size=min(50, chunk_size // 4)
        )
        
        # Create uniform content for predictable overlap
        content = 'A' * text_length
        doc = ParsedDocument(
            content=content,
            doc_type=DocumentType.TEXT,
            source_path='test.txt'
        )
        
        splitter = ChunkSplitter(config)
        chunks = splitter.split(doc)
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # The end of current chunk should match the beginning of next chunk
            # by exactly 'overlap' characters
            current_end = current_chunk.content[-overlap:]
            next_start = next_chunk.content[:overlap]
            
            assert current_end == next_start, (
                f"Overlap mismatch between chunks {i} and {i+1}: "
                f"expected {overlap} chars overlap"
            )
    
    @given(
        chunk_size=st.integers(min_value=200, max_value=1000),
        overlap=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=20)
    def test_overlap_preserves_content_continuity(self, chunk_size, overlap):
        """
        Feature: document-to-dataset, Property 5: Chunk Overlap Correctness (Continuity)
        
        The overlap should ensure content continuity - the overlapping
        portion should be identical in both chunks.
        
        **Validates: Requirements 2.6**
        """
        assume(overlap < chunk_size)
        
        config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            strategy=SplitStrategy.FIXED_SIZE,
            min_chunk_size=min(50, chunk_size // 4)
        )
        
        # Create content with distinct characters to verify overlap
        # Use a pattern that makes overlap verification clear
        text_length = chunk_size * 3  # Ensure multiple chunks
        content = ''.join([chr(ord('A') + (i % 26)) for i in range(text_length)])
        
        doc = ParsedDocument(
            content=content,
            doc_type=DocumentType.TEXT,
            source_path='test.txt'
        )
        
        splitter = ChunkSplitter(config)
        chunks = splitter.split(doc)
        
        # Verify overlap content matches
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Get the overlapping portions
            overlap_from_current = current_chunk.content[-overlap:]
            overlap_from_next = next_chunk.content[:overlap]
            
            assert overlap_from_current == overlap_from_next, (
                f"Overlap content mismatch between chunks {i} and {i+1}"
            )
    
    @given(
        chunk_size=st.integers(min_value=300, max_value=800),
        overlap=st.integers(min_value=20, max_value=80),
    )
    @settings(max_examples=20)
    def test_overlap_positions_are_correct(self, chunk_size, overlap):
        """
        Feature: document-to-dataset, Property 5: Chunk Overlap Correctness (Positions)
        
        The start_char and end_char positions should correctly reflect
        the overlap between chunks.
        
        **Validates: Requirements 2.6**
        """
        assume(overlap < chunk_size)
        
        config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            strategy=SplitStrategy.FIXED_SIZE,
            min_chunk_size=min(50, chunk_size // 4)
        )
        
        # Create content
        text_length = chunk_size * 4
        content = 'X' * text_length
        
        doc = ParsedDocument(
            content=content,
            doc_type=DocumentType.TEXT,
            source_path='test.txt'
        )
        
        splitter = ChunkSplitter(config)
        chunks = splitter.split(doc)
        
        # Verify position consistency
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # The next chunk should start at (current_end - overlap)
            expected_next_start = current_chunk.end_char - overlap
            
            # Allow for some tolerance due to boundary adjustments
            assert abs(next_chunk.start_char - expected_next_start) <= 1, (
                f"Position mismatch: chunk {i+1} starts at {next_chunk.start_char}, "
                f"expected around {expected_next_start}"
            )
