"""
Property-based tests for OllaForge batch processor.

This module contains property-based tests using Hypothesis to verify
the correctness properties of the batch processing module.

Feature: document-to-dataset
"""

import pytest
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from typing import List, Set

from ollaforge.batch_processor import (
    collect_supported_files,
    matches_pattern,
    filter_files_by_pattern,
    aggregate_results,
    merge_entries,
    BatchProcessor,
    BatchConfig,
)
from ollaforge.models import DocProcessingResult


# ============================================================================
# Strategies for generating test data
# ============================================================================

# Strategy for generating valid filenames (without extension)
valid_filename_chars = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-'),
    min_size=1,
    max_size=20
).filter(lambda x: x.strip() and not x.startswith('.') and not x.startswith('-'))

# Strategy for generating supported file extensions
supported_extensions = st.sampled_from(['.md', '.txt', '.html', '.json', '.py', '.js'])

# Strategy for generating glob patterns
glob_patterns = st.sampled_from([
    '*.md',
    '*.txt',
    '*.py',
    '*.json',
    '*.html',
    '*.js',
    'test_*',
    '*_test.*',
    'readme*',
    '*config*',
])


@st.composite
def filename_with_extension(draw, extension: str = None):
    """Generate a filename with a specific or random extension."""
    name = draw(valid_filename_chars)
    ext = extension if extension else draw(supported_extensions)
    return name + ext


@st.composite
def file_list_with_mixed_extensions(draw, min_files: int = 1, max_files: int = 10):
    """Generate a list of filenames with mixed extensions."""
    num_files = draw(st.integers(min_value=min_files, max_value=max_files))
    files = []
    for _ in range(num_files):
        files.append(draw(filename_with_extension()))
    return files


# ============================================================================
# Property Tests for Pattern Filtering
# ============================================================================

@given(data=st.data())
@settings(max_examples=20)
def test_pattern_filtering_correctness_extension_patterns(data):
    """
    **Feature: document-to-dataset, Property 10: Pattern Filtering Correctness**
    **Validates: Requirements 6.3**
    
    For any glob pattern and directory of files, the filtered file list SHALL
    contain exactly the files matching the pattern and no files that don't match.
    
    This test verifies extension-based patterns like "*.md", "*.py".
    """
    # Generate a list of files with mixed extensions
    all_files = data.draw(file_list_with_mixed_extensions(min_files=3, max_files=15))
    
    # Choose a specific extension to filter
    target_ext = data.draw(supported_extensions)
    pattern = f"*{target_ext}"
    
    # Create temporary directory with files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        created_files: List[Path] = []
        
        for filename in all_files:
            file_path = tmpdir_path / filename
            file_path.write_text(f"Content of {filename}")
            created_files.append(file_path)
        
        # Apply pattern filter
        filtered = filter_files_by_pattern(created_files, pattern)
        
        # Verify: all filtered files match the pattern
        for f in filtered:
            assert matches_pattern(f.name, pattern), \
                f"File '{f.name}' should match pattern '{pattern}'"
        
        # Verify: no files that match the pattern are excluded
        expected_matches = [f for f in created_files if f.suffix == target_ext]
        for expected in expected_matches:
            assert expected in filtered, \
                f"File '{expected.name}' matches pattern '{pattern}' but was excluded"
        
        # Verify: count matches
        assert len(filtered) == len(expected_matches), \
            f"Expected {len(expected_matches)} files, got {len(filtered)}"


@given(data=st.data())
@settings(max_examples=20)
def test_pattern_filtering_correctness_prefix_patterns(data):
    """
    **Feature: document-to-dataset, Property 10: Pattern Filtering Correctness**
    **Validates: Requirements 6.3**
    
    For any prefix-based glob pattern (e.g., "test_*"), the filtered file list
    SHALL contain exactly the files matching the pattern.
    """
    # Generate files with and without the prefix
    prefix = "test_"
    num_with_prefix = data.draw(st.integers(min_value=1, max_value=5))
    num_without_prefix = data.draw(st.integers(min_value=1, max_value=5))
    
    files_with_prefix = [
        f"{prefix}{data.draw(valid_filename_chars)}{data.draw(supported_extensions)}"
        for _ in range(num_with_prefix)
    ]
    files_without_prefix = [
        f"{data.draw(valid_filename_chars)}{data.draw(supported_extensions)}"
        for _ in range(num_without_prefix)
    ]
    
    # Ensure files without prefix don't accidentally start with "test_"
    files_without_prefix = [f for f in files_without_prefix if not f.startswith(prefix)]
    assume(len(files_without_prefix) > 0)  # Need at least one non-matching file
    
    all_files = files_with_prefix + files_without_prefix
    pattern = f"{prefix}*"
    
    # Create temporary directory with files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        created_files: List[Path] = []
        
        for filename in all_files:
            file_path = tmpdir_path / filename
            file_path.write_text(f"Content of {filename}")
            created_files.append(file_path)
        
        # Apply pattern filter
        filtered = filter_files_by_pattern(created_files, pattern)
        
        # Verify: all filtered files match the pattern
        for f in filtered:
            assert f.name.startswith(prefix), \
                f"File '{f.name}' should start with '{prefix}'"
        
        # Verify: all files with prefix are included
        for f in created_files:
            if f.name.startswith(prefix):
                assert f in filtered, \
                    f"File '{f.name}' starts with '{prefix}' but was excluded"


@given(data=st.data())
@settings(max_examples=20)
def test_matches_pattern_consistency(data):
    """
    **Feature: document-to-dataset, Property 10: Pattern Filtering Correctness**
    **Validates: Requirements 6.3**
    
    For any filename and pattern, matches_pattern should be consistent
    with fnmatch behavior and filter_files_by_pattern results.
    """
    import fnmatch
    
    filename = data.draw(filename_with_extension())
    pattern = data.draw(glob_patterns)
    
    # matches_pattern should be consistent with fnmatch
    expected = fnmatch.fnmatch(filename, pattern)
    actual = matches_pattern(filename, pattern)
    
    assert actual == expected, \
        f"matches_pattern('{filename}', '{pattern}') = {actual}, expected {expected}"


@given(data=st.data())
@settings(max_examples=20)
def test_filter_preserves_all_matching_files(data):
    """
    **Feature: document-to-dataset, Property 10: Pattern Filtering Correctness**
    **Validates: Requirements 6.3**
    
    For any set of files and pattern, filtering should preserve all and only
    the files that match the pattern.
    """
    # Generate files
    all_filenames = data.draw(file_list_with_mixed_extensions(min_files=5, max_files=20))
    pattern = data.draw(glob_patterns)
    
    # Create Path objects
    files = [Path(f"/fake/dir/{fn}") for fn in all_filenames]
    
    # Filter
    filtered = filter_files_by_pattern(files, pattern)
    filtered_set = set(filtered)
    
    # Verify bidirectional correctness
    for f in files:
        should_match = matches_pattern(f.name, pattern)
        is_in_filtered = f in filtered_set
        
        assert should_match == is_in_filtered, \
            f"File '{f.name}' matches={should_match} but in_filtered={is_in_filtered} for pattern '{pattern}'"


# ============================================================================
# Property Tests for Multi-File Result Aggregation
# ============================================================================

@given(data=st.data())
@settings(max_examples=20)
def test_multi_file_result_aggregation_entry_count(data):
    """
    **Feature: document-to-dataset, Property 11: Multi-File Result Aggregation**
    **Validates: Requirements 6.5**
    
    For any set of successfully processed files, the output dataset SHALL contain
    entries from all processed files, and the total entry count SHALL equal
    the sum of entries from each file.
    """
    # Generate file results
    num_files = data.draw(st.integers(min_value=1, max_value=10))
    
    file_results: List[DocProcessingResult] = []
    all_entries: List[dict] = []
    expected_total_entries = 0
    
    for i in range(num_files):
        # Generate entries for this file
        num_entries = data.draw(st.integers(min_value=0, max_value=20))
        file_entries = [{"id": f"file{i}_entry{j}", "content": f"content_{j}"} for j in range(num_entries)]
        
        # Decide if this file succeeds or fails
        has_errors = data.draw(st.booleans())
        errors = [f"Error in file {i}"] if has_errors else []
        
        file_results.append(DocProcessingResult(
            source_file=f"/path/to/file{i}.md",
            chunks_processed=data.draw(st.integers(min_value=0, max_value=10)),
            entries_generated=num_entries,
            errors=errors
        ))
        
        # Only count entries from successful files
        if not has_errors:
            all_entries.extend(file_entries)
            expected_total_entries += num_entries
    
    # Aggregate results
    result = aggregate_results(
        file_results=file_results,
        all_entries=all_entries,
        output_file="output.jsonl",
        duration=10.0
    )
    
    # Verify total entry count
    assert result.total_entries == len(all_entries), \
        f"Expected {len(all_entries)} total entries, got {result.total_entries}"
    
    # Verify file counts
    assert result.total_files == num_files, \
        f"Expected {num_files} total files, got {result.total_files}"
    
    successful = sum(1 for r in file_results if not r.errors)
    failed = sum(1 for r in file_results if r.errors)
    
    assert result.successful_files == successful, \
        f"Expected {successful} successful files, got {result.successful_files}"
    assert result.failed_files == failed, \
        f"Expected {failed} failed files, got {result.failed_files}"


@given(data=st.data())
@settings(max_examples=20)
def test_merge_entries_preserves_all_entries(data):
    """
    **Feature: document-to-dataset, Property 11: Multi-File Result Aggregation**
    **Validates: Requirements 6.5**
    
    For any list of entry lists, merging should preserve all entries
    and maintain their order within each source list.
    """
    # Generate multiple lists of entries
    num_lists = data.draw(st.integers(min_value=1, max_value=5))
    entries_lists: List[List[dict]] = []
    expected_total = 0
    
    for i in range(num_lists):
        num_entries = data.draw(st.integers(min_value=0, max_value=10))
        entries = [{"source": i, "index": j} for j in range(num_entries)]
        entries_lists.append(entries)
        expected_total += num_entries
    
    # Merge entries
    merged = merge_entries(entries_lists)
    
    # Verify total count
    assert len(merged) == expected_total, \
        f"Expected {expected_total} merged entries, got {len(merged)}"
    
    # Verify all entries are present
    all_original = []
    for lst in entries_lists:
        all_original.extend(lst)
    
    for entry in all_original:
        assert entry in merged, \
            f"Entry {entry} should be in merged result"


@given(data=st.data())
@settings(max_examples=20)
def test_aggregation_preserves_file_results(data):
    """
    **Feature: document-to-dataset, Property 11: Multi-File Result Aggregation**
    **Validates: Requirements 6.5**
    
    For any batch processing result, all individual file results should be
    preserved and accessible in the aggregated result.
    """
    # Generate file results
    num_files = data.draw(st.integers(min_value=1, max_value=10))
    
    file_results: List[DocProcessingResult] = []
    
    for i in range(num_files):
        file_results.append(DocProcessingResult(
            source_file=f"/path/to/file{i}.md",
            chunks_processed=data.draw(st.integers(min_value=0, max_value=10)),
            entries_generated=data.draw(st.integers(min_value=0, max_value=20)),
            errors=[]
        ))
    
    # Aggregate results
    result = aggregate_results(
        file_results=file_results,
        all_entries=[],
        output_file="output.jsonl",
        duration=10.0
    )
    
    # Verify all file results are preserved
    assert len(result.file_results) == len(file_results), \
        f"Expected {len(file_results)} file results, got {len(result.file_results)}"
    
    for original, aggregated in zip(file_results, result.file_results):
        assert original.source_file == aggregated.source_file, \
            f"Source file mismatch: {original.source_file} != {aggregated.source_file}"
        assert original.chunks_processed == aggregated.chunks_processed, \
            f"Chunks processed mismatch"
        assert original.entries_generated == aggregated.entries_generated, \
            f"Entries generated mismatch"


# ============================================================================
# Additional Property Tests for Batch Processor
# ============================================================================

@given(data=st.data())
@settings(max_examples=20, deadline=None)
def test_collect_supported_files_returns_only_supported(data):
    """
    **Feature: document-to-dataset, Property 10: Pattern Filtering Correctness**
    **Validates: Requirements 6.3**
    
    For any directory, collect_supported_files should return only files
    with supported extensions.
    """
    from ollaforge.doc_parser import DocumentParserFactory
    
    # Generate files with mixed extensions (some supported, some not)
    supported_exts = ['.md', '.txt', '.html', '.json', '.py']
    unsupported_exts = ['.xyz', '.abc', '.unsupported']
    
    num_supported = data.draw(st.integers(min_value=1, max_value=5))
    num_unsupported = data.draw(st.integers(min_value=1, max_value=5))
    
    supported_files = [
        f"{data.draw(valid_filename_chars)}{data.draw(st.sampled_from(supported_exts))}"
        for _ in range(num_supported)
    ]
    unsupported_files = [
        f"{data.draw(valid_filename_chars)}{data.draw(st.sampled_from(unsupported_exts))}"
        for _ in range(num_unsupported)
    ]
    
    all_files = supported_files + unsupported_files
    
    # Create temporary directory with files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        for filename in all_files:
            file_path = tmpdir_path / filename
            file_path.write_text(f"Content of {filename}")
        
        # Collect supported files
        collected = collect_supported_files(tmpdir_path)
        
        # Verify: all collected files have supported extensions
        for f in collected:
            assert DocumentParserFactory.is_supported(str(f)), \
                f"File '{f.name}' should have a supported extension"
        
        # Verify: no unsupported files are collected
        for f in collected:
            assert f.suffix not in unsupported_exts, \
                f"File '{f.name}' has unsupported extension but was collected"


@given(data=st.data())
@settings(max_examples=20, deadline=None)
def test_collect_with_pattern_filters_correctly(data):
    """
    **Feature: document-to-dataset, Property 10: Pattern Filtering Correctness**
    **Validates: Requirements 6.3**
    
    When a pattern is provided, collect_supported_files should return only
    files that match both the pattern AND have supported extensions.
    """
    # Create files with specific patterns
    target_ext = '.md'
    other_ext = '.py'
    
    # Files that match pattern and have target extension
    matching_files = [f"readme{i}.md" for i in range(3)]
    # Files that don't match pattern but have target extension
    non_matching_same_ext = [f"other{i}.md" for i in range(2)]
    # Files that match pattern but have different extension
    matching_diff_ext = [f"readme{i}.py" for i in range(2)]
    
    all_files = matching_files + non_matching_same_ext + matching_diff_ext
    pattern = "readme*"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        for filename in all_files:
            file_path = tmpdir_path / filename
            file_path.write_text(f"Content of {filename}")
        
        # Collect with pattern
        collected = collect_supported_files(tmpdir_path, pattern=pattern)
        collected_names = {f.name for f in collected}
        
        # Verify: only files matching pattern are collected
        for f in collected:
            assert matches_pattern(f.name, pattern), \
                f"File '{f.name}' should match pattern '{pattern}'"
        
        # Verify: matching files with supported extensions are included
        for filename in matching_files + matching_diff_ext:
            assert filename in collected_names, \
                f"File '{filename}' matches pattern but was not collected"
        
        # Verify: non-matching files are excluded
        for filename in non_matching_same_ext:
            assert filename not in collected_names, \
                f"File '{filename}' doesn't match pattern but was collected"
