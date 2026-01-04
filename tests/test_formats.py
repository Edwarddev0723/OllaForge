"""
Tests for multi-format file support in OllaForge.
"""

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ollaforge.formats import (
    FileFormat,
    FormatError,
    detect_format,
    get_format_description,
    get_supported_formats,
    read_file,
    validate_format_compatibility,
    write_file,
)


# Test data generators
@st.composite
def sample_entry(draw):
    """Generate a sample dataset entry."""
    return {
        "instruction": draw(st.text(min_size=1, max_size=100)),
        "input": draw(st.text(min_size=0, max_size=100)),
        "output": draw(st.text(min_size=1, max_size=100)),
    }


@st.composite
def sample_dataset(draw):
    """Generate a sample dataset."""
    return draw(st.lists(sample_entry(), min_size=1, max_size=10))


class TestFormatDetection:
    """Test format detection functionality."""

    def test_detect_format_from_extension(self):
        """Test format detection from file extensions."""
        test_cases = [
            ("data.jsonl", FileFormat.JSONL),
            ("data.json", FileFormat.JSON),
            ("data.csv", FileFormat.CSV),
            ("data.tsv", FileFormat.TSV),
            ("data.parquet", FileFormat.PARQUET),
        ]

        for filename, expected_format in test_cases:
            assert detect_format(filename) == expected_format

    def test_detect_format_unsupported(self):
        """Test detection of unsupported formats."""
        with pytest.raises(FormatError):
            detect_format("data.xml")

        with pytest.raises(FormatError):
            detect_format("data.xlsx")

    def test_get_supported_formats(self):
        """Test getting list of supported formats."""
        formats = get_supported_formats()
        expected = ["jsonl", "json", "csv", "tsv", "parquet"]
        assert all(fmt in formats for fmt in expected)

    def test_get_format_description(self):
        """Test getting format descriptions."""
        for fmt in FileFormat:
            description = get_format_description(fmt)
            assert isinstance(description, str)
            assert len(description) > 0


class TestJSONLFormat:
    """Test JSONL format reading and writing."""

    @given(dataset=sample_dataset())
    def test_jsonl_round_trip(self, dataset):
        """Test JSONL round-trip consistency."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # Write dataset
            write_file(dataset, temp_path, FileFormat.JSONL)

            # Read it back
            entries, field_names = read_file(temp_path, FileFormat.JSONL)

            # Verify consistency
            assert len(entries) == len(dataset)
            assert entries == dataset

            # Verify field names
            expected_fields = set()
            for entry in dataset:
                expected_fields.update(entry.keys())
            assert set(field_names) == expected_fields

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_jsonl_invalid_format(self):
        """Test handling of invalid JSONL."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
            temp_path = f.name

        try:
            with pytest.raises(FormatError, match="Invalid JSON"):
                read_file(temp_path, FileFormat.JSONL)
        finally:
            os.unlink(temp_path)


class TestJSONFormat:
    """Test JSON array format reading and writing."""

    @given(dataset=sample_dataset())
    def test_json_round_trip(self, dataset):
        """Test JSON array round-trip consistency."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Write dataset
            write_file(dataset, temp_path, FileFormat.JSON)

            # Read it back
            entries, field_names = read_file(temp_path, FileFormat.JSON)

            # Verify consistency
            assert len(entries) == len(dataset)
            assert entries == dataset

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_json_invalid_format(self):
        """Test handling of invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"not": "an array"}')
            temp_path = f.name

        try:
            with pytest.raises(FormatError, match="array of objects"):
                read_file(temp_path, FileFormat.JSON)
        finally:
            os.unlink(temp_path)


class TestCSVFormat:
    """Test CSV format reading and writing."""

    def test_csv_simple_round_trip(self):
        """Test CSV round-trip with simple data."""
        dataset = [
            {"name": "Alice", "age": "30", "city": "New York"},
            {"name": "Bob", "age": "25", "city": "London"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            # Write dataset
            write_file(dataset, temp_path, FileFormat.CSV)

            # Read it back
            entries, field_names = read_file(temp_path, FileFormat.CSV)

            # Verify consistency (CSV converts all values to strings)
            assert len(entries) == len(dataset)
            assert set(field_names) == {"name", "age", "city"}

            for original, read_back in zip(dataset, entries):
                for key in original:
                    assert read_back[key] == str(original[key])

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_csv_complex_values(self):
        """Test CSV handling of complex values (converted to JSON strings)."""
        dataset = [
            {
                "simple": "text",
                "complex": {"nested": "object"},
                "array": [1, 2, 3],
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            # Write dataset
            write_file(dataset, temp_path, FileFormat.CSV)

            # Read it back
            entries, field_names = read_file(temp_path, FileFormat.CSV)

            # Verify complex values are JSON-encoded strings
            assert len(entries) == 1
            entry = entries[0]

            assert entry["simple"] == "text"
            assert entry["complex"] == '{"nested": "object"}'
            assert entry["array"] == "[1, 2, 3]"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestTSVFormat:
    """Test TSV format reading and writing."""

    def test_tsv_round_trip(self):
        """Test TSV round-trip consistency."""
        dataset = [
            {"col1": "value1", "col2": "value2"},
            {"col1": "value3", "col2": "value4"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            temp_path = f.name

        try:
            # Write dataset
            write_file(dataset, temp_path, FileFormat.TSV)

            # Read it back
            entries, field_names = read_file(temp_path, FileFormat.TSV)

            # Verify consistency
            assert len(entries) == len(dataset)
            assert entries == dataset

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestFormatCompatibility:
    """Test format compatibility validation."""

    def test_validate_compatibility(self):
        """Test format compatibility validation."""
        simple_data = [{"key": "value"}]
        complex_data = [{"key": {"nested": "value"}}]

        # All formats should handle simple data
        for fmt in FileFormat:
            assert validate_format_compatibility(simple_data, fmt)

        # All formats should handle complex data (with JSON encoding for CSV/TSV)
        for fmt in FileFormat:
            assert validate_format_compatibility(complex_data, fmt)

    def test_empty_data_compatibility(self):
        """Test compatibility with empty datasets."""
        empty_data = []

        for fmt in FileFormat:
            assert validate_format_compatibility(empty_data, fmt)


class TestErrorHandling:
    """Test error handling in format operations."""

    def test_read_nonexistent_file(self):
        """Test reading non-existent file."""
        with pytest.raises(FormatError, match="File not found"):
            read_file("nonexistent.jsonl")

    def test_write_to_readonly_directory(self):
        """Test writing to read-only directory."""
        # This test may not work on all systems, so we'll skip if it fails
        try:
            readonly_path = "/readonly/test.jsonl"
            with pytest.raises(FormatError):
                write_file([{"test": "data"}], readonly_path)
        except (PermissionError, OSError):
            pytest.skip("Cannot test read-only directory on this system")


# Integration tests
class TestMultiFormatIntegration:
    """Test integration between different formats."""

    @given(dataset=sample_dataset())
    def test_format_conversion_chain(self, dataset):
        """Test converting between multiple formats."""
        formats_to_test = [FileFormat.JSONL, FileFormat.JSON, FileFormat.CSV]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Start with JSONL
            current_data = dataset
            current_file = temp_path / "data.jsonl"
            write_file(current_data, str(current_file), FileFormat.JSONL)

            # Convert through each format
            for i, target_format in enumerate(formats_to_test[1:], 1):
                next_file = temp_path / f"data_{i}.{target_format.value}"

                # Read current format
                entries, _ = read_file(str(current_file))

                # Write to next format
                write_file(entries, str(next_file), target_format)

                # Verify we can read it back
                read_back, _ = read_file(str(next_file), target_format)
                assert len(read_back) == len(entries)

                current_file = next_file

    def test_auto_format_detection(self):
        """Test automatic format detection during read/write."""
        dataset = [{"test": "data", "number": "123"}]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test each format with auto-detection
            for fmt in [FileFormat.JSONL, FileFormat.JSON, FileFormat.CSV]:
                file_path = temp_path / f"test.{fmt.value}"

                # Write with auto-detection
                write_file(dataset, str(file_path))

                # Read with auto-detection
                entries, field_names = read_file(str(file_path))

                assert len(entries) == 1
                assert "test" in field_names
                assert "number" in field_names
