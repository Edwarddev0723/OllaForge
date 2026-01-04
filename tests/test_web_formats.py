"""
Tests for multi-format file support in OllaForge web interface.

This module tests the format handling functionality including:
- Format detection and support for JSONL, JSON, CSV, TSV, Parquet
- Format conversion for downloads
- Error handling for unsupported formats

Requirements satisfied:
- 4.1: Support JSONL, JSON, CSV, TSV, and Parquet formats for upload
- 4.2: Provide format selection options for download
- 4.3: Preserve all data fields correctly during format conversion
- 4.4: Display clear error message for unsupported formats
"""

import asyncio
import csv
import io
import json
import os
import tempfile
from unittest.mock import patch

import httpx
import pytest
from httpx._transports.asgi import ASGITransport
from hypothesis import given, settings
from hypothesis import strategies as st

from ollaforge.formats import (
    FileFormat,
    FormatError,
    detect_format,
    get_supported_formats,
    read_file,
    validate_format_compatibility,
    write_file,
)
from ollaforge.web.server import app


def create_test_client():
    """Create an async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ============================================================================
# Strategies for Property-Based Testing
# ============================================================================

# Strategy for valid field names
field_name_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=15
).filter(lambda x: x[0].isalpha() and x.strip() == x)

# Strategy for simple field values (avoid special characters that break CSV)
simple_value_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
    min_size=1,
    max_size=50,
).filter(lambda x: x.strip() == x and len(x.strip()) > 0)

# Strategy for supported formats
supported_format_strategy = st.sampled_from(
    [FileFormat.JSONL, FileFormat.JSON, FileFormat.CSV, FileFormat.TSV]
)

# Strategy for download format strings
download_format_strategy = st.sampled_from(["jsonl", "json", "csv", "tsv"])

# Strategy for unsupported file extensions
unsupported_extension_strategy = st.sampled_from(
    [".xyz", ".doc", ".pdf", ".exe", ".bin", ".mp3", ".jpg"]
)


# ============================================================================
# Helper Functions
# ============================================================================


def create_jsonl_content(entries: list) -> bytes:
    """Create JSONL file content from entries."""
    lines = [json.dumps(entry, ensure_ascii=False) for entry in entries]
    return "\n".join(lines).encode("utf-8")


def create_json_content(entries: list) -> bytes:
    """Create JSON array file content from entries."""
    return json.dumps(entries, ensure_ascii=False, indent=2).encode("utf-8")


def create_csv_content(entries: list, delimiter: str = ",") -> bytes:
    """Create CSV/TSV file content from entries."""
    if not entries:
        return b""

    output = io.StringIO()
    fieldnames = list(entries[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=delimiter)
    writer.writeheader()
    for entry in entries:
        writer.writerow(entry)
    return output.getvalue().encode("utf-8")


def create_test_entries(field_names: list, count: int = 3) -> list:
    """Create test entries with given field names."""
    entries = []
    for i in range(count):
        entry = {field: f"value_{field}_{i}" for field in field_names}
        entries.append(entry)
    return entries


# ============================================================================
# Property Test 12: Format support is comprehensive
# ============================================================================


@given(
    field_names=st.lists(field_name_strategy, min_size=1, max_size=4, unique=True),
    entry_count=st.integers(min_value=1, max_value=5),
    file_format=supported_format_strategy,
)
@settings(max_examples=50, deadline=60000)
def test_format_support_is_comprehensive(field_names, entry_count, file_format):
    """
    **Feature: web-interface, Property 12: Format support is comprehensive**
    **Validates: Requirements 4.1**

    For any supported file format (JSONL, JSON, CSV, TSV), the system should
    correctly recognize and parse files in that format.
    """

    async def run_test():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            # Create test entries
            entries = create_test_entries(field_names, entry_count)

            # Create content in the specified format
            extension_map = {
                FileFormat.JSONL: ".jsonl",
                FileFormat.JSON: ".json",
                FileFormat.CSV: ".csv",
                FileFormat.TSV: ".tsv",
            }

            content_creators = {
                FileFormat.JSONL: lambda e: create_jsonl_content(e),
                FileFormat.JSON: lambda e: create_json_content(e),
                FileFormat.CSV: lambda e: create_csv_content(e, ","),
                FileFormat.TSV: lambda e: create_csv_content(e, "\t"),
            }

            extension = extension_map[file_format]
            content = content_creators[file_format](entries)
            filename = f"test{extension}"

            # Upload file
            files = {"file": (filename, content, "application/octet-stream")}
            response = await client.post("/api/augment/upload", files=files)

            # Should return 200 OK
            assert (
                response.status_code == 200
            ), f"Upload of {file_format.value} should return 200, got {response.status_code}: {response.text}"

            data = response.json()

            # Should have correct entry count
            assert (
                data["entry_count"] == entry_count
            ), f"Entry count should be {entry_count}, got {data['entry_count']}"

            # Should extract all field names
            returned_fields = set(data["fields"])
            expected_fields = set(field_names)
            assert (
                returned_fields == expected_fields
            ), f"Expected fields {expected_fields}, got {returned_fields}"

            # Preview should have entries
            assert len(data["preview"]) > 0, "Preview should have entries"

    asyncio.run(run_test())


# ============================================================================
# Property Test 13: Format conversion preserves data
# ============================================================================


@given(
    field_names=st.lists(field_name_strategy, min_size=1, max_size=3, unique=True),
    entry_count=st.integers(min_value=1, max_value=5),
    source_format=supported_format_strategy,
    target_format=download_format_strategy,
)
@settings(max_examples=50, deadline=60000)
def test_format_conversion_preserves_data(
    field_names, entry_count, source_format, target_format
):
    """
    **Feature: web-interface, Property 13: Format conversion preserves data**
    **Validates: Requirements 4.3**

    For any dataset, converting between supported formats should preserve
    all data fields correctly (round-trip property).
    """
    # Create test entries with simple values
    entries = []
    for i in range(entry_count):
        entry = {field: f"value{i}" for field in field_names}
        entries.append(entry)

    # Write to source format
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f".{source_format.value}", delete=False
    ) as tmp:
        source_path = tmp.name

    try:
        write_file(entries, source_path, source_format)

        # Read back
        read_entries, read_fields = read_file(source_path)

        # Verify entry count preserved
        assert (
            len(read_entries) == entry_count
        ), f"Entry count should be {entry_count}, got {len(read_entries)}"

        # Verify all fields preserved
        assert set(read_fields) == set(
            field_names
        ), f"Fields should be {set(field_names)}, got {set(read_fields)}"

        # Verify all values preserved
        for i, entry in enumerate(read_entries):
            for field in field_names:
                expected_value = f"value{i}"
                actual_value = entry.get(field, "")
                assert (
                    str(actual_value) == expected_value
                ), f"Value for {field} in entry {i} should be '{expected_value}', got '{actual_value}'"

        # Now convert to target format
        target_format_enum = {
            "jsonl": FileFormat.JSONL,
            "json": FileFormat.JSON,
            "csv": FileFormat.CSV,
            "tsv": FileFormat.TSV,
        }[target_format]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{target_format}", delete=False
        ) as tmp:
            target_path = tmp.name

        try:
            write_file(read_entries, target_path, target_format_enum)

            # Read from target format
            final_entries, final_fields = read_file(target_path)

            # Verify entry count still preserved
            assert (
                len(final_entries) == entry_count
            ), f"Entry count after conversion should be {entry_count}, got {len(final_entries)}"

            # Verify all fields still preserved
            assert set(final_fields) == set(
                field_names
            ), f"Fields after conversion should be {set(field_names)}, got {set(final_fields)}"

            # Verify all values still preserved
            for i, entry in enumerate(final_entries):
                for field in field_names:
                    expected_value = f"value{i}"
                    actual_value = entry.get(field, "")
                    assert (
                        str(actual_value) == expected_value
                    ), f"Value for {field} in entry {i} after conversion should be '{expected_value}', got '{actual_value}'"

        finally:
            os.unlink(target_path)

    finally:
        os.unlink(source_path)


# ============================================================================
# Property Test 14: Unsupported formats show errors
# ============================================================================


@given(extension=unsupported_extension_strategy)
@settings(max_examples=20, deadline=60000)
def test_unsupported_formats_show_errors(extension):
    """
    **Feature: web-interface, Property 14: Unsupported formats show errors**
    **Validates: Requirements 4.4**

    For any unsupported file format uploaded, the system should display
    a clear error message listing the supported formats.
    """

    async def run_test():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            # Create dummy content
            content = b"some random content"
            filename = f"test{extension}"

            # Upload file with unsupported extension
            files = {"file": (filename, content, "application/octet-stream")}
            response = await client.post("/api/augment/upload", files=files)

            # Should return 415 Unsupported Media Type
            assert (
                response.status_code == 415
            ), f"Unsupported format {extension} should return 415, got {response.status_code}"

            data = response.json()

            # Should have error detail
            assert "detail" in data, "Response should have error detail"

            detail = data["detail"]

            # Error should mention unsupported format
            error_str = str(detail).lower()
            assert (
                "unsupported" in error_str or "format" in error_str
            ), f"Error should mention unsupported format: {detail}"

            # Error should list supported formats
            supported = [".jsonl", ".json", ".csv", ".tsv", ".parquet"]
            has_supported_list = any(fmt in str(detail) for fmt in supported)
            assert has_supported_list, f"Error should list supported formats: {detail}"

    asyncio.run(run_test())


# ============================================================================
# Unit Tests for Format Handling
# ============================================================================


class TestFormatDetection:
    """Unit tests for format detection."""

    def test_detect_jsonl_format(self):
        """Test JSONL format detection from extension."""
        assert detect_format("test.jsonl") == FileFormat.JSONL
        assert detect_format("path/to/file.jsonl") == FileFormat.JSONL

    def test_detect_json_format(self):
        """Test JSON format detection from extension."""
        assert detect_format("test.json") == FileFormat.JSON
        assert detect_format("path/to/file.json") == FileFormat.JSON

    def test_detect_csv_format(self):
        """Test CSV format detection from extension."""
        assert detect_format("test.csv") == FileFormat.CSV
        assert detect_format("path/to/file.csv") == FileFormat.CSV

    def test_detect_tsv_format(self):
        """Test TSV format detection from extension."""
        assert detect_format("test.tsv") == FileFormat.TSV
        assert detect_format("path/to/file.tsv") == FileFormat.TSV

    def test_detect_parquet_format(self):
        """Test Parquet format detection from extension."""
        assert detect_format("test.parquet") == FileFormat.PARQUET
        assert detect_format("path/to/file.parquet") == FileFormat.PARQUET

    def test_detect_unsupported_format_raises_error(self):
        """Test that unsupported formats raise FormatError."""
        with pytest.raises(FormatError) as exc_info:
            detect_format("test.xyz")
        assert "unsupported" in str(exc_info.value).lower()

    def test_detect_format_case_insensitive(self):
        """Test that format detection is case-insensitive."""
        assert detect_format("test.JSONL") == FileFormat.JSONL
        assert detect_format("test.JSON") == FileFormat.JSON
        assert detect_format("test.CSV") == FileFormat.CSV


class TestFormatReadWrite:
    """Unit tests for format read/write operations."""

    def test_read_write_jsonl(self):
        """Test JSONL read/write round-trip."""
        entries = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_file(entries, tmp_path, FileFormat.JSONL)
            read_entries, fields = read_file(tmp_path)

            assert len(read_entries) == 2
            assert set(fields) == {"a", "b"}
            assert read_entries[0]["a"] == "1"
            assert read_entries[1]["b"] == "4"
        finally:
            os.unlink(tmp_path)

    def test_read_write_json(self):
        """Test JSON read/write round-trip."""
        entries = [{"x": "hello", "y": "world"}]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_file(entries, tmp_path, FileFormat.JSON)
            read_entries, fields = read_file(tmp_path)

            assert len(read_entries) == 1
            assert set(fields) == {"x", "y"}
            assert read_entries[0]["x"] == "hello"
        finally:
            os.unlink(tmp_path)

    def test_read_write_csv(self):
        """Test CSV read/write round-trip."""
        entries = [{"col1": "val1", "col2": "val2"}]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_file(entries, tmp_path, FileFormat.CSV)
            read_entries, fields = read_file(tmp_path)

            assert len(read_entries) == 1
            assert "col1" in fields
            assert "col2" in fields
            assert read_entries[0]["col1"] == "val1"
        finally:
            os.unlink(tmp_path)

    def test_read_write_tsv(self):
        """Test TSV read/write round-trip."""
        entries = [{"field1": "data1", "field2": "data2"}]

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_file(entries, tmp_path, FileFormat.TSV)
            read_entries, fields = read_file(tmp_path)

            assert len(read_entries) == 1
            assert "field1" in fields
            assert read_entries[0]["field1"] == "data1"
        finally:
            os.unlink(tmp_path)

    def test_read_nonexistent_file_raises_error(self):
        """Test that reading non-existent file raises FormatError."""
        with pytest.raises(FormatError) as exc_info:
            read_file("/nonexistent/path/file.jsonl")
        assert "not found" in str(exc_info.value).lower()

    def test_read_invalid_jsonl_raises_error(self):
        """Test that reading invalid JSONL raises FormatError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp.write("not valid json\n")
            tmp_path = tmp.name

        try:
            with pytest.raises(FormatError) as exc_info:
                read_file(tmp_path)
            assert "json" in str(exc_info.value).lower()
        finally:
            os.unlink(tmp_path)


class TestSupportedFormats:
    """Unit tests for supported formats listing."""

    def test_get_supported_formats_returns_all(self):
        """Test that get_supported_formats returns all formats."""
        formats = get_supported_formats()

        assert "jsonl" in formats
        assert "json" in formats
        assert "csv" in formats
        assert "tsv" in formats
        assert "parquet" in formats

    def test_validate_format_compatibility(self):
        """Test format compatibility validation."""
        entries = [{"a": "1"}, {"a": "2"}]

        # All formats should be compatible with simple entries
        assert validate_format_compatibility(entries, FileFormat.JSONL) is True
        assert validate_format_compatibility(entries, FileFormat.JSON) is True
        assert validate_format_compatibility(entries, FileFormat.CSV) is True
        assert validate_format_compatibility(entries, FileFormat.TSV) is True

        # Empty entries should be compatible
        assert validate_format_compatibility([], FileFormat.JSONL) is True


# ============================================================================
# Unit Tests for Web API Format Handling
# ============================================================================


@pytest.mark.asyncio
async def test_upload_jsonl_format():
    """Test uploading JSONL format file."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        entries = [{"instruction": "test", "output": "result"}]
        content = create_jsonl_content(entries)

        files = {"file": ("test.jsonl", content, "application/x-ndjson")}
        response = await client.post("/api/augment/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["entry_count"] == 1
        assert "instruction" in data["fields"]
        assert "output" in data["fields"]


@pytest.mark.asyncio
async def test_upload_json_format():
    """Test uploading JSON array format file."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        entries = [{"field1": "value1"}, {"field1": "value2"}]
        content = create_json_content(entries)

        files = {"file": ("test.json", content, "application/json")}
        response = await client.post("/api/augment/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["entry_count"] == 2
        assert "field1" in data["fields"]


@pytest.mark.asyncio
async def test_upload_csv_format():
    """Test uploading CSV format file."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        entries = [{"col_a": "data1", "col_b": "data2"}]
        content = create_csv_content(entries, ",")

        files = {"file": ("test.csv", content, "text/csv")}
        response = await client.post("/api/augment/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["entry_count"] == 1
        assert "col_a" in data["fields"]
        assert "col_b" in data["fields"]


@pytest.mark.asyncio
async def test_upload_tsv_format():
    """Test uploading TSV format file."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        entries = [{"tab_col1": "val1", "tab_col2": "val2"}]
        content = create_csv_content(entries, "\t")

        files = {"file": ("test.tsv", content, "text/tab-separated-values")}
        response = await client.post("/api/augment/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["entry_count"] == 1
        assert "tab_col1" in data["fields"]


@pytest.mark.asyncio
@patch("ollaforge.web.routes.augmentation.augmentation_service")
async def test_download_jsonl_format(mock_service):
    """Test downloading in JSONL format."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        task_id = "test_download_jsonl"
        mock_service.get_task.return_value = {
            "status": "completed",
            "progress": 1,
            "total": 1,
            "result": {"entries": [{"field": "value"}], "total": 1},
        }

        response = await client.get(f"/api/augment/{task_id}/download?format=jsonl")

        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers.get("content-type", "")
        assert b"field" in response.content


@pytest.mark.asyncio
@patch("ollaforge.web.routes.augmentation.augmentation_service")
async def test_download_json_format(mock_service):
    """Test downloading in JSON format."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        task_id = "test_download_json"
        mock_service.get_task.return_value = {
            "status": "completed",
            "progress": 1,
            "total": 1,
            "result": {"entries": [{"field": "value"}], "total": 1},
        }

        response = await client.get(f"/api/augment/{task_id}/download?format=json")

        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")


@pytest.mark.asyncio
@patch("ollaforge.web.routes.augmentation.augmentation_service")
async def test_download_csv_format(mock_service):
    """Test downloading in CSV format."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        task_id = "test_download_csv"
        mock_service.get_task.return_value = {
            "status": "completed",
            "progress": 1,
            "total": 1,
            "result": {"entries": [{"field": "value"}], "total": 1},
        }

        response = await client.get(f"/api/augment/{task_id}/download?format=csv")

        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")


@pytest.mark.asyncio
@patch("ollaforge.web.routes.augmentation.augmentation_service")
async def test_download_tsv_format(mock_service):
    """Test downloading in TSV format."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        task_id = "test_download_tsv"
        mock_service.get_task.return_value = {
            "status": "completed",
            "progress": 1,
            "total": 1,
            "result": {"entries": [{"field": "value"}], "total": 1},
        }

        response = await client.get(f"/api/augment/{task_id}/download?format=tsv")

        assert response.status_code == 200
        assert "text/tab-separated-values" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_download_invalid_format_rejected():
    """Test that invalid download format is rejected."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.get("/api/augment/some_task/download?format=invalid")

        # Should return 422 for validation error
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_empty_file_rejected():
    """Test that empty files are handled appropriately."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        content = b""
        files = {"file": ("empty.jsonl", content, "application/x-ndjson")}

        response = await client.post("/api/augment/upload", files=files)

        # Empty file should either return 200 with 0 entries or 400
        # depending on implementation
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            assert data["entry_count"] == 0


@pytest.mark.asyncio
async def test_upload_malformed_json_rejected():
    """Test that malformed JSON content is rejected."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        content = b'{"incomplete": json'
        files = {"file": ("bad.json", content, "application/json")}

        response = await client.post("/api/augment/upload", files=files)

        assert response.status_code == 400


# ============================================================================
# Integration Tests for Format Conversion in Downloads
# ============================================================================


@pytest.mark.asyncio
@patch("ollaforge.web.routes.generation.generation_service")
async def test_generation_download_all_formats(mock_service):
    """Test that generation download supports all formats."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        task_id = "gen_format_test"
        mock_service.get_task.return_value = {
            "status": "completed",
            "progress": 2,
            "total": 2,
            "result": {
                "entries": [
                    {"instruction": "q1", "input": "", "output": "a1"},
                    {"instruction": "q2", "input": "", "output": "a2"},
                ],
                "total": 2,
            },
        }

        for fmt in ["jsonl", "json", "csv", "tsv"]:
            response = await client.get(
                f"/api/generate/{task_id}/download?format={fmt}"
            )

            assert (
                response.status_code == 200
            ), f"Download in {fmt} format should succeed"
            assert (
                len(response.content) > 0
            ), f"Download in {fmt} format should have content"
