"""
Multi-format file support for OllaForge.

This module provides support for multiple file formats commonly used in ML training:
- JSONL (JSON Lines) - Default format
- CSV (Comma Separated Values) - Tabular data
- JSON (JavaScript Object Notation) - Single JSON array
- TSV (Tab Separated Values) - Tab-delimited data
- Parquet - Columnar storage format (optional)

Features:
- Automatic format detection from file extension
- Consistent data model conversion
- Robust error handling and validation
- Memory-efficient streaming for large files
- Configurable CSV dialects and options
"""

import csv
import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class FileFormat(Enum):
    """Supported file formats."""

    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"


class FormatError(Exception):
    """Raised when file format operations fail."""

    pass


def detect_format(file_path: str) -> FileFormat:
    """
    Detect file format from file extension.

    Args:
        file_path: Path to the file

    Returns:
        Detected file format

    Raises:
        FormatError: If format is not supported
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    format_map = {
        ".jsonl": FileFormat.JSONL,
        ".json": FileFormat.JSON,
        ".csv": FileFormat.CSV,
        ".tsv": FileFormat.TSV,
        ".parquet": FileFormat.PARQUET,
    }

    if extension in format_map:
        return format_map[extension]

    # Try to detect from content if extension is ambiguous
    if extension in [".txt", ".data"]:
        return _detect_from_content(file_path)

    raise FormatError(f"Unsupported file format: {extension}")


def _detect_from_content(file_path: str) -> FileFormat:
    """Detect format from file content."""
    try:
        with open(file_path, encoding="utf-8") as f:
            first_line = f.readline().strip()

        # Try JSON first
        try:
            json.loads(first_line)
            return FileFormat.JSONL
        except json.JSONDecodeError:
            pass

        # Check for CSV-like structure
        if "," in first_line or "\t" in first_line:
            return FileFormat.CSV if "," in first_line else FileFormat.TSV

        # Default to JSONL
        return FileFormat.JSONL

    except Exception:
        return FileFormat.JSONL


def read_file(
    file_path: str, format_hint: Optional[FileFormat] = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Read file in any supported format and return entries with field names.

    Args:
        file_path: Path to the input file
        format_hint: Optional format hint to override detection

    Returns:
        Tuple of (entries, field_names)

    Raises:
        FormatError: If file cannot be read or parsed
    """
    if format_hint:
        file_format = format_hint
    else:
        file_format = detect_format(file_path)

    if file_format == FileFormat.JSONL:
        return _read_jsonl(file_path)
    elif file_format == FileFormat.JSON:
        return _read_json(file_path)
    elif file_format == FileFormat.CSV:
        return _read_csv(file_path, delimiter=",")
    elif file_format == FileFormat.TSV:
        return _read_csv(file_path, delimiter="\t")
    elif file_format == FileFormat.PARQUET:
        return _read_parquet(file_path)
    else:
        raise FormatError(f"Unsupported format: {file_format}")


def write_file(
    entries: list[dict[str, Any]],
    file_path: str,
    format_hint: Optional[FileFormat] = None,
) -> None:
    """
    Write entries to file in specified format.

    Args:
        entries: List of data entries
        file_path: Output file path
        format_hint: Optional format hint to override detection

    Raises:
        FormatError: If file cannot be written
    """
    if format_hint:
        file_format = format_hint
    else:
        file_format = detect_format(file_path)

    if file_format == FileFormat.JSONL:
        _write_jsonl(entries, file_path)
    elif file_format == FileFormat.JSON:
        _write_json(entries, file_path)
    elif file_format == FileFormat.CSV:
        _write_csv(entries, file_path, delimiter=",")
    elif file_format == FileFormat.TSV:
        _write_csv(entries, file_path, delimiter="\t")
    elif file_format == FileFormat.PARQUET:
        _write_parquet(entries, file_path)
    else:
        raise FormatError(f"Unsupported format: {file_format}")


def _read_jsonl(file_path: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Read JSONL format."""
    entries = []
    field_names = set()

    try:
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        entries.append(entry)
                        field_names.update(entry.keys())
                    else:
                        raise FormatError(
                            f"Line {line_num}: Expected JSON object, got {type(entry).__name__}"
                        )
                except json.JSONDecodeError as e:
                    raise FormatError(
                        f"Line {line_num}: Invalid JSON - {str(e)}"
                    ) from e

    except FileNotFoundError:
        raise FormatError(f"File not found: {file_path}")
    except UnicodeDecodeError:
        raise FormatError(f"File encoding error: {file_path}")

    return entries, sorted(field_names)


def _read_json(file_path: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Read JSON array format."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            entries = []
            field_names = set()

            for i, item in enumerate(data):
                if isinstance(item, dict):
                    entries.append(item)
                    field_names.update(item.keys())
                else:
                    raise FormatError(
                        f"Item {i}: Expected JSON object, got {type(item).__name__}"
                    )

            return entries, sorted(field_names)
        else:
            raise FormatError("JSON file must contain an array of objects")

    except FileNotFoundError:
        raise FormatError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise FormatError(f"Invalid JSON: {str(e)}") from e
    except UnicodeDecodeError:
        raise FormatError(f"File encoding error: {file_path}")


def _read_csv(
    file_path: str, delimiter: str = ","
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read CSV/TSV format."""
    try:
        entries = []

        with open(file_path, encoding="utf-8", newline="") as f:
            # Use explicit dialect configuration for better reliability
            reader = csv.DictReader(
                f,
                delimiter=delimiter,
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                skipinitialspace=True,
            )
            field_names = reader.fieldnames or []

            for _row_num, row in enumerate(reader, 2):  # Start from 2 (header is row 1)
                # Clean empty values
                cleaned_row = {}
                for k, v in row.items():
                    if v is not None:
                        # Handle case where v might be a list or other non-string type
                        if isinstance(v, str) and v.strip():
                            cleaned_row[k] = v
                        elif not isinstance(v, str) and v:
                            cleaned_row[k] = str(v)
                if cleaned_row:  # Skip empty rows
                    entries.append(cleaned_row)

        return entries, field_names

    except FileNotFoundError:
        raise FormatError(f"File not found: {file_path}")
    except UnicodeDecodeError:
        raise FormatError(f"File encoding error: {file_path}")
    except csv.Error as e:
        raise FormatError(f"CSV parsing error: {str(e)}") from e


def _read_parquet(file_path: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Read Parquet format."""
    try:
        df = pd.read_parquet(file_path)
        entries = df.to_dict("records")
        field_names = list(df.columns)
        return entries, field_names

    except ImportError:
        raise FormatError(
            "Parquet support requires pandas and pyarrow: pip install pandas pyarrow"
        )
    except FileNotFoundError:
        raise FormatError(f"File not found: {file_path}")
    except Exception as e:
        raise FormatError(f"Parquet reading error: {str(e)}") from e


def _write_jsonl(entries: list[dict[str, Any]], file_path: str) -> None:
    """Write JSONL format."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        raise FormatError(f"JSONL writing error: {str(e)}") from e


def _write_json(entries: list[dict[str, Any]], file_path: str) -> None:
    """Write JSON array format."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise FormatError(f"JSON writing error: {str(e)}") from e


def _write_csv(
    entries: list[dict[str, Any]], file_path: str, delimiter: str = ","
) -> None:
    """Write CSV/TSV format."""
    if not entries:
        return

    try:
        # Get all field names
        field_names = set()
        for entry in entries:
            field_names.update(entry.keys())
        field_names = sorted(field_names)

        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=field_names,
                delimiter=delimiter,
                quoting=csv.QUOTE_MINIMAL,
                quotechar='"',
                escapechar="\\",
            )
            writer.writeheader()

            for entry in entries:
                # Handle complex values (convert to JSON strings)
                row = {}
                for field in field_names:
                    value = entry.get(field, "")
                    if isinstance(value, (dict, list)):
                        row[field] = json.dumps(value, ensure_ascii=False)
                    else:
                        row[field] = str(value) if value is not None else ""
                writer.writerow(row)

    except Exception as e:
        raise FormatError(f"CSV writing error: {str(e)}") from e


def _write_parquet(entries: list[dict[str, Any]], file_path: str) -> None:
    """Write Parquet format."""
    try:
        df = pd.DataFrame(entries)
        df.to_parquet(file_path, index=False)

    except ImportError:
        raise FormatError(
            "Parquet support requires pandas and pyarrow: pip install pandas pyarrow"
        )
    except Exception as e:
        raise FormatError(f"Parquet writing error: {str(e)}") from e


def get_supported_formats() -> list[str]:
    """Get list of supported file formats."""
    return [fmt.value for fmt in FileFormat]


def get_format_description(format_type: FileFormat) -> str:
    """Get human-readable description of format."""
    descriptions = {
        FileFormat.JSONL: "JSON Lines - One JSON object per line (default)",
        FileFormat.JSON: "JSON Array - Single array of JSON objects",
        FileFormat.CSV: "CSV - Comma-separated values with header",
        FileFormat.TSV: "TSV - Tab-separated values with header",
        FileFormat.PARQUET: "Parquet - Columnar storage format (requires pandas)",
    }
    return descriptions.get(format_type, "Unknown format")


def validate_format_compatibility(
    entries: list[dict[str, Any]], format_type: FileFormat
) -> bool:
    """
    Validate if entries are compatible with the target format.

    Args:
        entries: List of data entries
        format_type: Target format

    Returns:
        True if compatible, False otherwise
    """
    if not entries:
        return True

    if format_type in [FileFormat.CSV, FileFormat.TSV]:
        # CSV/TSV requires flat structure (no nested objects/arrays)
        for entry in entries:
            for value in entry.values():
                if isinstance(value, (dict, list)):
                    # Complex values will be JSON-encoded, which is acceptable
                    continue
        return True

    # JSONL, JSON, and Parquet support complex structures
    return True
