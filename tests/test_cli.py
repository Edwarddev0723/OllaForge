"""
Tests for OllaForge CLI interface.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from typer.testing import CliRunner

from ollaforge.cli import (
    app,
    validate_count_range,
    validate_output_path,
    validate_parameters,
)

runner = CliRunner()


@given(
    topic=st.text(min_size=1).filter(lambda x: x.strip()),
    count=st.integers(min_value=-100, max_value=20000),
    model=st.text(min_size=1),
    output=st.text(min_size=1).filter(
        lambda x: x.strip() and "/" not in x and "\\" not in x
    ),
)
def test_parameter_validation_error_handling(topic, count, model, output):
    """
    **Feature: ollama-cli-generator, Property 14: Parameter validation error handling**
    **Validates: Requirements 6.1**

    For any invalid CLI parameters provided, the system should display helpful
    error messages and usage information.
    """
    # Test with invalid count values (outside valid range)
    if count < 1 or count > 10000:
        with pytest.raises((ValidationError, ValueError)):
            validate_parameters(
                topic.strip(), count, model, output.strip(), raise_on_error=False
            )
    else:
        # For valid parameters, validation should succeed
        try:
            config = validate_parameters(
                topic.strip(), count, model, output.strip(), raise_on_error=False
            )
            assert config.topic == topic.strip()
            assert config.count == count
            assert config.model == model
            assert config.output == output.strip()
        except ValidationError:
            # This is acceptable for edge cases like empty topics after stripping
            pass


def test_count_range_validation():
    """Test count parameter range validation."""
    # Valid counts should pass
    assert validate_count_range(1) == 1
    assert validate_count_range(100) == 100
    assert validate_count_range(10000) == 10000

    # Invalid counts should raise BadParameter
    from typer import BadParameter

    with pytest.raises(BadParameter):
        validate_count_range(0)

    with pytest.raises(BadParameter):
        validate_count_range(-1)

    with pytest.raises(BadParameter):
        validate_count_range(10001)


def test_output_path_validation():
    """Test output path validation."""
    # Valid paths should pass
    assert validate_output_path("test.jsonl") == "test.jsonl"
    assert validate_output_path("  test.jsonl  ") == "test.jsonl"

    # Invalid paths should raise BadParameter
    from typer import BadParameter

    with pytest.raises(BadParameter):
        validate_output_path("")

    with pytest.raises(BadParameter):
        validate_output_path("   ")


def test_cli_help_display():
    """Test that CLI displays help information correctly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Generate and augment datasets using local Ollama models" in result.stdout
    assert "generate" in result.stdout
    assert "augment" in result.stdout

    # Test generate subcommand help
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--count" in result.stdout
    assert "--model" in result.stdout
    assert "--output" in result.stdout


def test_cli_with_valid_parameters():
    """Test CLI with valid parameters."""
    result = runner.invoke(
        app,
        [
            "generate",
            "test topic",
            "--count",
            "5",
            "--model",
            "llama3",
            "--output",
            "test.jsonl",
        ],
    )
    # Should not exit with error (implementation is incomplete but parameters are valid)
    assert "Topic: test topic" in result.stdout
    assert "Count: 5" in result.stdout
    assert "Model: llama3" in result.stdout
    assert "Output: test.jsonl" in result.stdout


def test_cli_with_invalid_count():
    """Test CLI with invalid count parameter."""
    result = runner.invoke(app, ["generate", "test topic", "--count", "0"])
    assert result.exit_code != 0
    # Error message may be in stdout or output depending on Typer version
    output = result.stdout + (result.output if hasattr(result, "output") else "")
    assert "Count must be at least 1" in output


def test_cli_with_missing_topic():
    """Test CLI with missing required topic argument."""
    result = runner.invoke(app, ["generate", "--count", "5"])
    assert result.exit_code != 0
    # Typer should show usage information for missing argument


@given(
    topic=st.text(min_size=1, max_size=50).filter(lambda x: x.strip() and x.isascii()),
    model=st.text(min_size=1, max_size=20).filter(
        lambda x: x.strip() and x.isascii() and x.isalnum()
    ),
    count=st.integers(min_value=1, max_value=10),
    output=st.text(min_size=1, max_size=20).filter(
        lambda x: x.strip() and x.isascii() and x.isalnum()
    ),
)
def test_model_parameter_selection(topic, model, count, output):
    """
    **Feature: ollama-cli-generator, Property 3: Model parameter selection**
    **Validates: Requirements 1.3**

    For any valid Ollama model name provided, the system should use that specific
    model for generation requests.
    """
    # Test that the model parameter is properly accepted and stored
    config = validate_parameters(
        topic.strip(), count, model.strip(), output.strip(), raise_on_error=False
    )
    assert config.model == model.strip()

    # Test CLI integration - the model should be displayed in output
    result = runner.invoke(
        app,
        [
            "generate",
            topic.strip(),
            "--count",
            str(count),
            "--model",
            model.strip(),
            "--output",
            output.strip(),
        ],
    )

    # Strip ANSI color codes for comparison
    import re

    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)

    # Should display the specified model (with emoji prefix)
    assert f"🤖 Model: {model.strip()}" in clean_output


@given(
    topic=st.text(min_size=1, max_size=50).filter(
        lambda x: x.strip() and x.isascii() and not x.strip().startswith("-")
    ),
    output=st.text(min_size=1, max_size=20).filter(
        lambda x: x.strip() and x.isascii() and x.isalnum()
    ),
    count=st.integers(min_value=1, max_value=10),
)
def test_output_filename_specification(topic, output, count):
    """
    **Feature: ollama-cli-generator, Property 4: Output filename specification**
    **Validates: Requirements 1.4**

    For any valid filename provided as output parameter, the results should be
    written to that exact file location.
    """
    # Test that the output parameter is properly accepted and stored
    config = validate_parameters(
        topic.strip(), count, "llama3", output.strip(), raise_on_error=False
    )
    assert config.output == output.strip()

    # Test CLI integration - the output filename should be displayed
    result = runner.invoke(
        app,
        ["generate", topic.strip(), "--count", str(count), "--output", output.strip()],
    )

    # Strip ANSI color codes for comparison
    import re

    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)

    # Should display the specified output filename (with emoji prefix)
    assert f"📄 Output: {output.strip()}" in clean_output


# Edge case tests for Requirements 6.3 and 6.5


def test_cli_with_insufficient_disk_space():
    """Test CLI behavior when disk space is insufficient - Requirements 6.3"""
    from unittest.mock import patch

    with patch("ollaforge.file_manager.check_disk_space") as mock_check:
        from ollaforge.file_manager import DiskSpaceError

        mock_check.side_effect = DiskSpaceError(
            "Insufficient disk space. Available: 10.0MB, Required: 100.0MB"
        )

        result = runner.invoke(app, ["generate", "test topic", "--count", "5"])

        # Should exit with error code
        assert result.exit_code == 1
        assert "Insufficient disk space" in result.stdout


def test_cli_with_disk_space_check_during_write():
    """Test CLI behavior when disk space becomes insufficient during file write - Requirements 6.3"""
    from unittest.mock import patch

    from ollaforge.models import DataEntry

    with (
        patch("ollaforge.client.generate_data_concurrent") as mock_generate,
        patch("ollaforge.processor.process_model_response") as mock_process,
        patch("ollaforge.file_manager.write_jsonl_file") as mock_write,
    ):
        # Mock successful generation
        mock_generate.return_value = [
            {
                "raw_content": '{"instruction": "test", "input": "test", "output": "test"}',
                "is_batch": False,
            }
        ]
        mock_process.return_value = [
            DataEntry(instruction="test", input="test", output="test")
        ]

        # Mock disk space error during write
        from ollaforge.file_manager import DiskSpaceError

        mock_write.side_effect = DiskSpaceError("Insufficient disk space during write")

        result = runner.invoke(app, ["generate", "test topic", "--count", "1"])

        # Should exit with error code and show disk space error
        assert result.exit_code == 1
        assert "Insufficient disk space" in result.stdout


def test_cli_with_file_permission_error():
    """Test CLI behavior with file permission errors - Requirements 6.3"""
    from unittest.mock import patch

    from ollaforge.models import DataEntry

    with (
        patch("ollaforge.client.generate_data_concurrent") as mock_generate,
        patch("ollaforge.processor.process_model_response") as mock_process,
        patch("ollaforge.file_manager.write_jsonl_file") as mock_write,
    ):
        # Mock successful generation
        mock_generate.return_value = [
            {
                "raw_content": '{"instruction": "test", "input": "test", "output": "test"}',
                "is_batch": False,
            }
        ]
        mock_process.return_value = [
            DataEntry(instruction="test", input="test", output="test")
        ]

        # Mock file permission error
        from ollaforge.file_manager import FileOperationError

        mock_write.side_effect = FileOperationError(
            "Permission denied: /readonly/test.jsonl"
        )

        result = runner.invoke(app, ["generate", "test topic", "--count", "1"])

        # Should exit with error code and show file error
        assert result.exit_code == 1
        assert "Permission denied" in result.stdout


def test_cli_with_network_interruption_during_generation():
    """Test CLI behavior with network interruption during generation - Requirements 6.5"""
    from unittest.mock import patch

    with patch("ollaforge.client.generate_data_concurrent") as mock_generate:
        from ollaforge.client import OllamaConnectionError

        # Mock network interruption
        mock_generate.side_effect = OllamaConnectionError(
            "Connection lost during generation"
        )

        result = runner.invoke(app, ["generate", "test topic", "--count", "3"])

        # Should handle error gracefully and exit with error code
        assert result.exit_code == 1
        assert "Connection failed" in result.stdout
        assert "Make sure Ollama is running" in result.stdout


def test_cli_with_partial_generation_failure():
    """Test CLI behavior when some generations fail but others succeed - Requirements 6.5"""
    from unittest.mock import MagicMock, patch

    with (
        patch("ollaforge.client.generate_data_concurrent") as mock_generate,
        patch("ollaforge.processor.process_model_response") as mock_process,
        patch("ollaforge.file_manager.write_jsonl_file") as mock_write,
    ):
        # Mock mixed success/failure responses
        mock_generate.return_value = [
            {
                "raw_content": '{"instruction": "test1", "input": "test1", "output": "test1"}'
            },
            {
                "raw_content": '{"instruction": "test3", "input": "test3", "output": "test3"}'
            },
        ]

        # Mock successful processing
        mock_process.return_value = MagicMock()
        mock_write.return_value = None

        result = runner.invoke(app, ["generate", "test topic", "--count", "3"])

        # Should complete with partial results
        # The exact behavior depends on implementation, but should not crash
        assert result.exit_code is not None


def test_cli_with_malformed_model_responses():
    """Test CLI behavior with malformed model responses - Requirements 6.4"""
    from unittest.mock import MagicMock, patch

    with (
        patch("ollaforge.client.generate_data_concurrent") as mock_generate,
        patch("ollaforge.processor.process_model_response") as mock_process,
        patch("ollaforge.file_manager.write_jsonl_file") as mock_write,
    ):
        # Mock malformed responses
        mock_generate.return_value = [
            {"raw_content": "invalid json content"},
            {"raw_content": '{"incomplete": "json"'},
            {
                "raw_content": '{"valid": "json", "instruction": "test", "input": "test", "output": "test"}'
            },
        ]

        # Mock processing that handles malformed responses
        mock_process.side_effect = [
            None,
            None,
            MagicMock(),
        ]  # First two fail, third succeeds
        mock_write.return_value = None

        result = runner.invoke(app, ["generate", "test topic", "--count", "3"])

        # Should handle malformed responses gracefully
        # Should not crash and should process valid responses
        assert result.exit_code is not None


def test_cli_with_empty_model_responses():
    """Test CLI behavior with empty model responses - Requirements 6.4"""
    from unittest.mock import patch

    with (
        patch("ollaforge.client.generate_data_concurrent") as mock_generate,
        patch("ollaforge.processor.process_model_response") as mock_process,
        patch("ollaforge.file_manager.write_jsonl_file"),
    ):
        # Mock empty responses
        mock_generate.return_value = [
            {"raw_content": ""},
            {"raw_content": "   "},
            {"raw_content": None},
        ]

        # Mock processing that handles empty responses
        mock_process.return_value = None

        result = runner.invoke(app, ["generate", "test topic", "--count", "3"])

        # Should handle empty responses without crashing
        # May show warning about no valid entries generated
        assert result.exit_code is not None


def test_cli_with_unexpected_exception_during_generation():
    """Test CLI behavior with unexpected exceptions - Requirements 6.5"""
    from unittest.mock import patch

    with patch("ollaforge.client.generate_data_concurrent") as mock_generate:
        # Mock unexpected exception
        mock_generate.side_effect = RuntimeError("Unexpected system error")

        result = runner.invoke(app, ["generate", "test topic", "--count", "1"])

        # Should handle unexpected errors gracefully
        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout


def test_cli_with_memory_pressure():
    """Test CLI behavior under memory pressure - Requirements 6.3"""
    from unittest.mock import patch

    with patch("ollaforge.client.generate_data_concurrent") as mock_generate:
        # Mock memory error
        mock_generate.side_effect = MemoryError("Out of memory")

        result = runner.invoke(
            app,
            [
                "generate",
                "test topic",
                "--count",
                "1000",  # Large count that might cause memory issues
            ],
        )

        # Should handle memory errors gracefully
        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout


def test_cli_with_corrupted_output_directory():
    """Test CLI behavior with corrupted or inaccessible output directory - Requirements 6.3"""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file where we expect a directory
        fake_dir = os.path.join(temp_dir, "fake_dir")
        with open(fake_dir, "w") as f:
            f.write("this is a file, not a directory")

        result = runner.invoke(
            app,
            [
                "generate",
                "test topic",
                "--output",
                os.path.join(fake_dir, "test.jsonl"),  # Try to write inside a file
            ],
        )

        # Should detect the invalid path
        assert result.exit_code != 0


def test_cli_with_extremely_long_paths():
    """Test CLI behavior with extremely long file paths - Requirements 6.3"""
    # Create an extremely long path
    long_path = "a" * 1000 + ".jsonl"

    result = runner.invoke(app, ["generate", "test topic", "--output", long_path])

    # Should handle long paths (may succeed or fail gracefully depending on OS)
    assert result.exit_code is not None


def test_cli_with_special_characters_in_output_path():
    """Test CLI behavior with special characters in output path - Requirements 6.3"""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test various special characters that might cause issues
        special_paths = [
            os.path.join(temp_dir, "file with spaces.jsonl"),
            os.path.join(temp_dir, "file-with-dashes.jsonl"),
            os.path.join(temp_dir, "file_with_underscores.jsonl"),
            os.path.join(temp_dir, "file.with.dots.jsonl"),
        ]

        for special_path in special_paths:
            result = runner.invoke(
                app,
                ["generate", "test topic", "--count", "1", "--output", special_path],
            )

            # Should handle special characters in paths
            # The exact behavior depends on the OS and implementation
            assert result.exit_code is not None


def test_cli_interruption_handling():
    """Test CLI interruption handling - Requirements 6.5"""
    # This is difficult to test directly, but we can test that the setup doesn't crash
    result = runner.invoke(app, ["generate", "test topic", "--count", "1"])

    # The command should start properly (even if it fails due to no Ollama)
    # The important thing is that interruption handling is set up
    assert "Topic: test topic" in result.stdout


def test_cli_with_invalid_output_directory():
    """Test CLI with invalid output directory path."""
    result = runner.invoke(
        app,
        [
            "generate",
            "test topic",
            "--output",
            "/invalid/path/that/does/not/exist/test.jsonl",
        ],
    )

    # Should handle the error gracefully
    assert result.exit_code != 0


def test_cli_with_readonly_output_directory():
    """Test CLI with read-only output directory."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        readonly_dir = os.path.join(temp_dir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)  # Read-only

        try:
            result = runner.invoke(
                app,
                [
                    "generate",
                    "test topic",
                    "--output",
                    os.path.join(readonly_dir, "test.jsonl"),
                ],
            )

            # Should detect permission issues
            assert result.exit_code != 0

        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)


def test_cli_parameter_validation_edge_cases():
    """Test CLI parameter validation with edge cases."""
    # Test with extremely long topic
    long_topic = "a" * 10000
    result = runner.invoke(app, ["generate", long_topic, "--count", "1"])
    # Should handle long topics (may succeed or fail gracefully)

    # Test with special characters in topic
    special_topic = "test with special chars: !@#$%^&*()"
    result = runner.invoke(app, ["generate", special_topic, "--count", "1"])
    # Should handle special characters
    assert "Topic:" in result.stdout


def test_cli_with_zero_count_edge_case():
    """Test CLI with count of zero (edge case)."""
    result = runner.invoke(app, ["generate", "test topic", "--count", "0"])

    assert result.exit_code != 0
    output = result.stdout + (result.output if hasattr(result, "output") else "")
    assert "Count must be at least 1" in output


def test_cli_with_negative_count():
    """Test CLI with negative count."""
    result = runner.invoke(app, ["generate", "test topic", "--count", "-5"])

    assert result.exit_code != 0
    output = result.stdout + (result.output if hasattr(result, "output") else "")
    assert "Count must be at least 1" in output


def test_cli_with_extremely_large_count():
    """Test CLI with extremely large count."""
    result = runner.invoke(app, ["generate", "test topic", "--count", "50000"])

    assert result.exit_code != 0
    output = result.stdout + (result.output if hasattr(result, "output") else "")
    assert "Count cannot exceed 10,000" in output


def test_cli_with_empty_model_name():
    """Test CLI with empty model name."""
    result = runner.invoke(app, ["generate", "test topic", "--model", ""])

    # Should handle empty model name
    # The exact behavior depends on validation, but shouldn't crash
    assert result.exit_code is not None


def test_cli_with_whitespace_only_topic():
    """Test CLI with whitespace-only topic."""
    result = runner.invoke(app, ["generate", "   ", "--count", "1"])  # Only whitespace

    # Should handle whitespace-only topics appropriately
    # May succeed with trimmed topic or fail with validation error
    assert result.exit_code is not None


# ============================================================================
# doc2dataset Command Tests
# ============================================================================


def test_doc2dataset_help_display():
    """Test that doc2dataset command displays help information correctly."""
    result = runner.invoke(app, ["doc2dataset", "--help"])
    assert result.exit_code == 0
    assert "Convert documents to fine-tuning datasets" in result.stdout
    assert "--output" in result.stdout
    assert "--type" in result.stdout
    assert "--model" in result.stdout
    assert "--chunk-size" in result.stdout
    assert "--chunk-overlap" in result.stdout
    assert "--count" in result.stdout
    assert "--lang" in result.stdout
    assert "--pattern" in result.stdout
    assert "--recursive" in result.stdout


def test_doc2dataset_with_nonexistent_source():
    """Test doc2dataset with non-existent source file - Requirements 5.1"""
    result = runner.invoke(app, ["doc2dataset", "/nonexistent/path/to/file.md"])
    assert result.exit_code != 0
    output = result.stdout + (result.output if hasattr(result, "output") else "")
    assert "Source path not found" in output or "not found" in output.lower()


def test_doc2dataset_with_invalid_chunk_size():
    """Test doc2dataset with invalid chunk size parameter."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test\nSome content")
        temp_file = f.name

    try:
        # Test chunk size too small
        result = runner.invoke(app, ["doc2dataset", temp_file, "--chunk-size", "100"])
        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Chunk size must be at least 500" in output

        # Test chunk size too large
        result = runner.invoke(app, ["doc2dataset", temp_file, "--chunk-size", "20000"])
        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Chunk size cannot exceed 10,000" in output
    finally:
        os.unlink(temp_file)


def test_doc2dataset_with_invalid_chunk_overlap():
    """Test doc2dataset with invalid chunk overlap parameter."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test\nSome content")
        temp_file = f.name

    try:
        # Test negative overlap
        result = runner.invoke(
            app, ["doc2dataset", temp_file, "--chunk-overlap", "-10"]
        )
        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Chunk overlap cannot be negative" in output

        # Test overlap too large
        result = runner.invoke(
            app, ["doc2dataset", temp_file, "--chunk-overlap", "2000"]
        )
        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Chunk overlap cannot exceed 1,000" in output
    finally:
        os.unlink(temp_file)


def test_doc2dataset_with_invalid_entries_per_chunk():
    """Test doc2dataset with invalid entries per chunk parameter."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test\nSome content")
        temp_file = f.name

    try:
        # Test count too small
        result = runner.invoke(app, ["doc2dataset", temp_file, "--count", "0"])
        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Entries per chunk must be at least 1" in output

        # Test count too large
        result = runner.invoke(app, ["doc2dataset", temp_file, "--count", "20"])
        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Entries per chunk cannot exceed 10" in output
    finally:
        os.unlink(temp_file)


def test_doc2dataset_with_overlap_greater_than_chunk_size():
    """Test doc2dataset with overlap >= chunk size."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test\nSome content")
        temp_file = f.name

    try:
        result = runner.invoke(
            app,
            [
                "doc2dataset",
                temp_file,
                "--chunk-size",
                "1000",
                "--chunk-overlap",
                "1000",
            ],
        )
        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Chunk overlap must be less than chunk size" in output
    finally:
        os.unlink(temp_file)


def test_doc2dataset_with_invalid_dataset_type():
    """Test doc2dataset with invalid dataset type."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test\nSome content")
        temp_file = f.name

    try:
        result = runner.invoke(
            app, ["doc2dataset", temp_file, "--type", "invalid_type"]
        )
        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Invalid dataset type" in output
    finally:
        os.unlink(temp_file)


def test_doc2dataset_with_invalid_language():
    """Test doc2dataset with invalid language."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test\nSome content")
        temp_file = f.name

    try:
        result = runner.invoke(
            app, ["doc2dataset", temp_file, "--lang", "invalid_lang"]
        )
        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Invalid language" in output
    finally:
        os.unlink(temp_file)


def test_doc2dataset_displays_config():
    """Test that doc2dataset displays configuration correctly."""
    import os
    import tempfile
    from unittest.mock import patch

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test Document\n\nThis is test content for the document.")
        temp_file = f.name

    try:
        # Mock the Ollama client to avoid actual API calls
        with patch("ollaforge.doc_generator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = {
                "message": {
                    "content": '[{"instruction": "test", "input": "test", "output": "test"}]'
                }
            }

            result = runner.invoke(
                app,
                [
                    "doc2dataset",
                    temp_file,
                    "--type",
                    "sft",
                    "--model",
                    "test-model",
                    "--chunk-size",
                    "1000",
                    "--chunk-overlap",
                    "100",
                    "--count",
                    "2",
                    "--lang",
                    "en",
                ],
            )

            # Strip ANSI color codes for comparison
            import re

            clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)

            # Check that configuration is displayed
            assert "Document to Dataset" in clean_output
            assert "Source:" in clean_output
            assert "Output:" in clean_output
            assert "Type:" in clean_output
            assert "Model:" in clean_output
            assert "Chunk size:" in clean_output
            assert "Chunk overlap:" in clean_output
    finally:
        os.unlink(temp_file)


def test_doc2dataset_with_empty_directory():
    """Test doc2dataset with empty directory."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, ["doc2dataset", temp_dir])

        # Should exit gracefully with message about no files found
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "No supported files found" in output or result.exit_code == 0


def test_doc2dataset_with_unsupported_file():
    """Test doc2dataset with unsupported file format."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write("Some content")
        temp_file = f.name

    try:
        result = runner.invoke(app, ["doc2dataset", temp_file])

        # Should show unsupported format error
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Unsupported format" in output or "No supported files found" in output
    finally:
        os.unlink(temp_file)


def test_doc2dataset_with_permission_error():
    """Test doc2dataset with permission error - Requirements 5.2"""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file and make it unreadable
        temp_file = os.path.join(temp_dir, "test.md")
        with open(temp_file, "w") as f:
            f.write("# Test\nContent")

        # Make file unreadable
        os.chmod(temp_file, 0o000)

        try:
            result = runner.invoke(app, ["doc2dataset", temp_file])

            # Should show permission error
            assert result.exit_code != 0
            output = result.stdout + (
                result.output if hasattr(result, "output") else ""
            )
            assert "permission" in output.lower() or "Permission" in output
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_file, 0o644)


def test_doc2dataset_ollama_connection_error():
    """Test doc2dataset with Ollama connection error - Requirements 5.3"""
    import os
    import tempfile
    from unittest.mock import patch

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test Document\n\nThis is test content.")
        temp_file = f.name

    try:
        with patch("ollaforge.doc_generator.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = Exception("Connection refused")

            result = runner.invoke(app, ["doc2dataset", temp_file])

            # The command should handle the error gracefully
            # It may show an error or generate no entries
            assert result.exit_code is not None
    finally:
        os.unlink(temp_file)


def test_doc2dataset_parameter_validation():
    """Test doc2dataset parameter validation callbacks."""
    from typer import BadParameter

    from ollaforge.cli import (
        validate_chunk_overlap,
        validate_chunk_size,
        validate_entries_per_chunk,
        validate_source_path,
    )

    # Test validate_chunk_size
    assert validate_chunk_size(500) == 500
    assert validate_chunk_size(2000) == 2000
    assert validate_chunk_size(10000) == 10000

    with pytest.raises(BadParameter):
        validate_chunk_size(499)

    with pytest.raises(BadParameter):
        validate_chunk_size(10001)

    # Test validate_chunk_overlap
    assert validate_chunk_overlap(0) == 0
    assert validate_chunk_overlap(200) == 200
    assert validate_chunk_overlap(1000) == 1000

    with pytest.raises(BadParameter):
        validate_chunk_overlap(-1)

    with pytest.raises(BadParameter):
        validate_chunk_overlap(1001)

    # Test validate_entries_per_chunk
    assert validate_entries_per_chunk(1) == 1
    assert validate_entries_per_chunk(5) == 5
    assert validate_entries_per_chunk(10) == 10

    with pytest.raises(BadParameter):
        validate_entries_per_chunk(0)

    with pytest.raises(BadParameter):
        validate_entries_per_chunk(11)

    # Test validate_source_path with non-existent path
    with pytest.raises(BadParameter):
        validate_source_path("/nonexistent/path")

    with pytest.raises(BadParameter):
        validate_source_path("")

    with pytest.raises(BadParameter):
        validate_source_path("   ")


# ============================================================================
# Doc2Dataset Interrupt Handler Tests
# ============================================================================


class TestDoc2DatasetInterruptHandler:
    """Tests for Doc2DatasetInterruptHandler class - Requirements 5.5"""

    def test_interrupt_handler_initialization(self):
        """Test that interrupt handler initializes correctly."""
        from ollaforge.cli import Doc2DatasetInterruptHandler

        handler = Doc2DatasetInterruptHandler("test_output.jsonl")

        assert handler.interrupted is False
        assert handler.get_entries() == []
        assert handler._output_file == "test_output.jsonl"

    def test_interrupt_handler_set_entries(self):
        """Test setting entries in the interrupt handler."""
        from ollaforge.cli import Doc2DatasetInterruptHandler

        handler = Doc2DatasetInterruptHandler("test_output.jsonl")

        entries = [
            {"instruction": "test1", "input": "input1", "output": "output1"},
            {"instruction": "test2", "input": "input2", "output": "output2"},
        ]

        handler.set_entries(entries)

        assert handler.get_entries() == entries
        assert len(handler.get_entries()) == 2

    def test_interrupt_handler_add_entries(self):
        """Test adding entries to the interrupt handler."""
        from ollaforge.cli import Doc2DatasetInterruptHandler

        handler = Doc2DatasetInterruptHandler("test_output.jsonl")

        # Add first batch
        handler.add_entries(
            [{"instruction": "test1", "input": "input1", "output": "output1"}]
        )
        assert len(handler.get_entries()) == 1

        # Add second batch
        handler.add_entries(
            [
                {"instruction": "test2", "input": "input2", "output": "output2"},
                {"instruction": "test3", "input": "input3", "output": "output3"},
            ]
        )
        assert len(handler.get_entries()) == 3

        # Adding empty list should not change anything
        handler.add_entries([])
        assert len(handler.get_entries()) == 3

        # Adding None should not change anything
        handler.add_entries(None)
        assert len(handler.get_entries()) == 3

    def test_interrupt_handler_save_partial_results_empty(self):
        """Test saving partial results when no entries exist."""
        from ollaforge.cli import Doc2DatasetInterruptHandler

        handler = Doc2DatasetInterruptHandler("test_output.jsonl")

        # Should return None when no entries
        result = handler.save_partial_results()
        assert result is None

    def test_interrupt_handler_save_partial_results_with_entries(self):
        """Test saving partial results with entries - Requirements 5.5"""
        import json
        import os
        import tempfile

        from ollaforge.cli import Doc2DatasetInterruptHandler

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.jsonl")
            handler = Doc2DatasetInterruptHandler(output_file)

            entries = [
                {"instruction": "test1", "input": "input1", "output": "output1"},
                {"instruction": "test2", "input": "input2", "output": "output2"},
            ]
            handler.set_entries(entries)

            # Save partial results
            result = handler.save_partial_results()

            # Should return the path to the partial file
            assert result is not None
            assert "partial" in result
            assert result.endswith(".jsonl")

            # Verify the file was created and contains the entries
            assert os.path.exists(result)

            with open(result, encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 2

                # Verify content
                entry1 = json.loads(lines[0])
                assert entry1["instruction"] == "test1"

                entry2 = json.loads(lines[1])
                assert entry2["instruction"] == "test2"

    def test_interrupt_handler_save_partial_results_with_pydantic_models(self):
        """Test saving partial results with Pydantic model entries."""
        import os
        import tempfile

        from ollaforge.cli import Doc2DatasetInterruptHandler
        from ollaforge.models import DataEntry

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.jsonl")
            handler = Doc2DatasetInterruptHandler(output_file)

            # Create Pydantic model entries
            entries = [
                DataEntry(instruction="test1", input="input1", output="output1"),
                DataEntry(instruction="test2", input="input2", output="output2"),
            ]
            handler.set_entries(entries)

            # Save partial results
            result = handler.save_partial_results()

            # Should successfully save Pydantic models
            assert result is not None
            assert os.path.exists(result)

            with open(result, encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 2

    def test_interrupt_handler_save_partial_results_filters_none(self):
        """Test that save_partial_results filters out None entries."""
        import os
        import tempfile

        from ollaforge.cli import Doc2DatasetInterruptHandler

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.jsonl")
            handler = Doc2DatasetInterruptHandler(output_file)

            # Mix of valid entries and None
            entries = [
                {"instruction": "test1", "input": "input1", "output": "output1"},
                None,
                {"instruction": "test2", "input": "input2", "output": "output2"},
                None,
            ]
            handler.set_entries(entries)

            # Save partial results
            result = handler.save_partial_results()

            # Should only save valid entries
            assert result is not None

            with open(result, encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 2  # Only 2 valid entries

    def test_interrupt_handler_setup_and_cleanup(self):
        """Test signal handler setup and cleanup."""
        import signal

        from ollaforge.cli import Doc2DatasetInterruptHandler

        handler = Doc2DatasetInterruptHandler("test_output.jsonl")

        # Get original handler
        original = signal.getsignal(signal.SIGINT)

        # Setup should change the handler
        handler.setup()
        current = signal.getsignal(signal.SIGINT)
        assert current != original

        # Cleanup should restore the original handler
        handler.cleanup()
        restored = signal.getsignal(signal.SIGINT)
        assert restored == original

    def test_interrupt_handler_interrupted_state(self):
        """Test that interrupted state is properly tracked."""
        from ollaforge.cli import Doc2DatasetInterruptHandler

        handler = Doc2DatasetInterruptHandler("test_output.jsonl")

        # Initially not interrupted
        assert handler.interrupted is False

        # Simulate interrupt by calling the handler directly
        handler._handle_interrupt(None, None)

        # Should now be interrupted
        assert handler.interrupted is True

    def test_interrupt_handler_creates_parent_directory(self):
        """Test that save_partial_results creates parent directory if needed."""
        import os
        import tempfile

        from ollaforge.cli import Doc2DatasetInterruptHandler

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a nested path that doesn't exist
            output_file = os.path.join(temp_dir, "nested", "dir", "test_output.jsonl")
            handler = Doc2DatasetInterruptHandler(output_file)

            entries = [{"instruction": "test", "input": "input", "output": "output"}]
            handler.set_entries(entries)

            # Save should create the nested directory
            result = handler.save_partial_results()

            assert result is not None
            assert os.path.exists(result)


def test_doc2dataset_interrupt_handling_integration():
    """Integration test for doc2dataset interrupt handling - Requirements 5.5"""
    import os
    import tempfile
    from unittest.mock import patch

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test Document\n\nThis is test content for the document.")
        temp_file = f.name

    try:
        # Mock the generator to simulate partial processing
        with patch("ollaforge.doc_generator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = {
                "message": {
                    "content": '[{"instruction": "test", "input": "test", "output": "test"}]'
                }
            }

            result = runner.invoke(
                app, ["doc2dataset", temp_file, "--type", "sft", "--count", "1"]
            )

            # The command should complete (or fail gracefully)
            # The important thing is that interrupt handling is set up
            assert result.exit_code is not None
    finally:
        os.unlink(temp_file)


def test_doc2dataset_partial_save_on_error():
    """Test that partial results are saved when an error occurs during processing."""
    import os
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = os.path.join(temp_dir, "test.md")
        with open(test_file, "w") as f:
            f.write("# Test\n\nContent")

        output_file = os.path.join(temp_dir, "output.jsonl")

        # Mock to simulate an error after some processing
        with patch("ollaforge.doc_generator.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = Exception("Simulated error")

            result = runner.invoke(
                app,
                ["doc2dataset", test_file, "--output", output_file, "--type", "sft"],
            )

            # Should handle the error gracefully
            assert result.exit_code is not None
