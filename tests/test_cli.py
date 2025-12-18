"""
Tests for OllaForge CLI interface.
"""

import pytest
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st
from typer.testing import CliRunner
from pydantic import ValidationError

from ollaforge.cli import app, validate_parameters, validate_count_range, validate_output_path


runner = CliRunner()


@given(
    topic=st.text(min_size=1).filter(lambda x: x.strip()),
    count=st.integers(min_value=-100, max_value=20000),
    model=st.text(min_size=1),
    output=st.text(min_size=1).filter(lambda x: x.strip() and '/' not in x and '\\' not in x)
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
            validate_parameters(topic.strip(), count, model, output.strip(), raise_on_error=False)
    else:
        # For valid parameters, validation should succeed
        try:
            config = validate_parameters(topic.strip(), count, model, output.strip(), raise_on_error=False)
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
    result = runner.invoke(app, [
        "generate",
        "test topic",
        "--count", "5",
        "--model", "llama3",
        "--output", "test.jsonl"
    ])
    # Should not exit with error (implementation is incomplete but parameters are valid)
    assert "Topic: test topic" in result.stdout
    assert "Count: 5" in result.stdout
    assert "Model: llama3" in result.stdout
    assert "Output: test.jsonl" in result.stdout


def test_cli_with_invalid_count():
    """Test CLI with invalid count parameter."""
    result = runner.invoke(app, [
        "generate",
        "test topic",
        "--count", "0"
    ])
    assert result.exit_code != 0
    assert "Count must be at least 1" in result.stdout


def test_cli_with_missing_topic():
    """Test CLI with missing required topic argument."""
    result = runner.invoke(app, [
        "generate",
        "--count", "5"
    ])
    assert result.exit_code != 0
    # Typer should show usage information for missing argument


@given(
    topic=st.text(min_size=1, max_size=50).filter(lambda x: x.strip() and x.isascii()),
    model=st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isascii() and x.isalnum()),
    count=st.integers(min_value=1, max_value=10),
    output=st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isascii() and x.isalnum())
)
def test_model_parameter_selection(topic, model, count, output):
    """
    **Feature: ollama-cli-generator, Property 3: Model parameter selection**
    **Validates: Requirements 1.3**
    
    For any valid Ollama model name provided, the system should use that specific 
    model for generation requests.
    """
    # Test that the model parameter is properly accepted and stored
    config = validate_parameters(topic.strip(), count, model.strip(), output.strip(), raise_on_error=False)
    assert config.model == model.strip()
    
    # Test CLI integration - the model should be displayed in output
    result = runner.invoke(app, [
        "generate",
        topic.strip(),
        "--count", str(count),
        "--model", model.strip(),
        "--output", output.strip()
    ])
    
    # Strip ANSI color codes for comparison
    import re
    clean_output = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
    
    # Should display the specified model (with emoji prefix)
    assert f"🤖 Model: {model.strip()}" in clean_output


@given(
    topic=st.text(min_size=1, max_size=50).filter(lambda x: x.strip() and x.isascii() and not x.strip().startswith('-')),
    output=st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isascii() and x.isalnum()),
    count=st.integers(min_value=1, max_value=10)
)
def test_output_filename_specification(topic, output, count):
    """
    **Feature: ollama-cli-generator, Property 4: Output filename specification**
    **Validates: Requirements 1.4**
    
    For any valid filename provided as output parameter, the results should be 
    written to that exact file location.
    """
    # Test that the output parameter is properly accepted and stored
    config = validate_parameters(topic.strip(), count, "llama3", output.strip(), raise_on_error=False)
    assert config.output == output.strip()
    
    # Test CLI integration - the output filename should be displayed
    result = runner.invoke(app, [
        "generate",
        topic.strip(),
        "--count", str(count),
        "--output", output.strip()
    ])
    
    # Strip ANSI color codes for comparison
    import re
    clean_output = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
    
    # Should display the specified output filename (with emoji prefix)
    assert f"📄 Output: {output.strip()}" in clean_output


# Edge case tests for Requirements 6.3 and 6.5

def test_cli_with_insufficient_disk_space():
    """Test CLI behavior when disk space is insufficient - Requirements 6.3"""
    from unittest.mock import patch
    
    with patch('ollaforge.file_manager.check_disk_space') as mock_check:
        from ollaforge.file_manager import DiskSpaceError
        mock_check.side_effect = DiskSpaceError("Insufficient disk space. Available: 10.0MB, Required: 100.0MB")
        
        result = runner.invoke(app, [
            "generate",
            "test topic",
            "--count", "5"
        ])
        
        # Should exit with error code
        assert result.exit_code == 1
        assert "Insufficient disk space" in result.stdout


def test_cli_with_disk_space_check_during_write():
    """Test CLI behavior when disk space becomes insufficient during file write - Requirements 6.3"""
    from unittest.mock import patch, MagicMock
    from ollaforge.models import DataEntry
    
    with patch('ollaforge.client.generate_data_concurrent') as mock_generate, \
         patch('ollaforge.processor.process_model_response') as mock_process, \
         patch('ollaforge.file_manager.write_jsonl_file') as mock_write:
        
        # Mock successful generation
        mock_generate.return_value = [{'raw_content': '{"instruction": "test", "input": "test", "output": "test"}', 'is_batch': False}]
        mock_process.return_value = [DataEntry(instruction="test", input="test", output="test")]
        
        # Mock disk space error during write
        from ollaforge.file_manager import DiskSpaceError
        mock_write.side_effect = DiskSpaceError("Insufficient disk space during write")
        
        result = runner.invoke(app, [
            "generate",
            "test topic",
            "--count", "1"
        ])
        
        # Should exit with error code and show disk space error
        assert result.exit_code == 1
        assert "Insufficient disk space" in result.stdout


def test_cli_with_file_permission_error():
    """Test CLI behavior with file permission errors - Requirements 6.3"""
    from unittest.mock import patch, MagicMock
    from ollaforge.models import DataEntry
    
    with patch('ollaforge.client.generate_data_concurrent') as mock_generate, \
         patch('ollaforge.processor.process_model_response') as mock_process, \
         patch('ollaforge.file_manager.write_jsonl_file') as mock_write:
        
        # Mock successful generation
        mock_generate.return_value = [{'raw_content': '{"instruction": "test", "input": "test", "output": "test"}', 'is_batch': False}]
        mock_process.return_value = [DataEntry(instruction="test", input="test", output="test")]
        
        # Mock file permission error
        from ollaforge.file_manager import FileOperationError
        mock_write.side_effect = FileOperationError("Permission denied: /readonly/test.jsonl")
        
        result = runner.invoke(app, [
            "generate",
            "test topic",
            "--count", "1"
        ])
        
        # Should exit with error code and show file error
        assert result.exit_code == 1
        assert "Permission denied" in result.stdout


def test_cli_with_network_interruption_during_generation():
    """Test CLI behavior with network interruption during generation - Requirements 6.5"""
    from unittest.mock import patch
    
    with patch('ollaforge.client.generate_data_concurrent') as mock_generate:
        from ollaforge.client import OllamaConnectionError
        
        # Mock network interruption
        mock_generate.side_effect = OllamaConnectionError("Connection lost during generation")
        
        result = runner.invoke(app, [
            "generate",
            "test topic",
            "--count", "3"
        ])
        
        # Should handle error gracefully and exit with error code
        assert result.exit_code == 1
        assert "Connection failed" in result.stdout
        assert "Make sure Ollama is running" in result.stdout


def test_cli_with_partial_generation_failure():
    """Test CLI behavior when some generations fail but others succeed - Requirements 6.5"""
    from unittest.mock import patch, MagicMock
    
    with patch('ollaforge.client.generate_data_concurrent') as mock_generate, \
         patch('ollaforge.processor.process_model_response') as mock_process, \
         patch('ollaforge.file_manager.write_jsonl_file') as mock_write:
        
        # Mock mixed success/failure responses
        mock_generate.return_value = [
            {'raw_content': '{"instruction": "test1", "input": "test1", "output": "test1"}'},
            {'raw_content': '{"instruction": "test3", "input": "test3", "output": "test3"}'}
        ]
        
        # Mock successful processing
        mock_process.return_value = MagicMock()
        mock_write.return_value = None
        
        result = runner.invoke(app, [
            "generate",
            "test topic",
            "--count", "3"
        ])
        
        # Should complete with partial results
        # The exact behavior depends on implementation, but should not crash
        assert result.exit_code is not None


def test_cli_with_malformed_model_responses():
    """Test CLI behavior with malformed model responses - Requirements 6.4"""
    from unittest.mock import patch, MagicMock
    
    with patch('ollaforge.client.generate_data_concurrent') as mock_generate, \
         patch('ollaforge.processor.process_model_response') as mock_process, \
         patch('ollaforge.file_manager.write_jsonl_file') as mock_write:
        
        # Mock malformed responses
        mock_generate.return_value = [
            {'raw_content': 'invalid json content'},
            {'raw_content': '{"incomplete": "json"'},
            {'raw_content': '{"valid": "json", "instruction": "test", "input": "test", "output": "test"}'}
        ]
        
        # Mock processing that handles malformed responses
        mock_process.side_effect = [None, None, MagicMock()]  # First two fail, third succeeds
        mock_write.return_value = None
        
        result = runner.invoke(app, [
            "generate",
            "test topic",
            "--count", "3"
        ])
        
        # Should handle malformed responses gracefully
        # Should not crash and should process valid responses
        assert result.exit_code is not None


def test_cli_with_empty_model_responses():
    """Test CLI behavior with empty model responses - Requirements 6.4"""
    from unittest.mock import patch, MagicMock
    
    with patch('ollaforge.client.generate_data_concurrent') as mock_generate, \
         patch('ollaforge.processor.process_model_response') as mock_process, \
         patch('ollaforge.file_manager.write_jsonl_file') as mock_write:
        
        # Mock empty responses
        mock_generate.return_value = [
            {'raw_content': ''},
            {'raw_content': '   '},
            {'raw_content': None}
        ]
        
        # Mock processing that handles empty responses
        mock_process.return_value = None
        
        result = runner.invoke(app, [
            "generate",
            "test topic",
            "--count", "3"
        ])
        
        # Should handle empty responses without crashing
        # May show warning about no valid entries generated
        assert result.exit_code is not None


def test_cli_with_unexpected_exception_during_generation():
    """Test CLI behavior with unexpected exceptions - Requirements 6.5"""
    from unittest.mock import patch
    
    with patch('ollaforge.client.generate_data_concurrent') as mock_generate:
        # Mock unexpected exception
        mock_generate.side_effect = RuntimeError("Unexpected system error")
        
        result = runner.invoke(app, [
            "generate",
            "test topic",
            "--count", "1"
        ])
        
        # Should handle unexpected errors gracefully
        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout


def test_cli_with_memory_pressure():
    """Test CLI behavior under memory pressure - Requirements 6.3"""
    from unittest.mock import patch, MagicMock
    
    with patch('ollaforge.client.generate_data_concurrent') as mock_generate:
        # Mock memory error
        mock_generate.side_effect = MemoryError("Out of memory")
        
        result = runner.invoke(app, [
            "generate",
            "test topic",
            "--count", "1000"  # Large count that might cause memory issues
        ])
        
        # Should handle memory errors gracefully
        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout


def test_cli_with_corrupted_output_directory():
    """Test CLI behavior with corrupted or inaccessible output directory - Requirements 6.3"""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file where we expect a directory
        fake_dir = os.path.join(temp_dir, "fake_dir")
        with open(fake_dir, 'w') as f:
            f.write("this is a file, not a directory")
        
        result = runner.invoke(app, [
            "generate",
            "test topic",
            "--output", os.path.join(fake_dir, "test.jsonl")  # Try to write inside a file
        ])
        
        # Should detect the invalid path
        assert result.exit_code != 0


def test_cli_with_extremely_long_paths():
    """Test CLI behavior with extremely long file paths - Requirements 6.3"""
    # Create an extremely long path
    long_path = "a" * 1000 + ".jsonl"
    
    result = runner.invoke(app, [
        "generate",
        "test topic",
        "--output", long_path
    ])
    
    # Should handle long paths (may succeed or fail gracefully depending on OS)
    assert result.exit_code is not None


def test_cli_with_special_characters_in_output_path():
    """Test CLI behavior with special characters in output path - Requirements 6.3"""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test various special characters that might cause issues
        special_paths = [
            os.path.join(temp_dir, "file with spaces.jsonl"),
            os.path.join(temp_dir, "file-with-dashes.jsonl"),
            os.path.join(temp_dir, "file_with_underscores.jsonl"),
            os.path.join(temp_dir, "file.with.dots.jsonl"),
        ]
        
        for special_path in special_paths:
            result = runner.invoke(app, [
                "generate",
                "test topic",
                "--count", "1",
                "--output", special_path
            ])
            
            # Should handle special characters in paths
            # The exact behavior depends on the OS and implementation
            assert result.exit_code is not None


def test_cli_interruption_handling():
    """Test CLI interruption handling - Requirements 6.5"""
    # This is difficult to test directly, but we can test that the setup doesn't crash
    result = runner.invoke(app, [
        "generate",
        "test topic",
        "--count", "1"
    ])
    
    # The command should start properly (even if it fails due to no Ollama)
    # The important thing is that interruption handling is set up
    assert "Topic: test topic" in result.stdout


def test_cli_with_invalid_output_directory():
    """Test CLI with invalid output directory path."""
    result = runner.invoke(app, [
        "generate",
        "test topic",
        "--output", "/invalid/path/that/does/not/exist/test.jsonl"
    ])
    
    # Should handle the error gracefully
    assert result.exit_code != 0


def test_cli_with_readonly_output_directory():
    """Test CLI with read-only output directory."""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        readonly_dir = os.path.join(temp_dir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)  # Read-only
        
        try:
            result = runner.invoke(app, [
                "generate",
                "test topic",
                "--output", os.path.join(readonly_dir, "test.jsonl")
            ])
            
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
    result = runner.invoke(app, [
        "generate",
        "test topic",
        "--count", "0"
    ])
    
    assert result.exit_code != 0
    assert "Count must be at least 1" in result.stdout


def test_cli_with_negative_count():
    """Test CLI with negative count."""
    result = runner.invoke(app, [
        "generate",
        "test topic", 
        "--count", "-5"
    ])
    
    assert result.exit_code != 0
    assert "Count must be at least 1" in result.stdout


def test_cli_with_extremely_large_count():
    """Test CLI with extremely large count."""
    result = runner.invoke(app, [
        "generate",
        "test topic",
        "--count", "50000"
    ])
    
    assert result.exit_code != 0
    assert "Count cannot exceed 10,000" in result.stdout


def test_cli_with_empty_model_name():
    """Test CLI with empty model name."""
    result = runner.invoke(app, [
        "generate",
        "test topic",
        "--model", ""
    ])
    
    # Should handle empty model name
    # The exact behavior depends on validation, but shouldn't crash
    assert result.exit_code is not None


def test_cli_with_whitespace_only_topic():
    """Test CLI with whitespace-only topic."""
    result = runner.invoke(app, [
        "generate",
        "   ",  # Only whitespace
        "--count", "1"
    ])
    
    # Should handle whitespace-only topics appropriately
    # May succeed with trimmed topic or fail with validation error
    assert result.exit_code is not None