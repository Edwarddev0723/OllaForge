"""
Tests for OllaForge file operations and JSONL output management.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ollaforge.file_manager import (
    DiskSpaceError,
    FileOperationError,
    append_jsonl_entries,
    check_disk_space,
    check_file_overwrite,
    create_partial_backup,
    ensure_jsonl_extension,
    estimate_file_size,
    get_file_size,
    is_interrupted,
    read_jsonl_file,
    setup_interruption_handling,
    validate_jsonl_file,
    write_jsonl_file,
)
from ollaforge.models import DataEntry

# Strategy for generating valid DataEntry objects
valid_data_entry_strategy = st.builds(
    DataEntry,
    instruction=st.text(min_size=1),
    input=st.text(min_size=1),
    output=st.text(min_size=1)
)


@given(entries=st.lists(valid_data_entry_strategy, min_size=1, max_size=10))
def test_jsonl_format_compliance(entries):
    """
    **Feature: ollama-cli-generator, Property 9: JSONL format compliance**
    **Validates: Requirements 3.2**

    For any data written to output file, each line should contain a valid JSON object.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_output.jsonl")

        # Write entries to JSONL file
        write_jsonl_file(entries, output_path)

        # Verify file exists
        assert os.path.exists(output_path)

        # Read file line by line and verify each line is valid JSON
        with open(output_path, encoding='utf-8') as f:
            lines = f.readlines()

        # Should have same number of lines as entries
        assert len(lines) == len(entries)

        # Each line should be valid JSON
        for i, line in enumerate(lines):
            line = line.strip()
            assert line, f"Line {i+1} is empty"

            # Should be parseable as JSON
            try:
                parsed = json.loads(line)
                assert isinstance(parsed, dict), f"Line {i+1} is not a JSON object"

                # Should contain required fields
                assert 'instruction' in parsed, f"Line {i+1} missing 'instruction' field"
                assert 'input' in parsed, f"Line {i+1} missing 'input' field"
                assert 'output' in parsed, f"Line {i+1} missing 'output' field"

                # Values should match original entry
                original_entry = entries[i]
                assert parsed['instruction'] == original_entry.instruction
                assert parsed['input'] == original_entry.input
                assert parsed['output'] == original_entry.output

            except json.JSONDecodeError as e:
                pytest.fail(f"Line {i+1} is not valid JSON: {line} - Error: {e}")


@given(entries=st.lists(valid_data_entry_strategy, min_size=1, max_size=10))
def test_output_validation(entries):
    """
    **Feature: ollama-cli-generator, Property 11: Output validation**
    **Validates: Requirements 3.5**

    For any completed generation, all entries in the output file should be valid JSONL format.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_validation.jsonl")

        # Write entries to file
        write_jsonl_file(entries, output_path)

        # Validate the file using our validation function
        is_valid = validate_jsonl_file(output_path)
        assert is_valid, "Generated JSONL file failed validation"

        # Also verify by reading back the entries
        read_entries = read_jsonl_file(output_path)
        assert len(read_entries) == len(entries)

        # Each read entry should be a valid dictionary with required fields
        for i, read_entry in enumerate(read_entries):
            assert isinstance(read_entry, dict)
            assert 'instruction' in read_entry
            assert 'input' in read_entry
            assert 'output' in read_entry

            # Values should match original
            original = entries[i]
            assert read_entry['instruction'] == original.instruction
            assert read_entry['input'] == original.input
            assert read_entry['output'] == original.output


@given(
    entries=st.lists(valid_data_entry_strategy, min_size=1, max_size=5),
    filename=st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126, blacklist_characters='/<>:"|?*\\'),
        min_size=1,
        max_size=50
    ).filter(lambda x: x.strip() and not x.startswith('.') and '\x00' not in x)
)
def test_file_overwrite_handling(entries, filename):
    """
    **Feature: ollama-cli-generator, Property 15: File overwrite handling**
    **Validates: Requirements 6.2**

    For any output file that already exists, the system should handle the overwrite operation appropriately.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, filename.strip())

        # Create initial file with some content
        initial_entries = entries[:len(entries)//2] if len(entries) > 1 else entries
        write_jsonl_file(initial_entries, output_path)

        # Verify file exists
        assert check_file_overwrite(output_path) is True
        get_file_size(output_path)

        # Test overwrite with overwrite=True (default)
        new_entries = entries
        write_jsonl_file(new_entries, output_path, overwrite=True)

        # File should be overwritten
        assert os.path.exists(output_path)
        get_file_size(output_path)

        # Verify content matches new entries
        read_entries = read_jsonl_file(output_path)
        assert len(read_entries) == len(new_entries)

        # Test overwrite with overwrite=False
        with pytest.raises(FileOperationError, match="already exists"):
            write_jsonl_file(entries, output_path, overwrite=False)


def test_edge_cases():
    """Test edge cases for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test empty entries list
        with pytest.raises(FileOperationError, match="No entries provided"):
            write_jsonl_file([], os.path.join(temp_dir, "empty.jsonl"))

        # Test invalid file path
        invalid_path = "/invalid/path/that/does/not/exist/file.jsonl"
        entries = [DataEntry(instruction="test", input="test", output="test")]

        with pytest.raises(FileOperationError):
            write_jsonl_file(entries, invalid_path)

        # Test reading non-existent file
        with pytest.raises(FileOperationError, match="File not found"):
            read_jsonl_file(os.path.join(temp_dir, "nonexistent.jsonl"))


def test_jsonl_extension_handling():
    """Test JSONL extension handling."""
    assert ensure_jsonl_extension("test") == "test.jsonl"
    assert ensure_jsonl_extension("test.jsonl") == "test.jsonl"
    assert ensure_jsonl_extension("test.txt") == "test.txt.jsonl"
    assert ensure_jsonl_extension("") == ".jsonl"


def test_append_functionality():
    """Test appending entries to existing JSONL file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "append_test.jsonl")

        # Create initial entries
        initial_entries = [
            DataEntry(instruction="first", input="input1", output="output1")
        ]
        write_jsonl_file(initial_entries, output_path)

        # Append more entries
        additional_entries = [
            DataEntry(instruction="second", input="input2", output="output2"),
            DataEntry(instruction="third", input="input3", output="output3")
        ]
        append_jsonl_entries(additional_entries, output_path)

        # Verify all entries are present
        all_entries = read_jsonl_file(output_path)
        assert len(all_entries) == 3

        assert all_entries[0]['instruction'] == "first"
        assert all_entries[1]['instruction'] == "second"
        assert all_entries[2]['instruction'] == "third"


def test_validation_with_invalid_jsonl():
    """Test validation function with invalid JSONL content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_path = os.path.join(temp_dir, "invalid.jsonl")

        # Create file with invalid JSON
        with open(invalid_path, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')

        # Should fail validation
        assert validate_jsonl_file(invalid_path) is False


# Edge case tests for Requirements 6.3 and 6.5

def test_disk_space_checking():
    """Test disk space insufficient scenarios - Requirements 6.3"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "test.jsonl")

        # Test with reasonable space requirement (should pass)
        assert check_disk_space(test_path, 1024) is True

        # Test with unreasonably large space requirement (should fail)
        with pytest.raises(DiskSpaceError, match="Insufficient disk space"):
            check_disk_space(test_path, 1024 * 1024 * 1024 * 1024)  # 1TB


def test_disk_space_checking_with_invalid_path():
    """Test disk space checking with invalid paths."""
    # Test with invalid path - should handle gracefully
    result = check_disk_space("/invalid/path/that/does/not/exist", 1024)
    # Should return True (graceful handling) or raise appropriate error
    assert isinstance(result, bool)


def test_file_size_estimation():
    """Test file size estimation for disk space checking."""
    # Test basic estimation
    size = estimate_file_size(10)
    assert size > 0
    assert isinstance(size, int)

    # Test with different parameters
    size_100 = estimate_file_size(100, 1000)
    size_10 = estimate_file_size(10, 1000)
    assert size_100 > size_10


def test_partial_backup_creation():
    """Test interruption handling with partial result saving - Requirements 6.5"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test.jsonl")

        # Create test entries
        entries = [
            DataEntry(instruction="test1", input="input1", output="output1"),
            DataEntry(instruction="test2", input="input2", output="output2")
        ]

        # Create partial backup
        backup_path = create_partial_backup(entries, output_path)

        # Verify backup was created
        assert os.path.exists(backup_path)
        assert "partial" in backup_path

        # Verify backup content
        backup_entries = read_jsonl_file(backup_path)
        assert len(backup_entries) == 2
        assert backup_entries[0]['instruction'] == "test1"
        assert backup_entries[1]['instruction'] == "test2"


def test_partial_backup_with_empty_entries():
    """Test partial backup with empty entries list."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test.jsonl")

        # Should return empty string for empty entries
        backup_path = create_partial_backup([], output_path)
        assert backup_path == ""


def test_interruption_handling_setup():
    """Test interruption handling setup."""
    entries = [DataEntry(instruction="test", input="test", output="test")]
    output_path = "test.jsonl"

    # Should not raise any exceptions
    setup_interruption_handling(entries, output_path)

    # Initially should not be interrupted
    assert is_interrupted() is False


@patch('ollaforge.file_manager.os.statvfs')
def test_disk_space_with_mocked_statvfs(mock_statvfs):
    """Test disk space checking with mocked system calls."""
    # Mock insufficient space
    mock_stat = MagicMock()
    mock_stat.f_bavail = 100  # 100 blocks available
    mock_stat.f_frsize = 1024  # 1KB per block = 100KB available
    mock_statvfs.return_value = mock_stat

    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "test.jsonl")

        # Should raise DiskSpaceError for large requirement
        with pytest.raises(DiskSpaceError):
            check_disk_space(test_path, 1024 * 1024)  # Require 1MB

        # Should pass for small requirement
        assert check_disk_space(test_path, 50 * 1024) is True  # Require 50KB


def test_write_jsonl_with_disk_space_check():
    """Test JSONL writing with disk space checking."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test.jsonl")
        entries = [DataEntry(instruction="test", input="test", output="test")]

        # Should work normally with sufficient space
        write_jsonl_file(entries, output_path)
        assert os.path.exists(output_path)


@patch('ollaforge.file_manager.check_disk_space')
def test_write_jsonl_with_insufficient_disk_space(mock_check_disk_space):
    """Test JSONL writing when disk space is insufficient."""
    # Mock disk space check to raise DiskSpaceError
    mock_check_disk_space.side_effect = DiskSpaceError("Insufficient disk space")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test.jsonl")
        entries = [DataEntry(instruction="test", input="test", output="test")]

        # Should raise DiskSpaceError
        with pytest.raises(DiskSpaceError):
            write_jsonl_file(entries, output_path)


def test_various_error_conditions():
    """Test various error conditions for comprehensive coverage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with read-only directory (permission error)
        readonly_dir = os.path.join(temp_dir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)  # Read-only

        readonly_file = os.path.join(readonly_dir, "test.jsonl")
        entries = [DataEntry(instruction="test", input="test", output="test")]

        try:
            # Should handle permission errors gracefully
            write_jsonl_file(entries, readonly_file)
        except FileOperationError:
            # This is expected and acceptable
            pass
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)


def test_interruption_signal_handling():
    """Test signal handling for interruption."""
    # This test is tricky because we can't easily test actual signal handling
    # in unit tests, but we can test the setup doesn't crash
    entries = [DataEntry(instruction="test", input="test", output="test")]

    # Should not raise exceptions
    setup_interruption_handling(entries, "test.jsonl")

    # Test the interrupted state
    assert is_interrupted() is False


def test_create_partial_backup_error_handling():
    """Test partial backup creation with error conditions."""
    # Test with invalid output path
    entries = [DataEntry(instruction="test", input="test", output="test")]
    invalid_path = "/invalid/path/that/cannot/be/created/test.jsonl"

    with pytest.raises(FileOperationError, match="Failed to create partial backup"):
        create_partial_backup(entries, invalid_path)


def test_disk_space_edge_cases():
    """Test disk space checking with various edge cases - Requirements 6.3"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "test.jsonl")

        # Test with zero space requirement
        assert check_disk_space(test_path, 0) is True

        # Test with negative space requirement (should handle gracefully)
        assert check_disk_space(test_path, -1000) is True

        # Test with extremely large space requirement
        with pytest.raises(DiskSpaceError):
            check_disk_space(test_path, 10**18)  # Exabyte requirement


def test_file_operations_with_unicode_content():
    """Test file operations with Unicode content - Requirements 6.4"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "unicode_test.jsonl")

        # Create entries with Unicode content
        unicode_entries = [
            DataEntry(instruction="测试指令", input="输入内容", output="输出结果"),
            DataEntry(instruction="🚀 Rocket test", input="emoji input 🎉", output="emoji output ✅"),
            DataEntry(instruction="Ñoño niño", input="café résumé", output="naïve façade"),
        ]

        # Should handle Unicode content without issues
        write_jsonl_file(unicode_entries, output_path)

        # Verify content can be read back correctly
        read_entries = read_jsonl_file(output_path)
        assert len(read_entries) == 3
        assert read_entries[0]['instruction'] == "测试指令"
        assert read_entries[1]['instruction'] == "🚀 Rocket test"
        assert read_entries[2]['instruction'] == "Ñoño niño"


def test_file_operations_with_very_large_entries():
    """Test file operations with very large entries - Requirements 6.3"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "large_test.jsonl")

        # Create entries with very large content
        large_content = "x" * 100000  # 100KB of content
        large_entries = [
            DataEntry(instruction=large_content, input="small input", output="small output"),
            DataEntry(instruction="small instruction", input=large_content, output="small output"),
            DataEntry(instruction="small instruction", input="small input", output=large_content),
        ]

        # Should handle large entries
        write_jsonl_file(large_entries, output_path)

        # Verify content
        read_entries = read_jsonl_file(output_path)
        assert len(read_entries) == 3
        assert len(read_entries[0]['instruction']) == 100000


def test_atomic_write_failure_recovery():
    """Test atomic write failure and recovery - Requirements 6.3"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "atomic_test.jsonl")
        entries = [DataEntry(instruction="test", input="test", output="test")]

        # Mock a failure during the atomic move operation
        with patch('shutil.move') as mock_move:
            mock_move.side_effect = OSError("Disk full")

            with pytest.raises(FileOperationError, match="Failed to write JSONL file"):
                write_jsonl_file(entries, output_path)

            # Verify original file doesn't exist (atomic operation failed)
            assert not os.path.exists(output_path)


def test_concurrent_file_access():
    """Test concurrent file access scenarios - Requirements 6.3"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "concurrent_test.jsonl")
        entries = [DataEntry(instruction="test", input="test", output="test")]

        # Create initial file
        write_jsonl_file(entries, output_path)

        # Test reading while another process might be writing
        # This is a basic test - real concurrent access is hard to test in unit tests
        read_entries = read_jsonl_file(output_path)
        assert len(read_entries) == 1

        # Test appending to existing file
        additional_entries = [DataEntry(instruction="test2", input="test2", output="test2")]
        append_jsonl_entries(additional_entries, output_path)

        # Verify both entries exist
        all_entries = read_jsonl_file(output_path)
        assert len(all_entries) == 2


def test_file_operations_with_network_drive_simulation():
    """Test file operations that might fail on network drives - Requirements 6.3"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "network_test.jsonl")
        entries = [DataEntry(instruction="test", input="test", output="test")]

        # Mock network-related failures
        with patch('tempfile.NamedTemporaryFile', side_effect=OSError("Network path not found")):
            with pytest.raises(FileOperationError):
                write_jsonl_file(entries, output_path)


def test_interruption_handling_with_large_dataset():
    """Test interruption handling with large datasets - Requirements 6.5"""
    # Create a large number of entries to simulate long-running generation
    large_entries = [
        DataEntry(instruction=f"test{i}", input=f"input{i}", output=f"output{i}")
        for i in range(1000)
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "large_dataset.jsonl")

        # Set up interruption handling
        setup_interruption_handling(large_entries, output_path)

        # Verify setup doesn't crash with large datasets
        assert is_interrupted() is False

        # Test partial backup creation with large dataset
        backup_path = create_partial_backup(large_entries[:100], output_path)
        assert os.path.exists(backup_path)

        # Verify backup content
        backup_entries = read_jsonl_file(backup_path)
        assert len(backup_entries) == 100


def test_file_validation_with_corrupted_content():
    """Test file validation with various types of corrupted content - Requirements 6.4"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with mixed valid/invalid content
        corrupted_path = os.path.join(temp_dir, "corrupted.jsonl")

        with open(corrupted_path, 'w', encoding='utf-8') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
            f.write('{"incomplete": "json"\n')  # Missing closing brace
            f.write('{"valid": "final"}\n')

        # Should detect corruption
        assert validate_jsonl_file(corrupted_path) is False

        # Test with completely empty file
        empty_path = os.path.join(temp_dir, "empty.jsonl")
        with open(empty_path, 'w') as f:
            pass

        # Empty file should be considered valid
        assert validate_jsonl_file(empty_path) is True

        # Test with file containing only whitespace
        whitespace_path = os.path.join(temp_dir, "whitespace.jsonl")
        with open(whitespace_path, 'w') as f:
            f.write('   \n\n  \t  \n')

        # Whitespace-only file should be considered valid
        assert validate_jsonl_file(whitespace_path) is True


def test_disk_space_monitoring_during_write():
    """Test disk space monitoring during file write operations - Requirements 6.3"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "space_test.jsonl")

        # Create entries that would require significant space
        large_entries = [
            DataEntry(instruction="x" * 10000, input="y" * 10000, output="z" * 10000)
            for _ in range(10)
        ]

        # Test with mocked insufficient space during write
        with patch('os.statvfs') as mock_statvfs:
            mock_stat = MagicMock()
            mock_stat.f_bavail = 1  # Only 1 block available
            mock_stat.f_frsize = 1024  # 1KB per block
            mock_statvfs.return_value = mock_stat

            with pytest.raises(DiskSpaceError):
                write_jsonl_file(large_entries, output_path)


def test_file_operations_with_readonly_filesystem():
    """Test file operations on read-only filesystem - Requirements 6.3"""
    with tempfile.TemporaryDirectory() as temp_dir:
        readonly_dir = os.path.join(temp_dir, "readonly")
        os.makedirs(readonly_dir)

        # Make directory read-only
        os.chmod(readonly_dir, 0o444)

        readonly_file = os.path.join(readonly_dir, "test.jsonl")
        entries = [DataEntry(instruction="test", input="test", output="test")]

        try:
            # Should fail with permission error
            with pytest.raises(FileOperationError):
                write_jsonl_file(entries, readonly_file)
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)


def test_estimate_file_size_edge_cases():
    """Test file size estimation with edge cases - Requirements 6.3"""
    # Test with zero entries
    assert estimate_file_size(0) >= 0

    # Test with very large entry count
    large_size = estimate_file_size(1000000)
    assert large_size > 0
    assert isinstance(large_size, int)

    # Test with custom average entry size
    custom_size = estimate_file_size(100, 2000)
    default_size = estimate_file_size(100)
    assert custom_size > default_size

    # Test with zero average entry size
    zero_avg_size = estimate_file_size(100, 0)
    assert zero_avg_size >= 1024  # Should still have buffer


def test_partial_backup_with_filesystem_errors():
    """Test partial backup creation with filesystem errors - Requirements 6.5"""
    entries = [DataEntry(instruction="test", input="test", output="test")]

    # Test with path that cannot be created
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        mock_mkdir.side_effect = OSError("Read-only file system")

        with pytest.raises(FileOperationError, match="Failed to create partial backup"):
            create_partial_backup(entries, "/readonly/path/test.jsonl")


def test_signal_handler_edge_cases():
    """Test signal handler with various edge cases - Requirements 6.5"""

    # Test signal handler setup with empty results
    setup_interruption_handling([], "test.jsonl")
    assert is_interrupted() is False

    # Test with very large results list
    large_entries = [
        DataEntry(instruction=f"test{i}", input=f"input{i}", output=f"output{i}")
        for i in range(1000)
    ]

    setup_interruption_handling(large_entries, "large_test.jsonl")
    assert is_interrupted() is False



# Tests for dataset augmentation file operations (Requirements 1.1, 1.3, 1.4)

def test_read_jsonl_with_field_names():
    """Test reading JSONL file with field names extraction - Requirements 1.1, 1.4"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "test.jsonl")

        # Create test file with various fields
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('{"instruction": "test1", "input": "in1", "output": "out1"}\n')
            f.write('{"instruction": "test2", "input": "in2", "output": "out2", "extra": "field"}\n')
            f.write('{"instruction": "test3", "input": "in3", "output": "out3", "another": "one"}\n')

        # Test with return_field_names=True
        entries, field_names = read_jsonl_file(test_path, return_field_names=True)

        assert len(entries) == 3
        assert isinstance(field_names, list)
        # Should have all unique fields sorted
        assert set(field_names) == {"instruction", "input", "output", "extra", "another"}
        # Should be sorted
        assert field_names == sorted(field_names)

        # Test with return_field_names=False (default behavior)
        entries_only = read_jsonl_file(test_path)
        assert len(entries_only) == 3
        assert isinstance(entries_only, list)
        assert not isinstance(entries_only, tuple)


def test_read_jsonl_field_names_empty_file():
    """Test reading empty JSONL file with field names - Requirements 1.1, 1.4"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "empty.jsonl")

        # Create empty file
        with open(test_path, 'w'):
            pass

        entries, field_names = read_jsonl_file(test_path, return_field_names=True)

        assert entries == []
        assert field_names == []


def test_read_jsonl_error_with_line_number():
    """Test that invalid JSONL reports line number - Requirements 1.3"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "invalid.jsonl")

        # Create file with invalid JSON on line 3
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('{"valid": "json"}\n')
            f.write('{"also": "valid"}\n')
            f.write('invalid json here\n')
            f.write('{"more": "valid"}\n')

        with pytest.raises(FileOperationError) as exc_info:
            read_jsonl_file(test_path)

        # Error message should contain line number
        assert "line 3" in str(exc_info.value).lower()


def test_read_jsonl_error_non_object_json():
    """Test that non-object JSON reports error with line number - Requirements 1.3"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "array.jsonl")

        # Create file with JSON array instead of object
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('{"valid": "object"}\n')
            f.write('[1, 2, 3]\n')  # Array, not object

        with pytest.raises(FileOperationError) as exc_info:
            read_jsonl_file(test_path)

        # Error message should contain line number and indicate type issue
        error_msg = str(exc_info.value).lower()
        assert "line 2" in error_msg
        assert "object" in error_msg or "list" in error_msg


def test_read_jsonl_field_names_with_unicode():
    """Test field names extraction with Unicode field names - Requirements 1.1, 1.4"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "unicode_fields.jsonl")

        # Create file with Unicode field names
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('{"指令": "test1", "輸入": "in1", "輸出": "out1"}\n')
            f.write('{"指令": "test2", "輸入": "in2", "輸出": "out2"}\n')

        entries, field_names = read_jsonl_file(test_path, return_field_names=True)

        assert len(entries) == 2
        assert set(field_names) == {"指令", "輸入", "輸出"}


def test_read_jsonl_field_names_varying_fields():
    """Test field names extraction when entries have different fields - Requirements 1.1, 1.4"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "varying.jsonl")

        # Create file where each entry has different fields
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('{"a": 1}\n')
            f.write('{"b": 2}\n')
            f.write('{"c": 3}\n')
            f.write('{"a": 4, "d": 5}\n')

        entries, field_names = read_jsonl_file(test_path, return_field_names=True)

        assert len(entries) == 4
        assert set(field_names) == {"a", "b", "c", "d"}


# Strategy for generating valid JSON objects as strings
valid_json_object_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    values=st.one_of(
        st.text(max_size=50),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.none()
    ),
    min_size=1,
    max_size=5
).map(lambda d: json.dumps(d))


# Strategy for generating invalid JSON strings (not parseable as JSON)
invalid_json_strategy = st.one_of(
    # Incomplete JSON objects (exclude newline characters to avoid line count issues)
    st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters='\r\n')).filter(
        lambda x: x.strip() and not _is_valid_json(x)
    ),
    # Strings that look like JSON but are malformed
    st.sampled_from([
        'invalid json here',
        '{missing: quotes}',
        '{"unclosed": "string',
        '{"missing": }',
        '{incomplete',
        'not json at all',
        '{"trailing": "comma",}',
        '[1, 2, 3]',  # Array instead of object - also invalid for our use case
    ])
)


def _is_valid_json(s: str) -> bool:
    """Helper to check if a string is valid JSON."""
    try:
        result = json.loads(s)
        return isinstance(result, dict)  # We require JSON objects, not arrays
    except (json.JSONDecodeError, ValueError):
        return False


@given(
    valid_lines_before=st.lists(valid_json_object_strategy, min_size=0, max_size=10),
    invalid_line=invalid_json_strategy,
    valid_lines_after=st.lists(valid_json_object_strategy, min_size=0, max_size=5)
)
def test_invalid_jsonl_error_reports_line_number(valid_lines_before, invalid_line, valid_lines_after):
    """
    **Feature: dataset-augmentation, Property 2: Invalid JSONL Error Reporting**
    **Validates: Requirements 1.3**

    For any JSONL file containing at least one malformed JSON line, the parser
    SHALL report an error that includes the line number of the first malformed line.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "test_invalid.jsonl")

        # Build the file content with valid lines before, invalid line, then valid lines after
        all_lines = valid_lines_before + [invalid_line] + valid_lines_after

        with open(test_path, 'w', encoding='utf-8') as f:
            for line in all_lines:
                f.write(line + '\n')

        # The expected line number of the first invalid line (1-indexed)
        expected_line_num = len(valid_lines_before) + 1

        # Attempt to read the file - should raise FileOperationError
        with pytest.raises(FileOperationError) as exc_info:
            read_jsonl_file(test_path)

        # The error message should contain the line number
        error_message = str(exc_info.value).lower()
        assert f"line {expected_line_num}" in error_message, (
            f"Error message should contain 'line {expected_line_num}', "
            f"but got: {exc_info.value}"
        )


# Strategy for generating valid dictionary entries for augmentation
augmentation_entry_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=30).filter(lambda x: x.strip() and '\x00' not in x),
    values=st.one_of(
        st.text(max_size=100),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.none(),
        st.lists(st.one_of(st.text(max_size=20), st.integers()), max_size=5),
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
            values=st.one_of(st.text(max_size=20), st.integers()),
            max_size=3
        )
    ),
    min_size=1,
    max_size=10
)


@given(entry=augmentation_entry_strategy)
def test_json_round_trip_consistency(entry):
    """
    **Feature: dataset-augmentation, Property 1: JSON Round-Trip Consistency**
    **Validates: Requirements 1.1, 3.4, 4.1**

    For any valid dictionary entry, serializing it to JSON and deserializing it back
    SHALL produce an equivalent dictionary.

    This property validates the core data integrity of the augmentation pipeline -
    entries must survive the serialization/deserialization cycle without data loss
    or corruption.
    """
    # Serialize the entry to JSON string
    json_string = json.dumps(entry)

    # Deserialize back to dictionary
    deserialized = json.loads(json_string)

    # The round-trip should produce an equivalent dictionary
    assert deserialized == entry, (
        f"Round-trip failed: original={entry}, deserialized={deserialized}"
    )


@given(entries=st.lists(augmentation_entry_strategy, min_size=1, max_size=10))
def test_jsonl_file_round_trip_consistency(entries):
    """
    **Feature: dataset-augmentation, Property 1: JSON Round-Trip Consistency (File-level)**
    **Validates: Requirements 1.1, 3.4, 4.1**

    For any list of valid dictionary entries, writing them to a JSONL file and reading
    them back SHALL produce an equivalent list of dictionaries.

    This tests the full file I/O round-trip for the augmentation pipeline.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "round_trip_test.jsonl")

        # Write entries to JSONL file manually (simulating augmentation output)
        with open(test_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                json_line = json.dumps(entry)
                f.write(json_line + '\n')

        # Read entries back using the file_manager function
        read_entries = read_jsonl_file(test_path)

        # Should have same number of entries
        assert len(read_entries) == len(entries), (
            f"Entry count mismatch: wrote {len(entries)}, read {len(read_entries)}"
        )

        # Each entry should be equivalent
        for i, (original, read_back) in enumerate(zip(entries, read_entries)):
            assert read_back == original, (
                f"Entry {i} mismatch: original={original}, read_back={read_back}"
            )
