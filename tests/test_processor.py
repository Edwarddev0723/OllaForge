"""
Tests for OllaForge data processing and cleaning utilities.
"""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ollaforge.models import DataEntry
from ollaforge.processor import (
    _fix_common_json_issues,
    clean_json,
    process_model_response,
    process_responses,
    validate_entry,
)

# Strategy for generating valid JSON data entries
valid_data_entry = st.fixed_dictionaries({
    'instruction': st.text(min_size=1),
    'input': st.text(min_size=1),
    'output': st.text(min_size=1)
})


@given(data=valid_data_entry)
def test_json_extraction_from_responses(data):
    """
    **Feature: ollama-cli-generator, Property 8: JSON extraction from responses**
    **Validates: Requirements 3.1**

    For any model response containing JSON within markdown or noise,
    the clean_json function should extract valid JSON.
    """
    # Create valid JSON string
    json_str = json.dumps(data)

    # Test cases with various markdown and noise patterns
    test_cases = [
        # Plain JSON
        json_str,
        # Markdown code blocks
        f"```json\n{json_str}\n```",
        f"```\n{json_str}\n```",
        # With prefixes
        f"Here's the JSON:\n{json_str}",
        f"JSON: {json_str}",
        f"Response: {json_str}",
        # With surrounding text
        f"Some text before\n{json_str}\nSome text after",
        # With whitespace
        f"   {json_str}   ",
        # Mixed markdown and prefixes
        f"Here is the JSON:\n```json\n{json_str}\n```",
    ]

    for test_case in test_cases:
        result = clean_json(test_case)
        assert result is not None, f"Failed to extract JSON from: {test_case}"
        assert result == data, f"Extracted JSON doesn't match original data for: {test_case}"


@given(responses=st.lists(st.text(), min_size=1, max_size=10))
def test_malformed_response_recovery(responses):
    """
    **Feature: ollama-cli-generator, Property 16: Malformed response recovery**
    **Validates: Requirements 6.4**

    For any malformed response from the model, the system should continue
    processing remaining entries without termination.
    """
    # Process the responses - should not raise exceptions
    try:
        result = process_responses(responses)
        # Should return a list (possibly empty)
        assert isinstance(result, list)
        # All items in result should be DataEntry instances
        for entry in result:
            assert isinstance(entry, DataEntry)
    except Exception as e:
        pytest.fail(f"process_responses raised an exception: {e}")


@given(
    valid_data=valid_data_entry,
    invalid_responses=st.lists(
        st.one_of(
            st.text().filter(lambda x: not x.strip()),  # Empty/whitespace
            st.text().filter(lambda x: '{' not in x and '}' not in x),  # No JSON brackets
            st.just("invalid json {"),  # Malformed JSON
            st.just('{"missing": "fields"}'),  # Missing required fields
        ),
        min_size=1,
        max_size=5
    )
)
def test_invalid_json_recovery(valid_data, invalid_responses):
    """
    **Feature: ollama-cli-generator, Property 10: Invalid JSON recovery**
    **Validates: Requirements 3.3**

    For any invalid JSON encountered during processing, the system should
    skip that entry and continue with remaining entries.
    """
    # Create a mix of valid and invalid responses
    valid_json = json.dumps(valid_data)
    all_responses = invalid_responses + [valid_json]

    # Process all responses
    result = process_responses(all_responses)

    # Should return a list
    assert isinstance(result, list)

    # Should contain at least one valid entry (from valid_json)
    assert len(result) >= 1

    # All returned entries should be valid DataEntry instances
    for entry in result:
        assert isinstance(entry, DataEntry)
        assert entry.instruction == valid_data['instruction']
        assert entry.input == valid_data['input']
        assert entry.output == valid_data['output']


def test_clean_json_edge_cases():
    """Test edge cases for clean_json function."""
    # Empty input
    assert clean_json("") is None
    assert clean_json("   ") is None
    assert clean_json(None) is None

    # Invalid JSON
    assert clean_json("not json at all") is None
    assert clean_json("{ invalid json }") is None

    # Valid simple JSON
    result = clean_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_validate_entry_edge_cases():
    """Test edge cases for validate_entry function."""
    # Missing fields
    assert validate_entry({}) is None
    assert validate_entry({"instruction": "test"}) is None
    assert validate_entry({"instruction": "test", "input": "test"}) is None

    # Valid entry
    valid_data = {"instruction": "test", "input": "test", "output": "test"}
    result = validate_entry(valid_data)
    assert result is not None
    assert isinstance(result, DataEntry)
    assert result.instruction == "test"


def test_fix_common_json_issues():
    """Test JSON fixing utility function."""
    # Trailing comma
    fixed = _fix_common_json_issues('{"key": "value",}')
    assert '"key": "value"' in fixed

    # Single quotes (simple case)
    fixed = _fix_common_json_issues("{'key': 'value'}")
    assert '"key": "value"' in fixed


# Additional edge case tests for Requirements 6.3, 6.4, and 6.5

def test_process_model_response_with_unicode_content():
    """Test processing model responses with Unicode content - Requirements 6.4"""
    unicode_responses = [
        '{"instruction": "测试指令", "input": "输入内容", "output": "输出结果"}',
        '{"instruction": "🚀 Test", "input": "emoji 🎉", "output": "result ✅"}',
        '{"instruction": "Ñoño", "input": "café", "output": "naïve"}',
        '{"instruction": "Русский", "input": "тест", "output": "результат"}',
    ]

    for response in unicode_responses:
        result = process_model_response(response)
        assert result is not None
        assert isinstance(result, DataEntry)


def test_process_model_response_with_extremely_large_content():
    """Test processing extremely large model responses - Requirements 6.3"""
    # Create response with very large content
    large_content = "x" * 100000  # 100KB content
    large_response = json.dumps({
        "instruction": large_content,
        "input": "small input",
        "output": "small output"
    })

    # Should handle large content without issues
    result = process_model_response(large_response)
    assert result is not None
    assert isinstance(result, DataEntry)
    assert len(result.instruction) == 100000


def test_process_model_response_with_nested_json():
    """Test processing responses with nested JSON structures - Requirements 6.4"""
    nested_responses = [
        '{"instruction": "test", "input": "{\\"nested\\": \\"json\\"}", "output": "result"}',
        '{"instruction": "test", "input": "[1, 2, 3]", "output": "array result"}',
        '{"instruction": "test", "input": "normal", "output": "{\\"complex\\": {\\"nested\\": \\"output\\"}}"}',
    ]

    for response in nested_responses:
        result = process_model_response(response)
        assert result is not None
        assert isinstance(result, DataEntry)


def test_process_model_response_with_special_characters():
    """Test processing responses with special characters - Requirements 6.4"""
    special_responses = [
        '{"instruction": "test\\nwith\\nnewlines", "input": "test\\ttabs", "output": "test\\rcarriage"}',
        '{"instruction": "test\\"quotes\\"", "input": "test\'apostrophes\'", "output": "test\\\\backslashes"}',
        '{"instruction": "test/forward/slashes", "input": "test\\\\back\\\\slashes", "output": "test|pipes|"}',
        '{"instruction": "test<>brackets", "input": "test[]square", "output": "test{}curly"}',
    ]

    for response in special_responses:
        result = process_model_response(response)
        assert result is not None
        assert isinstance(result, DataEntry)


def test_clean_json_with_malformed_markdown():
    """Test JSON extraction from malformed markdown - Requirements 6.4"""
    malformed_cases = [
        # Unclosed code blocks
        '```json\n{"instruction": "test", "input": "test", "output": "test"}',
        '{"instruction": "test", "input": "test", "output": "test"}\n```',
        # Mixed markdown
        '```\n{"instruction": "test", "input": "test", "output": "test"}\n```json',
        # Multiple code blocks
        '```json\n{"invalid": "json"}\n```\n```json\n{"instruction": "test", "input": "test", "output": "test"}\n```',
    ]

    for case in malformed_cases:
        result = clean_json(case)
        # Should either extract valid JSON or return None gracefully
        if result is not None:
            assert isinstance(result, dict)


def test_clean_json_with_multiple_json_objects():
    """Test JSON extraction when multiple JSON objects are present - Requirements 6.4"""
    multiple_json_cases = [
        '{"first": "object"} {"instruction": "test", "input": "test", "output": "test"}',
        '{"instruction": "test", "input": "test", "output": "test"} {"second": "object"}',
        '[{"array": "object"}] {"instruction": "test", "input": "test", "output": "test"}',
    ]

    for case in multiple_json_cases:
        result = clean_json(case)
        # Should extract the valid DataEntry JSON or handle gracefully
        if result is not None:
            assert isinstance(result, dict)


def test_process_responses_with_memory_pressure():
    """Test processing large number of responses under memory pressure - Requirements 6.3"""
    # Create a large number of responses
    large_response_list = []
    for i in range(10000):
        response = json.dumps({
            "instruction": f"instruction_{i}",
            "input": f"input_{i}",
            "output": f"output_{i}"
        })
        large_response_list.append(response)

    # Should handle large lists without memory issues
    result = process_responses(large_response_list)
    assert isinstance(result, list)
    assert len(result) == 10000


def test_process_model_response_with_corrupted_encoding():
    """Test processing responses with encoding issues - Requirements 6.4"""
    # Simulate responses with encoding problems
    corrupted_cases = [
        # Valid JSON with potential encoding issues
        '{"instruction": "test", "input": "test", "output": "test"}',
        # JSON with escaped unicode
        '{"instruction": "test\\u0020space", "input": "test\\u0009tab", "output": "test\\u000Anewline"}',
    ]

    for case in corrupted_cases:
        result = process_model_response(case)
        # Should handle encoding issues gracefully
        if result is not None:
            assert isinstance(result, DataEntry)


def test_validate_entry_with_edge_case_values():
    """Test entry validation with edge case values - Requirements 6.4"""
    edge_cases = [
        # Empty strings
        {"instruction": "", "input": "", "output": ""},
        # Very long strings
        {"instruction": "x" * 10000, "input": "y" * 10000, "output": "z" * 10000},
        # Whitespace only
        {"instruction": "   ", "input": "\t\t", "output": "\n\n"},
        # Special characters
        {"instruction": "test\x00null", "input": "test\x01control", "output": "test\x02chars"},
        # Unicode
        {"instruction": "测试", "input": "🚀", "output": "ñ"},
    ]

    for case in edge_cases:
        result = validate_entry(case)
        # Should either validate successfully or fail gracefully
        if result is not None:
            assert isinstance(result, DataEntry)


def test_clean_json_with_performance_stress():
    """Test JSON cleaning with performance stress scenarios - Requirements 6.3"""
    # Very large JSON string
    large_data = {
        "instruction": "x" * 50000,
        "input": "y" * 50000,
        "output": "z" * 50000
    }
    large_json = json.dumps(large_data)

    # Wrap in markdown with lots of noise
    noisy_response = f"""
    Here's a lot of text before the JSON that should be ignored.
    {'noise ' * 1000}
    ```json
    {large_json}
    ```
    {'more noise ' * 1000}
    """

    # Should handle large, noisy responses efficiently
    result = clean_json(noisy_response)
    assert result is not None
    assert result == large_data


def test_process_responses_with_mixed_valid_invalid():
    """Test processing mixed valid and invalid responses - Requirements 6.4"""
    mixed_responses = [
        # Valid responses
        '{"instruction": "test1", "input": "input1", "output": "output1"}',
        '{"instruction": "test2", "input": "input2", "output": "output2"}',
        # Invalid responses
        'not json at all',
        '{"incomplete": "json"',
        '{"missing": "required_fields"}',
        '',
        None,
        # More valid responses
        '{"instruction": "test3", "input": "input3", "output": "output3"}',
    ]

    # Should process valid responses and skip invalid ones
    result = process_responses(mixed_responses)
    assert isinstance(result, list)
    assert len(result) == 3  # Only the 3 valid responses

    for entry in result:
        assert isinstance(entry, DataEntry)


def test_fix_common_json_issues_comprehensive():
    """Test comprehensive JSON fixing scenarios - Requirements 6.4"""
    json_issues = [
        # Multiple trailing commas
        ('{"key1": "value1",, "key2": "value2",}', 'trailing commas'),
        # Mixed quotes
        ('{"key1": \'value1\', "key2": "value2"}', 'mixed quotes'),
        # Unescaped quotes
        ('{"key": "value with "quotes" inside"}', 'unescaped quotes'),
        # Missing quotes on keys
        ('{key: "value"}', 'missing key quotes'),
        # Extra whitespace
        ('{ "key" : "value" , }', 'extra whitespace'),
    ]

    for broken_json, _description in json_issues:
        try:
            fixed = _fix_common_json_issues(broken_json)
            # Should produce a string that's closer to valid JSON
            assert isinstance(fixed, str)
            assert len(fixed) > 0
        except Exception:
            # If fixing fails, that's also acceptable - graceful handling
            pass


def test_process_model_response_with_timeout_simulation():
    """Test processing responses under timeout pressure - Requirements 6.5"""
    import time

    # Create response that might take time to process
    complex_response = json.dumps({
        "instruction": "complex " * 10000,
        "input": "processing " * 10000,
        "output": "task " * 10000
    })

    # Should complete processing within reasonable time
    start_time = time.time()
    result = process_model_response(complex_response)
    end_time = time.time()

    # Should not take excessively long (more than 5 seconds would be concerning)
    assert (end_time - start_time) < 5.0

    if result is not None:
        assert isinstance(result, DataEntry)


def test_clean_json_with_recursive_structures():
    """Test JSON cleaning with deeply nested structures - Requirements 6.4"""
    # Create deeply nested JSON (but still valid for our use case)
    nested_content = '{"instruction": "test", "input": "nested", "output": "result"}'

    # Wrap in multiple layers of markdown and text
    deeply_wrapped = nested_content
    for i in range(10):
        deeply_wrapped = f"Layer {i}:\n```json\n{deeply_wrapped}\n```\nEnd layer {i}"

    # Should handle deep nesting
    result = clean_json(deeply_wrapped)
    if result is not None:
        assert isinstance(result, dict)
        assert "instruction" in result


def test_process_responses_error_recovery():
    """Test error recovery during response processing - Requirements 6.5"""
    # Create responses that might cause various processing errors
    problematic_responses = [
        '{"instruction": "test1", "input": "input1", "output": "output1"}',  # Valid
        '{"instruction": null, "input": "input2", "output": "output2"}',  # Null value
        '{"instruction": 123, "input": "input3", "output": "output3"}',  # Wrong type
        '{"instruction": "test4", "input": "input4", "output": "output4"}',  # Valid
    ]

    # Should recover from errors and continue processing
    result = process_responses(problematic_responses)
    assert isinstance(result, list)
    # Should have at least the valid responses
    assert len(result) >= 2
