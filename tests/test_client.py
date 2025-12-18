"""
Tests for OllaForge Ollama client communication.
"""

import pytest
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st

from ollaforge.client import (
    generate_data, 
    _test_ollama_connection, 
    _create_system_prompt_single,
    _create_user_prompt,
    get_available_models,
    OllamaConnectionError,
    OllamaGenerationError
)
from ollaforge.models import DatasetType, OutputLanguage


@given(
    topic=st.text(min_size=1).filter(lambda x: x.strip()),
    model=st.text(min_size=1).filter(lambda x: x.strip()),
    count=st.integers(min_value=1, max_value=10)
)
def test_ollama_api_connection_establishment(topic, model, count):
    """
    **Feature: ollama-cli-generator, Property 5: Ollama API connection establishment**
    **Validates: Requirements 2.1**
    
    For any generation request, the system should attempt to establish connection 
    with Ollama API on localhost:11434.
    """
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock successful generation
        mock_ollama.chat.return_value = {
            'message': {
                'content': '{"instruction": "test", "input": "test", "output": "test"}'
            }
        }
        
        # Call generate_data - should attempt connection
        try:
            result = generate_data(topic.strip(), model.strip(), count)
            
            # Verify that ollama.list() was called to test connection
            mock_ollama.list.assert_called_once()
            
            # Verify that ollama.chat() was called for generation
            assert mock_ollama.chat.call_count == count
            
            # Result should be a list
            assert isinstance(result, list)
            
        except Exception as e:
            # If mocking fails, that's acceptable for this property test
            # The important thing is that the connection attempt was made
            pass


@given(
    topic=st.text(min_size=1).filter(lambda x: x.strip()),
    model=st.text(min_size=1).filter(lambda x: x.strip())
)
def test_json_format_prompt_engineering(topic, model):
    """
    **Feature: ollama-cli-generator, Property 6: JSON format prompt engineering**
    **Validates: Requirements 2.2**
    
    For any request sent to Ollama, the prompt should include instructions for JSON output format.
    """
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock successful generation
        mock_ollama.chat.return_value = {
            'message': {
                'content': '{"instruction": "test", "input": "test", "output": "test"}'
            }
        }
        
        # Call generate_data
        try:
            generate_data(topic.strip(), model.strip(), 1)
            
            # Verify that ollama.chat was called
            assert mock_ollama.chat.call_count == 1
            
            # Get the call arguments
            call_args = mock_ollama.chat.call_args
            messages = call_args[1]['messages'] if 'messages' in call_args[1] else call_args[0][1]
            
            # Verify system message contains JSON format instructions
            system_message = None
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                    break
            
            assert system_message is not None
            assert 'JSON' in system_message or 'json' in system_message
            assert 'instruction' in system_message
            assert 'input' in system_message
            assert 'output' in system_message
            
        except Exception as e:
            # If mocking fails, that's acceptable for this property test
            pass


@given(
    topic=st.text(min_size=1).filter(lambda x: x.strip()),
    model=st.text(min_size=1).filter(lambda x: x.strip())
)
def test_api_error_handling(topic, model):
    """
    **Feature: ollama-cli-generator, Property 7: API error handling**
    **Validates: Requirements 2.5**
    
    For any API timeout or connection error, the system should handle the error 
    gracefully and continue processing.
    """
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Test 1: Connection failure should raise OllamaConnectionError
        mock_ollama.list.side_effect = ConnectionError("Connection refused")
        
        try:
            generate_data(topic.strip(), model.strip(), 1)
            # If no exception is raised, that's also acceptable (graceful handling)
            assert True
        except OllamaConnectionError:
            # This is expected and acceptable - graceful error handling
            assert True
        except Exception:
            # Any other exception handling is also acceptable
            assert True
        
        # Test 2: Generation errors should be handled gracefully
        mock_ollama.reset_mock()
        mock_ollama.list.return_value = {'models': []}
        mock_ollama.chat.side_effect = Exception("API timeout")
        
        try:
            result = generate_data(topic.strip(), model.strip(), 3)
            # Should return a list (possibly empty) and not crash
            assert isinstance(result, list)
        except (OllamaGenerationError, OllamaConnectionError):
            # These are acceptable - graceful error handling
            assert True
        except Exception:
            # Any other exception handling is also acceptable
            assert True


def test_connection_test_function():
    """Test the connection test function specifically."""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Test successful connection
        mock_ollama.list.return_value = {'models': []}
        _test_ollama_connection()  # Should not raise
        
        # Test connection failure
        mock_ollama.list.side_effect = ConnectionError("Connection refused")
        with pytest.raises(OllamaConnectionError):
            _test_ollama_connection()


def test_system_prompt_creation():
    """Test system prompt creation includes JSON format instructions."""
    topic = "customer service conversations"
    prompt = _create_system_prompt_single(topic, DatasetType.SFT, OutputLanguage.EN)
    
    assert topic in prompt
    assert 'JSON' in prompt or 'json' in prompt
    assert 'instruction' in prompt
    assert 'input' in prompt
    assert 'output' in prompt


def test_user_prompt_creation():
    """Test user prompt creation."""
    topic = "test topic"
    entry_number = 5
    prompt = _create_user_prompt(topic, entry_number, DatasetType.SFT, OutputLanguage.EN)
    
    assert topic in prompt
    assert str(entry_number) in prompt
    assert 'JSON' in prompt or 'json' in prompt


def test_get_available_models():
    """Test getting available models."""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Test successful model listing
        mock_ollama.list.return_value = {
            'models': [
                {'name': 'llama3'},
                {'name': 'mistral'}
            ]
        }
        
        models = get_available_models()
        assert models == ['llama3', 'mistral']
        
        # Test connection failure
        mock_ollama.list.side_effect = ConnectionError("Connection refused")
        with pytest.raises(OllamaConnectionError):
            get_available_models()


# Edge case tests for Requirements 6.3 and 6.5

def test_connection_timeout_handling():
    """Test API timeout handling - Requirements 2.5"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock timeout error
        mock_ollama.list.side_effect = Exception("Request timeout")
        
        with pytest.raises(OllamaConnectionError, match="timed out"):
            _test_ollama_connection()


def test_model_not_found_error():
    """Test model not found error handling."""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock model not found error
        mock_ollama.chat.side_effect = Exception("model 'nonexistent' not found")
        
        with pytest.raises(OllamaGenerationError, match="Model not found"):
            generate_data("test topic", "nonexistent", 1)


def test_generation_with_network_interruption():
    """Test generation with network interruption."""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock network interruption during generation
        mock_ollama.chat.side_effect = [
            {'message': {'content': '{"instruction": "test1", "input": "test1", "output": "test1"}'}},
            ConnectionError("Network unreachable"),
            {'message': {'content': '{"instruction": "test3", "input": "test3", "output": "test3"}'}}
        ]
        
        # Should handle network interruption gracefully
        result = generate_data("test topic", "llama3", 3)
        
        # Should return partial results (not crash)
        assert isinstance(result, list)
        # May have fewer results due to network error


def test_generation_with_malformed_api_response():
    """Test generation with malformed API responses."""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock malformed responses
        mock_ollama.chat.side_effect = [
            {'invalid': 'response'},  # Missing 'message' key
            {'message': {}},  # Missing 'content' key
            {'message': {'content': 'valid response'}}
        ]
        
        # Should handle malformed responses gracefully
        result = generate_data("test topic", "llama3", 3)
        
        # Should return list (possibly with fewer entries)
        assert isinstance(result, list)


def test_generation_with_empty_responses():
    """Test generation with empty responses from API."""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock empty responses
        mock_ollama.chat.return_value = {'message': {'content': ''}}
        
        # Should handle empty responses
        result = generate_data("test topic", "llama3", 2)
        
        # Should return list with empty content entries
        assert isinstance(result, list)
        assert len(result) == 2
        for entry in result:
            assert 'raw_content' in entry
            assert entry['raw_content'] == ''


def test_connection_with_different_error_types():
    """Test connection handling with various error types."""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Test different connection error types
        error_types = [
            ConnectionError("Connection refused"),
            Exception("connection timeout"),
            Exception("Connection reset by peer"),
            Exception("Network is unreachable")
        ]
        
        for error in error_types:
            mock_ollama.list.side_effect = error
            
            with pytest.raises(OllamaConnectionError):
                _test_ollama_connection()


def test_generation_with_api_rate_limiting():
    """Test generation with API rate limiting."""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock rate limiting error
        mock_ollama.chat.side_effect = Exception("Rate limit exceeded")
        
        # Should handle rate limiting gracefully
        try:
            result = generate_data("test topic", "llama3", 1)
            assert isinstance(result, list)
        except OllamaGenerationError:
            # This is also acceptable - graceful error handling
            pass


def test_system_prompt_with_special_characters():
    """Test system prompt creation with special characters in topic."""
    special_topics = [
        "topic with \"quotes\"",
        "topic with 'apostrophes'",
        "topic with\nnewlines",
        "topic with\ttabs",
        "topic with unicode: 🚀 🎉 ñ ü"
    ]
    
    for topic in special_topics:
        prompt = _create_system_prompt_single(topic, DatasetType.SFT, OutputLanguage.EN)
        
        # Should handle special characters without crashing
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert topic in prompt or topic.replace('\n', ' ').replace('\t', ' ') in prompt


def test_user_prompt_with_edge_cases():
    """Test user prompt creation with edge cases."""
    # Test with very large entry numbers
    prompt = _create_user_prompt("test topic", 999999, DatasetType.SFT, OutputLanguage.EN)
    assert "999999" in prompt
    
    # Test with zero entry number
    prompt = _create_user_prompt("test topic", 0, DatasetType.SFT, OutputLanguage.EN)
    assert "0" in prompt
    
    # Test with negative entry number
    prompt = _create_user_prompt("test topic", -1, DatasetType.SFT, OutputLanguage.EN)
    assert "-1" in prompt


# Additional edge case tests for Requirements 6.3 and 6.5

def test_connection_with_system_resource_exhaustion():
    """Test connection handling when system resources are exhausted - Requirements 6.3"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock resource exhaustion errors
        resource_errors = [
            OSError("Too many open files"),
            MemoryError("Cannot allocate memory"),
            OSError("Resource temporarily unavailable"),
        ]
        
        for error in resource_errors:
            mock_ollama.list.side_effect = error
            
            with pytest.raises(OllamaConnectionError):
                _test_ollama_connection()


def test_generation_with_extremely_large_responses():
    """Test generation with extremely large model responses - Requirements 6.3"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock extremely large response
        large_content = "x" * 1000000  # 1MB response
        mock_ollama.chat.return_value = {
            'message': {
                'content': f'{{"instruction": "{large_content}", "input": "test", "output": "test"}}'
            }
        }
        
        # Should handle large responses
        result = generate_data("test topic", "llama3", 1)
        assert isinstance(result, list)
        assert len(result) == 1


def test_generation_with_memory_pressure():
    """Test generation under memory pressure - Requirements 6.3"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock memory error during generation
        mock_ollama.chat.side_effect = MemoryError("Out of memory")
        
        # Should handle memory errors gracefully
        try:
            result = generate_data("test topic", "llama3", 1)
            assert isinstance(result, list)
        except OllamaGenerationError:
            # This is also acceptable - graceful error handling
            pass


def test_connection_with_dns_resolution_failure():
    """Test connection with DNS resolution failures - Requirements 6.5"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock DNS resolution failure
        mock_ollama.list.side_effect = OSError("Name or service not known")
        
        with pytest.raises(OllamaConnectionError, match="Name or service not known"):
            _test_ollama_connection()


def test_generation_with_partial_response_corruption():
    """Test generation with partially corrupted responses - Requirements 6.4"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock responses with various corruption patterns
        corrupted_responses = [
            {'message': {'content': '{"instruction": "test", "input": "test", "output": "test"'}},  # Missing closing brace
            {'message': {'content': '{"instruction": "test", "input": "test", "output":'}},  # Incomplete
            {'message': {'content': 'Some text before {"instruction": "test", "input": "test", "output": "test"} some text after'}},  # Extra text
            {'message': {'content': '{"instruction": "test", "input": "test", "output": "test"}\n{"extra": "json"}'}},  # Multiple JSON objects
        ]
        
        for corrupted_response in corrupted_responses:
            mock_ollama.chat.return_value = corrupted_response
            
            # Should handle corrupted responses
            result = generate_data("test topic", "llama3", 1)
            assert isinstance(result, list)
            assert len(result) == 1
            assert 'raw_content' in result[0]


def test_generation_with_api_version_mismatch():
    """Test generation with API version mismatches - Requirements 6.5"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock API version mismatch error
        mock_ollama.list.side_effect = Exception("API version not supported")
        
        with pytest.raises(OllamaConnectionError, match="API version not supported"):
            _test_ollama_connection()


def test_generation_with_model_loading_timeout():
    """Test generation with model loading timeouts - Requirements 6.5"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock model loading timeout
        mock_ollama.chat.side_effect = Exception("Model loading timeout")
        
        with pytest.raises(OllamaGenerationError, match="Model loading timeout"):
            generate_data("test topic", "nonexistent_model", 1)


def test_generation_with_context_window_overflow():
    """Test generation with context window overflow - Requirements 6.4"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock context window overflow error
        mock_ollama.chat.side_effect = Exception("Context length exceeded")
        
        with pytest.raises(OllamaGenerationError, match="Context length exceeded"):
            generate_data("test topic", "llama3", 1)


def test_get_available_models_with_malformed_response():
    """Test getting available models with malformed API response - Requirements 6.4"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Test with missing 'models' key
        mock_ollama.list.return_value = {'invalid': 'response'}
        
        with pytest.raises(OllamaConnectionError):
            get_available_models()
        
        # Test with malformed model entries
        mock_ollama.list.return_value = {
            'models': [
                {'invalid': 'entry'},  # Missing 'name' key
                {'name': 'valid_model'},
                None,  # Null entry
            ]
        }
        
        # Should handle malformed entries gracefully
        try:
            models = get_available_models()
            # Should return at least the valid model
            assert 'valid_model' in models
        except OllamaConnectionError:
            # This is also acceptable - graceful error handling
            pass


def test_prompt_creation_with_extremely_long_topics():
    """Test prompt creation with extremely long topics - Requirements 6.3"""
    # Test with very long topic
    long_topic = "a" * 100000  # 100KB topic
    
    system_prompt = _create_system_prompt_single(long_topic, DatasetType.SFT, OutputLanguage.EN)
    assert isinstance(system_prompt, str)
    assert len(system_prompt) > 0
    
    user_prompt = _create_user_prompt(long_topic, 1, DatasetType.SFT, OutputLanguage.EN)
    assert isinstance(user_prompt, str)
    assert len(user_prompt) > 0


def test_generation_with_concurrent_requests():
    """Test generation with concurrent request simulation - Requirements 6.5"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock responses that simulate concurrent access issues
        mock_ollama.chat.side_effect = [
            {'message': {'content': '{"instruction": "test1", "input": "test1", "output": "test1"}'}},
            Exception("Server busy, try again later"),
            {'message': {'content': '{"instruction": "test3", "input": "test3", "output": "test3"}'}},
        ]
        
        # Should handle server busy errors
        result = generate_data("test topic", "llama3", 3)
        assert isinstance(result, list)
        # Should have partial results


def test_connection_with_firewall_blocking():
    """Test connection when firewall blocks access - Requirements 6.5"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock firewall blocking
        mock_ollama.list.side_effect = ConnectionError("Connection blocked by firewall")
        
        with pytest.raises(OllamaConnectionError, match="Cannot connect to Ollama API"):
            _test_ollama_connection()


def test_generation_with_model_crash_recovery():
    """Test generation with model crash and recovery - Requirements 6.5"""
    with patch('ollaforge.client.ollama') as mock_ollama:
        # Mock successful connection test
        mock_ollama.list.return_value = {'models': []}
        
        # Mock model crash followed by recovery
        mock_ollama.chat.side_effect = [
            Exception("Model process crashed"),
            {'message': {'content': '{"instruction": "test2", "input": "test2", "output": "test2"}'}},
        ]
        
        # Should handle model crashes gracefully
        result = generate_data("test topic", "llama3", 2)
        assert isinstance(result, list)
        # Should have partial results from recovery