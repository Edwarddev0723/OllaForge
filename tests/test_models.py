"""
Tests for OllaForge data models.
"""

import pytest
from hypothesis import given, strategies as st
from ollaforge.models import GenerationConfig, DataEntry, GenerationResult


@given(count=st.integers(min_value=1, max_value=10000))
def test_count_parameter_controls_output_quantity(count):
    """
    **Feature: ollama-cli-generator, Property 2: Count parameter controls output quantity**
    **Validates: Requirements 1.2**
    
    For any valid count parameter, the GenerationConfig should accept and store 
    that exact count value.
    """
    config = GenerationConfig(
        topic="test topic",
        count=count,
        model="llama3",
        output="test.jsonl"
    )
    
    assert config.count == count


@given(topic=st.text(min_size=1).filter(lambda x: x.strip()))
def test_topic_parameter_acceptance(topic):
    """
    **Feature: ollama-cli-generator, Property 1: Topic parameter acceptance**
    **Validates: Requirements 1.1**
    
    For any valid topic string provided as CLI parameter, the system should 
    accept and pass it to the generation process.
    """
    config = GenerationConfig(
        topic=topic,
        count=10,
        model="llama3",
        output="test.jsonl"
    )
    
    assert config.topic == topic.strip()