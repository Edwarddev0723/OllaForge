"""
Tests for OllaForge data models.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ollaforge.models import (
    AugmentationResult,
    FieldValidationError,
    GenerationConfig,
    validate_target_field,
)


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


# ============================================================================
# Property Tests for AugmentationConfig Field Validation
# ============================================================================

# Strategy for generating valid field names (non-empty strings without whitespace-only)
valid_field_names = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
    min_size=1,
    max_size=50
).filter(lambda x: x.strip() and x.isidentifier())

# Strategy for generating dataset entries with known fields
@st.composite
def dataset_with_fields(draw):
    """Generate a dataset with a known set of fields."""
    # Generate 1-5 field names
    num_fields = draw(st.integers(min_value=1, max_value=5))
    field_names = draw(st.lists(
        valid_field_names,
        min_size=num_fields,
        max_size=num_fields,
        unique=True
    ))

    # Generate 1-10 entries with these fields
    num_entries = draw(st.integers(min_value=1, max_value=10))
    entries = []
    for _ in range(num_entries):
        entry = {}
        for field in field_names:
            entry[field] = draw(st.text(min_size=0, max_size=100))
        entries.append(entry)

    return entries, set(field_names)


@given(data=st.data())
@settings(max_examples=20)
def test_field_validation_existing_field_accepted(data):
    """
    **Feature: dataset-augmentation, Property 3: Field Validation - Existing Field Accepted**
    **Validates: Requirements 2.1, 2.2**

    For any dataset containing entries with a set of fields F, and for any
    field name f ∈ F, the field validation SHALL accept f as a valid target field.
    """
    # Generate dataset with known fields
    entries, field_names = data.draw(dataset_with_fields())

    # Pick one of the existing fields
    target_field = data.draw(st.sampled_from(sorted(field_names)))

    # Validation should succeed (return True) for existing field
    result = validate_target_field(entries, target_field, create_new_field=False)
    assert result is True, f"Field '{target_field}' should be accepted as it exists in {field_names}"


@given(data=st.data())
@settings(max_examples=20)
def test_field_validation_non_existing_field_rejected(data):
    """
    **Feature: dataset-augmentation, Property 4: Field Validation - Non-Existing Field Rejected**
    **Validates: Requirements 2.1, 2.2**

    For any dataset containing entries with a set of fields F, and for any
    field name f ∉ F (where create_new_field is False), the field validation
    SHALL reject f and the error message SHALL contain at least one field from F.
    """
    # Generate dataset with known fields
    entries, field_names = data.draw(dataset_with_fields())

    # Generate a field name that is NOT in the dataset
    non_existing_field = data.draw(
        valid_field_names.filter(lambda x: x not in field_names)
    )

    # Validation should raise FieldValidationError for non-existing field
    with pytest.raises(FieldValidationError) as exc_info:
        validate_target_field(entries, non_existing_field, create_new_field=False)

    # Error should contain at least one available field from the dataset
    error = exc_info.value
    assert len(error.available_fields) > 0, "Error should list available fields"
    assert any(f in field_names for f in error.available_fields), \
        f"Error available_fields {error.available_fields} should contain at least one field from {field_names}"


# ============================================================================
# Property Tests for AugmentationResult Statistics Accuracy
# ============================================================================

@given(
    success_count=st.integers(min_value=0, max_value=10000),
    failure_count=st.integers(min_value=0, max_value=10000),
    duration=st.floats(min_value=0.0, max_value=86400.0, allow_nan=False, allow_infinity=False),
    output_file=st.text(min_size=1, max_size=100).filter(lambda x: x.strip())
)
@settings(max_examples=20)
def test_statistics_accuracy(success_count, failure_count, duration, output_file):
    """
    **Feature: dataset-augmentation, Property 10: Statistics Accuracy**
    **Validates: Requirements 1.4, 5.2, 5.3**

    For any augmentation run with S successes and F failures out of T total entries,
    the result statistics SHALL report success_count = S, failure_count = F,
    and total_entries = T, where S + F = T.
    """
    # Calculate total entries as sum of successes and failures
    total_entries = success_count + failure_count

    # Create AugmentationResult with the generated values
    result = AugmentationResult(
        total_entries=total_entries,
        success_count=success_count,
        failure_count=failure_count,
        output_file=output_file,
        duration=duration,
        errors=[]
    )

    # Verify statistics are accurately reported
    assert result.success_count == success_count, \
        f"success_count should be {success_count}, got {result.success_count}"
    assert result.failure_count == failure_count, \
        f"failure_count should be {failure_count}, got {result.failure_count}"
    assert result.total_entries == total_entries, \
        f"total_entries should be {total_entries}, got {result.total_entries}"

    # Verify the invariant: success + failure = total
    assert result.success_count + result.failure_count == result.total_entries, \
        f"success_count ({result.success_count}) + failure_count ({result.failure_count}) " \
        f"should equal total_entries ({result.total_entries})"

    # Verify success_rate calculation is accurate
    if total_entries > 0:
        expected_rate = (success_count / total_entries) * 100
        assert abs(result.success_rate - expected_rate) < 0.0001, \
            f"success_rate should be {expected_rate}, got {result.success_rate}"
    else:
        assert result.success_rate == 0.0, \
            f"success_rate should be 0.0 for empty dataset, got {result.success_rate}"
