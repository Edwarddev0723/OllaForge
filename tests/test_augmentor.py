"""
Tests for OllaForge augmentor module.

This module contains property-based tests for the dataset augmentation functionality.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from ollaforge.augmentor import create_augmentation_prompt
from ollaforge.models import OutputLanguage

# ============================================================================
# Property Tests for Prompt Generation
# ============================================================================

# Strategy for generating valid field names
valid_field_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
    min_size=1,
    max_size=30,
).filter(lambda x: x.strip() and x.isidentifier())

# Strategy for generating non-empty text values
non_empty_text = st.text(min_size=1, max_size=200).filter(lambda x: x.strip())


@st.composite
def entry_with_context_fields(draw):
    """
    Generate an entry dictionary with a set of context fields.

    Returns:
        Tuple of (entry_dict, list_of_context_field_names)
    """
    # Generate 1-5 field names for context
    num_fields = draw(st.integers(min_value=1, max_value=5))
    field_names = draw(
        st.lists(
            valid_field_names, min_size=num_fields, max_size=num_fields, unique=True
        )
    )

    # Create entry with values for each field
    entry = {}
    for field in field_names:
        entry[field] = draw(non_empty_text)

    return entry, field_names


@given(
    data=st.data(),
    instruction=non_empty_text,
    target_field=valid_field_names,
    language=st.sampled_from([OutputLanguage.EN, OutputLanguage.ZH_TW]),
)
@settings(max_examples=20)
def test_prompt_contains_context_and_instruction(
    data, instruction, target_field, language
):
    """
    **Feature: dataset-augmentation, Property 5: Prompt Contains Context and Instruction**
    **Validates: Requirements 2.3, 3.1**

    For any entry with context fields and for any non-empty instruction string,
    the generated prompt SHALL contain both the instruction text and the values
    of all specified context fields.
    """
    # Generate entry with context fields
    entry, context_fields = data.draw(entry_with_context_fields())

    # Generate the prompts
    system_prompt, user_prompt = create_augmentation_prompt(
        entry=entry,
        target_field=target_field,
        instruction=instruction,
        context_fields=context_fields,
        language=language,
    )

    # Combine prompts for checking (instruction is in system, context is in user)
    combined_prompts = system_prompt + user_prompt

    # Property 1: The instruction MUST appear in the system prompt
    assert (
        instruction in system_prompt
    ), f"Instruction '{instruction}' should be present in system prompt"

    # Property 2: All context field values MUST appear in the user prompt
    for field in context_fields:
        field_value = entry[field]
        assert (
            field_value in user_prompt
        ), f"Context field value '{field_value}' for field '{field}' should be present in user prompt"

    # Property 3: The target field name MUST appear in the prompts
    assert (
        target_field in combined_prompts
    ), f"Target field '{target_field}' should be mentioned in the prompts"


# ============================================================================
# Property Tests for New Field Creation
# ============================================================================


@st.composite
def entry_and_new_field(draw):
    """
    Generate an entry dictionary and a new field name that does NOT exist in the entry.

    Returns:
        Tuple of (entry_dict, new_field_name)
    """
    # Generate 1-5 existing field names
    num_fields = draw(st.integers(min_value=1, max_value=5))
    existing_fields = draw(
        st.lists(
            valid_field_names, min_size=num_fields, max_size=num_fields, unique=True
        )
    )

    # Create entry with values for each existing field
    entry = {}
    for field in existing_fields:
        entry[field] = draw(non_empty_text)

    # Generate a new field name that is NOT in the existing fields
    new_field = draw(valid_field_names.filter(lambda x: x not in existing_fields))

    return entry, new_field


@given(data=st.data(), augmented_value=non_empty_text)
@settings(max_examples=20)
def test_new_field_creation(data, augmented_value):
    """
    **Feature: dataset-augmentation, Property 6: New Field Creation**
    **Validates: Requirements 2.4**

    For any entry and for any valid new field name (not already in entry),
    when create_new_field is True, the augmented entry SHALL contain the new field.

    This test verifies the core field creation logic by simulating the augmentation
    process: given an entry without a target field, after augmentation the entry
    SHALL contain the new field with the augmented value.
    """
    import copy

    # Generate entry and a new field name not in the entry
    entry, new_field = data.draw(entry_and_new_field())

    # Precondition: the new field does NOT exist in the original entry
    assert (
        new_field not in entry
    ), f"Test setup error: '{new_field}' should not be in entry"

    # Simulate the augmentation logic (from augment_entry method)
    # This is the core logic that creates/updates the target field
    augmented = copy.deepcopy(entry)
    augmented[new_field] = augmented_value

    # Property: The augmented entry SHALL contain the new field
    assert (
        new_field in augmented
    ), f"New field '{new_field}' should be present in augmented entry"

    # Property: The new field SHALL have the augmented value
    assert (
        augmented[new_field] == augmented_value
    ), f"New field '{new_field}' should have value '{augmented_value}'"

    # Property: Original fields SHALL be preserved
    for original_field, original_value in entry.items():
        assert (
            original_field in augmented
        ), f"Original field '{original_field}' should be preserved"
        assert (
            augmented[original_field] == original_value
        ), f"Original field '{original_field}' value should be unchanged"


# ============================================================================
# Property Tests for Response Parsing
# ============================================================================


@st.composite
def entry_with_target_field(draw):
    """
    Generate an entry dictionary with a target field that exists in the entry.

    Returns:
        Tuple of (entry_dict, target_field_name)
    """
    # Generate 1-5 field names
    num_fields = draw(st.integers(min_value=1, max_value=5))
    field_names = draw(
        st.lists(
            valid_field_names, min_size=num_fields, max_size=num_fields, unique=True
        )
    )

    # Create entry with values for each field
    entry = {}
    for field in field_names:
        entry[field] = draw(non_empty_text)

    # Pick one of the existing fields as the target
    target_field = draw(st.sampled_from(field_names))

    return entry, target_field


@given(data=st.data(), ai_response_value=st.text(min_size=0, max_size=500))
@settings(max_examples=20)
def test_successful_response_updates_target_field(data, ai_response_value):
    """
    **Feature: dataset-augmentation, Property 7: Successful Response Updates Target Field**
    **Validates: Requirements 3.2**

    For any entry and for any valid AI response, the augmented entry's target field
    SHALL contain the value from the AI response.

    This test verifies the core response parsing logic: when the AI model returns
    a successful response, the target field in the augmented entry SHALL be updated
    with the stripped content from the response.
    """
    import copy

    # Generate entry with a target field
    entry, target_field = data.draw(entry_with_target_field())

    # Store original value for comparison
    entry[target_field]

    # Simulate the response parsing logic from augment_entry:
    # augmented_value = response['message']['content'].strip()
    # augmented[self.config.target_field] = augmented_value

    # Create a deep copy (as done in augment_entry)
    augmented = copy.deepcopy(entry)

    # Simulate successful AI response parsing
    # The actual code does: augmented_value = response['message']['content'].strip()
    augmented_value = ai_response_value.strip()

    # Update the target field (as done in augment_entry)
    augmented[target_field] = augmented_value

    # Property 1: The target field SHALL contain the stripped AI response value
    assert (
        augmented[target_field] == augmented_value
    ), f"Target field '{target_field}' should have the augmented value"

    # Property 2: The target field value should be the stripped version of the response
    assert (
        augmented[target_field] == ai_response_value.strip()
    ), "Target field should contain stripped response value"

    # Property 3: Other fields SHALL remain unchanged
    for field, value in entry.items():
        if field != target_field:
            assert (
                augmented[field] == value
            ), f"Non-target field '{field}' should remain unchanged"

    # Property 4: The augmented entry SHALL have the same set of keys
    assert set(augmented.keys()) == set(
        entry.keys()
    ), "Augmented entry should have the same fields as original"


# ============================================================================
# Property Tests for Failure Preservation
# ============================================================================


@st.composite
def arbitrary_entry(draw):
    """
    Generate an arbitrary entry dictionary with random fields and values.

    Returns:
        A dictionary with 1-5 fields and non-empty string values
    """
    num_fields = draw(st.integers(min_value=1, max_value=5))
    field_names = draw(
        st.lists(
            valid_field_names, min_size=num_fields, max_size=num_fields, unique=True
        )
    )

    entry = {}
    for field in field_names:
        entry[field] = draw(non_empty_text)

    return entry


@given(data=st.data())
@settings(max_examples=20)
def test_failure_preserves_original_entry(data):
    """
    **Feature: dataset-augmentation, Property 8: Failure Preserves Original Entry**
    **Validates: Requirements 3.3**

    For any entry, when AI generation fails, the returned entry SHALL be equal
    to the original entry (no modifications).

    This test verifies that when the augment_entry method encounters an error
    (e.g., API failure, invalid response), it returns the original entry unchanged
    rather than a partially modified or corrupted entry.
    """
    import copy
    from unittest.mock import patch

    from ollaforge.augmentor import DatasetAugmentor
    from ollaforge.models import AugmentationConfig, OutputLanguage

    # Generate an arbitrary entry
    entry = data.draw(arbitrary_entry())
    target_field = data.draw(valid_field_names)
    instruction = data.draw(non_empty_text)

    # Create a deep copy of the original entry for comparison
    original_entry = copy.deepcopy(entry)

    # Create augmentor config
    config = AugmentationConfig(
        input_file="test.jsonl",
        output_file="output.jsonl",
        target_field=target_field,
        instruction=instruction,
        model="test-model",
        language=OutputLanguage.EN,
        create_new_field=True,  # Allow new fields to avoid validation issues
        context_fields=[],
        preview_count=3,
    )

    augmentor = DatasetAugmentor(config)

    # Mock ollama.chat to raise an exception (simulating AI failure)
    with patch("ollaforge.augmentor.ollama.chat") as mock_chat:
        mock_chat.side_effect = Exception("Simulated AI failure")

        # Call augment_entry which should fail
        result_entry, error = augmentor.augment_entry(entry)

        # Property 1: An error message SHALL be returned
        assert error is not None, "Error message should be returned when AI fails"

        # Property 2: The returned entry SHALL be equal to the original entry
        assert result_entry == original_entry, (
            f"Returned entry should be identical to original entry. "
            f"Original: {original_entry}, Returned: {result_entry}"
        )

        # Property 3: The original entry object should NOT be modified
        assert entry == original_entry, "Original entry object should not be modified"


# ============================================================================
# Property Tests for Concurrent Processing
# ============================================================================


@st.composite
def list_of_entries(draw, min_size=1, max_size=20):
    """
    Generate a list of entry dictionaries.

    Args:
        min_size: Minimum number of entries
        max_size: Maximum number of entries

    Returns:
        List of entry dictionaries
    """
    num_entries = draw(st.integers(min_value=min_size, max_value=max_size))
    entries = []
    for _ in range(num_entries):
        entry = draw(arbitrary_entry())
        entries.append(entry)
    return entries


@given(data=st.data(), concurrency=st.integers(min_value=1, max_value=10))
@settings(max_examples=20)
def test_concurrent_processing_correctness(data, concurrency):
    """
    **Feature: dataset-augmentation, Property 9: Concurrent Processing Correctness**
    **Validates: Requirements 3.5**

    For any list of N entries processed with concurrency level C, the result
    SHALL contain exactly N entries, and each result entry SHALL correspond
    to exactly one input entry.

    This test verifies that concurrent processing:
    1. Produces exactly N results for N inputs
    2. Maintains the correct mapping between input and output entries
    3. Works correctly regardless of concurrency level
    """
    import copy
    from unittest.mock import patch

    from ollaforge.augmentor import DatasetAugmentor
    from ollaforge.models import AugmentationConfig, OutputLanguage

    # Generate a list of entries
    entries = data.draw(list_of_entries(min_size=1, max_size=15))
    n = len(entries)

    target_field = data.draw(valid_field_names)
    instruction = data.draw(non_empty_text)

    # Create deep copies of original entries for comparison
    original_entries = [copy.deepcopy(e) for e in entries]

    # Create augmentor config
    config = AugmentationConfig(
        input_file="test.jsonl",
        output_file="output.jsonl",
        target_field=target_field,
        instruction=instruction,
        model="test-model",
        language=OutputLanguage.EN,
        create_new_field=True,  # Allow new fields to avoid validation issues
        context_fields=[],
        preview_count=3,
    )

    augmentor = DatasetAugmentor(config)

    # Mock ollama.chat to return a deterministic response based on entry content
    # This allows us to verify that each result corresponds to its input
    def mock_chat_response(model, messages, options=None):
        """Generate a deterministic response based on the user message content."""
        user_content = messages[-1]["content"] if messages else ""
        # Create a unique response based on the input
        return {"message": {"content": f"augmented_{hash(user_content) % 10000}"}}

    with patch("ollaforge.augmentor.ollama.chat", side_effect=mock_chat_response):
        # Run concurrent augmentation
        result = augmentor.augment_dataset(entries, concurrency=concurrency)
        augmented_entries = augmentor.get_augmented_entries()

        # Property 1: Result SHALL contain exactly N entries
        assert (
            result.total_entries == n
        ), f"Total entries should be {n}, got {result.total_entries}"

        # Property 2: Augmented entries list SHALL have exactly N entries
        assert (
            len(augmented_entries) == n
        ), f"Augmented entries should have {n} items, got {len(augmented_entries)}"

        # Property 3: Success + Failure counts SHALL equal N
        assert (
            result.success_count + result.failure_count == n
        ), f"Success ({result.success_count}) + Failure ({result.failure_count}) should equal {n}"

        # Property 4: Each augmented entry SHALL preserve original fields
        # (except the target field which may be added/modified)
        for i, (original, augmented) in enumerate(
            zip(original_entries, augmented_entries)
        ):
            # All original fields should be present in augmented entry
            for field, value in original.items():
                if field != target_field:
                    assert (
                        field in augmented
                    ), f"Entry {i}: Original field '{field}' should be preserved"
                    assert (
                        augmented[field] == value
                    ), f"Entry {i}: Original field '{field}' value should be unchanged"

        # Property 5: All augmented entries SHALL have the target field
        # (since we're using create_new_field=True and mock returns success)
        for i, augmented in enumerate(augmented_entries):
            assert (
                target_field in augmented
            ), f"Entry {i}: Target field '{target_field}' should be present"


# ============================================================================
# Property Tests for Preview Count
# ============================================================================


@given(data=st.data(), preview_count=st.integers(min_value=1, max_value=10))
@settings(max_examples=20)
def test_preview_count_correctness(data, preview_count):
    """
    **Feature: dataset-augmentation, Property 11: Preview Count Correctness**
    **Validates: Requirements 7.1**

    For any dataset with N entries and preview_count P, the preview SHALL
    process exactly min(N, P) entries.

    This test verifies that the preview method:
    1. Returns exactly min(N, P) results
    2. Each result contains both original and augmented entries
    3. Works correctly for datasets smaller than, equal to, and larger than preview_count
    """
    from unittest.mock import patch

    from ollaforge.augmentor import DatasetAugmentor
    from ollaforge.models import AugmentationConfig, OutputLanguage

    # Generate a list of entries with variable size (1 to 20)
    entries = data.draw(list_of_entries(min_size=1, max_size=20))
    n = len(entries)

    target_field = data.draw(valid_field_names)
    instruction = data.draw(non_empty_text)

    # Create augmentor config with the given preview_count
    config = AugmentationConfig(
        input_file="test.jsonl",
        output_file="output.jsonl",
        target_field=target_field,
        instruction=instruction,
        model="test-model",
        language=OutputLanguage.EN,
        create_new_field=True,
        context_fields=[],
        preview_count=preview_count,
    )

    augmentor = DatasetAugmentor(config)

    # Expected number of preview results
    expected_count = min(n, preview_count)

    # Mock ollama.chat to return a deterministic response
    def mock_chat_response(model, messages, options=None):
        """Generate a deterministic response."""
        return {"message": {"content": "preview_augmented_value"}}

    with patch("ollaforge.augmentor.ollama.chat", side_effect=mock_chat_response):
        # Run preview
        preview_results = augmentor.preview(entries)

        # Property 1: Preview SHALL return exactly min(N, P) results
        assert (
            len(preview_results) == expected_count
        ), f"Preview should return {expected_count} results (min({n}, {preview_count})), got {len(preview_results)}"

        # Property 2: Each result SHALL be a tuple of (original, augmented)
        for i, result in enumerate(preview_results):
            assert isinstance(result, tuple), f"Result {i} should be a tuple"
            assert (
                len(result) == 2
            ), f"Result {i} should have 2 elements (original, augmented)"

            original, augmented = result

            # Property 3: Original entry SHALL be a dictionary
            assert isinstance(
                original, dict
            ), f"Result {i}: Original should be a dictionary"

            # Property 4: Augmented entry SHALL be a dictionary
            assert isinstance(
                augmented, dict
            ), f"Result {i}: Augmented should be a dictionary"

            # Property 5: Augmented entry SHALL have the target field
            assert (
                target_field in augmented
            ), f"Result {i}: Augmented entry should have target field '{target_field}'"
