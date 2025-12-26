"""
Property-based tests for the document dataset generator module.

This module contains property-based tests using Hypothesis to verify
the correctness of the DocumentDatasetGenerator implementation.

Feature: document-to-dataset
"""

import pytest
import json
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ollaforge.doc_generator import (
    DocGenerationConfig,
    DocumentDatasetGenerator,
    validate_entry,
    validate_sft_entry,
    validate_pretrain_entry,
    validate_conversation_entry,
    validate_dpo_entry,
    entry_to_dict,
    dict_to_entry,
)
from ollaforge.models import (
    DatasetType,
    OutputLanguage,
    DataEntry,
    PretrainEntry,
    SFTConversationEntry,
    DPOEntry,
    ConversationMessage,
)


# ============================================================================
# Strategies for generating test data
# ============================================================================

@st.composite
def non_empty_strings(draw, min_size=1, max_size=200):
    """Generate non-empty strings that are not just whitespace."""
    s = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'P')),
        min_size=min_size,
        max_size=max_size
    ))
    assume(s.strip())
    return s


@st.composite
def sft_entries(draw):
    """Generate valid SFT (DataEntry) instances."""
    return DataEntry(
        instruction=draw(non_empty_strings()),
        input=draw(non_empty_strings()),
        output=draw(non_empty_strings())
    )


@st.composite
def pretrain_entries(draw):
    """Generate valid PretrainEntry instances."""
    return PretrainEntry(
        text=draw(non_empty_strings(min_size=10, max_size=500))
    )


@st.composite
def conversation_messages(draw, role=None):
    """Generate valid ConversationMessage instances."""
    if role is None:
        role = draw(st.sampled_from(['system', 'user', 'assistant']))
    content = draw(non_empty_strings())
    return ConversationMessage(role=role, content=content)


@st.composite
def valid_conversation_entries(draw):
    """Generate valid SFTConversationEntry instances with required roles."""
    # Must have at least one user and one assistant message
    user_msg = draw(conversation_messages(role='user'))
    assistant_msg = draw(conversation_messages(role='assistant'))
    
    # Optionally add system message
    has_system = draw(st.booleans())
    
    # Optionally add more messages
    extra_count = draw(st.integers(min_value=0, max_value=3))
    extra_msgs = [draw(conversation_messages()) for _ in range(extra_count)]
    
    # Build conversation list
    conversations = []
    if has_system:
        conversations.append(draw(conversation_messages(role='system')))
    conversations.append(user_msg)
    conversations.append(assistant_msg)
    conversations.extend(extra_msgs)
    
    return SFTConversationEntry(conversations=conversations)


@st.composite
def dpo_entries(draw):
    """Generate valid DPOEntry instances with different chosen/rejected."""
    prompt = draw(non_empty_strings())
    chosen = draw(non_empty_strings())
    rejected = draw(non_empty_strings())
    
    # Ensure chosen and rejected are different
    assume(chosen.strip() != rejected.strip())
    
    return DPOEntry(prompt=prompt, chosen=chosen, rejected=rejected)


# ============================================================================
# Property Tests
# ============================================================================

class TestDatasetEntrySchemaValidation:
    """
    Property 6: Dataset Entry Schema Validation
    
    *For any* generated dataset entry, the entry SHALL conform to its format schema:
    - SFT: non-empty instruction, input, output fields
    - Pre-training: non-empty text field
    - Conversation: conversations array with role/content objects
    - DPO: non-empty prompt, chosen, rejected fields
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 7.1, 7.2**
    """
    
    @given(entry=sft_entries())
    @settings(max_examples=20)
    def test_valid_sft_entries_pass_validation(self, entry):
        """
        Feature: document-to-dataset, Property 6: Dataset Entry Schema Validation (SFT)
        
        For any valid SFT entry with non-empty fields, validation should pass.
        
        **Validates: Requirements 3.1, 7.1, 7.2**
        """
        assert validate_entry(entry, DatasetType.SFT) is True
        assert validate_sft_entry(entry) is True
    
    @given(entry=pretrain_entries())
    @settings(max_examples=20)
    def test_valid_pretrain_entries_pass_validation(self, entry):
        """
        Feature: document-to-dataset, Property 6: Dataset Entry Schema Validation (Pretrain)
        
        For any valid pre-training entry with non-empty text, validation should pass.
        
        **Validates: Requirements 3.2, 7.1**
        """
        assert validate_entry(entry, DatasetType.PRETRAIN) is True
        assert validate_pretrain_entry(entry) is True
    
    @given(entry=valid_conversation_entries())
    @settings(max_examples=20)
    def test_valid_conversation_entries_pass_validation(self, entry):
        """
        Feature: document-to-dataset, Property 6: Dataset Entry Schema Validation (Conversation)
        
        For any valid conversation entry with user and assistant messages,
        validation should pass.
        
        **Validates: Requirements 3.3, 7.1**
        """
        assert validate_entry(entry, DatasetType.SFT_CONVERSATION) is True
        assert validate_conversation_entry(entry) is True
    
    @given(entry=dpo_entries())
    @settings(max_examples=20)
    def test_valid_dpo_entries_pass_validation(self, entry):
        """
        Feature: document-to-dataset, Property 6: Dataset Entry Schema Validation (DPO)
        
        For any valid DPO entry with non-empty fields and different responses,
        validation should pass.
        
        **Validates: Requirements 3.4, 7.1**
        """
        assert validate_entry(entry, DatasetType.DPO) is True
        assert validate_dpo_entry(entry) is True
    
    @given(
        instruction=st.text(max_size=100),
        input_text=st.text(max_size=100),
        output_text=st.text(max_size=100)
    )
    @settings(max_examples=20)
    def test_sft_empty_fields_fail_validation(self, instruction, input_text, output_text):
        """
        Feature: document-to-dataset, Property 6: Dataset Entry Schema Validation (SFT Empty)
        
        SFT entries with any empty field should fail validation.
        
        **Validates: Requirements 7.2**
        """
        # If any field is empty or whitespace-only, validation should fail
        entry = DataEntry(
            instruction=instruction,
            input=input_text,
            output=output_text
        )
        
        has_empty = (
            not instruction.strip() or
            not input_text.strip() or
            not output_text.strip()
        )
        
        if has_empty:
            assert validate_sft_entry(entry) is False
        else:
            assert validate_sft_entry(entry) is True


class TestConversationRoleRequirements:
    """
    Property 7: Conversation Role Requirements
    
    *For any* generated Conversation entry, the conversations array SHALL contain
    at least one message with role "user" and at least one message with role "assistant".
    
    **Validates: Requirements 7.3**
    """
    
    @given(entry=valid_conversation_entries())
    @settings(max_examples=20)
    def test_valid_conversations_have_required_roles(self, entry):
        """
        Feature: document-to-dataset, Property 7: Conversation Role Requirements
        
        Valid conversation entries must have both user and assistant roles.
        
        **Validates: Requirements 7.3**
        """
        assert validate_conversation_entry(entry) is True
        
        # Verify the roles are present
        roles = [msg.role for msg in entry.conversations]
        assert 'user' in roles
        assert 'assistant' in roles
    
    @given(
        system_msgs=st.lists(conversation_messages(role='system'), min_size=0, max_size=2),
        user_msgs=st.lists(conversation_messages(role='user'), min_size=0, max_size=3),
        assistant_msgs=st.lists(conversation_messages(role='assistant'), min_size=0, max_size=3)
    )
    @settings(max_examples=20)
    def test_missing_roles_fail_validation(self, system_msgs, user_msgs, assistant_msgs):
        """
        Feature: document-to-dataset, Property 7: Conversation Role Requirements (Missing)
        
        Conversation entries missing user or assistant roles should fail validation.
        
        **Validates: Requirements 7.3**
        """
        conversations = system_msgs + user_msgs + assistant_msgs
        
        if not conversations:
            # Empty conversations should fail
            entry = SFTConversationEntry(conversations=[])
            assert validate_conversation_entry(entry) is False
            return
        
        entry = SFTConversationEntry(conversations=conversations)
        
        has_user = len(user_msgs) > 0
        has_assistant = len(assistant_msgs) > 0
        
        if has_user and has_assistant:
            assert validate_conversation_entry(entry) is True
        else:
            assert validate_conversation_entry(entry) is False


class TestDPOResponseDifferentiation:
    """
    Property 8: DPO Response Differentiation
    
    *For any* generated DPO entry, the chosen response SHALL be different
    from the rejected response (chosen != rejected).
    
    **Validates: Requirements 7.4**
    """
    
    @given(entry=dpo_entries())
    @settings(max_examples=20)
    def test_valid_dpo_has_different_responses(self, entry):
        """
        Feature: document-to-dataset, Property 8: DPO Response Differentiation
        
        Valid DPO entries must have different chosen and rejected responses.
        
        **Validates: Requirements 7.4**
        """
        assert validate_dpo_entry(entry) is True
        assert entry.chosen.strip() != entry.rejected.strip()
    
    @given(
        prompt=non_empty_strings(),
        response=non_empty_strings()
    )
    @settings(max_examples=20)
    def test_identical_responses_fail_validation(self, prompt, response):
        """
        Feature: document-to-dataset, Property 8: DPO Response Differentiation (Identical)
        
        DPO entries with identical chosen and rejected responses should fail validation.
        
        **Validates: Requirements 7.4**
        """
        entry = DPOEntry(
            prompt=prompt,
            chosen=response,
            rejected=response  # Same as chosen
        )
        
        assert validate_dpo_entry(entry) is False
    
    @given(
        prompt=non_empty_strings(),
        chosen=non_empty_strings(),
        rejected=non_empty_strings()
    )
    @settings(max_examples=20)
    def test_dpo_validation_depends_on_difference(self, prompt, chosen, rejected):
        """
        Feature: document-to-dataset, Property 8: DPO Response Differentiation (General)
        
        DPO validation should pass iff chosen and rejected are different.
        
        **Validates: Requirements 7.4**
        """
        entry = DPOEntry(prompt=prompt, chosen=chosen, rejected=rejected)
        
        are_different = chosen.strip() != rejected.strip()
        
        if are_different:
            assert validate_dpo_entry(entry) is True
        else:
            assert validate_dpo_entry(entry) is False


class TestSerializationRoundTrip:
    """
    Property 9: Serialization Round-Trip
    
    *For any* valid dataset entry, serializing to JSON and deserializing back
    SHALL produce an equivalent entry.
    
    **Validates: Requirements 3.6**
    """
    
    @given(entry=sft_entries())
    @settings(max_examples=20)
    def test_sft_round_trip(self, entry):
        """
        Feature: document-to-dataset, Property 9: Serialization Round-Trip (SFT)
        
        SFT entries should survive serialization round-trip.
        
        **Validates: Requirements 3.6**
        """
        # Serialize to dict
        data = entry_to_dict(entry)
        
        # Serialize to JSON and back
        json_str = json.dumps(data)
        parsed_data = json.loads(json_str)
        
        # Deserialize to entry
        restored = dict_to_entry(parsed_data, DatasetType.SFT)
        
        assert restored is not None
        assert isinstance(restored, DataEntry)
        assert restored.instruction == entry.instruction
        assert restored.input == entry.input
        assert restored.output == entry.output
    
    @given(entry=pretrain_entries())
    @settings(max_examples=20)
    def test_pretrain_round_trip(self, entry):
        """
        Feature: document-to-dataset, Property 9: Serialization Round-Trip (Pretrain)
        
        Pre-training entries should survive serialization round-trip.
        
        **Validates: Requirements 3.6**
        """
        # Serialize to dict
        data = entry_to_dict(entry)
        
        # Serialize to JSON and back
        json_str = json.dumps(data)
        parsed_data = json.loads(json_str)
        
        # Deserialize to entry
        restored = dict_to_entry(parsed_data, DatasetType.PRETRAIN)
        
        assert restored is not None
        assert isinstance(restored, PretrainEntry)
        assert restored.text == entry.text
    
    @given(entry=valid_conversation_entries())
    @settings(max_examples=20)
    def test_conversation_round_trip(self, entry):
        """
        Feature: document-to-dataset, Property 9: Serialization Round-Trip (Conversation)
        
        Conversation entries should survive serialization round-trip.
        
        **Validates: Requirements 3.6**
        """
        # Serialize to dict
        data = entry_to_dict(entry)
        
        # Serialize to JSON and back
        json_str = json.dumps(data)
        parsed_data = json.loads(json_str)
        
        # Deserialize to entry
        restored = dict_to_entry(parsed_data, DatasetType.SFT_CONVERSATION)
        
        assert restored is not None
        assert isinstance(restored, SFTConversationEntry)
        assert len(restored.conversations) == len(entry.conversations)
        
        for orig, rest in zip(entry.conversations, restored.conversations):
            assert rest.role == orig.role
            assert rest.content == orig.content
    
    @given(entry=dpo_entries())
    @settings(max_examples=20)
    def test_dpo_round_trip(self, entry):
        """
        Feature: document-to-dataset, Property 9: Serialization Round-Trip (DPO)
        
        DPO entries should survive serialization round-trip.
        
        **Validates: Requirements 3.6**
        """
        # Serialize to dict
        data = entry_to_dict(entry)
        
        # Serialize to JSON and back
        json_str = json.dumps(data)
        parsed_data = json.loads(json_str)
        
        # Deserialize to entry
        restored = dict_to_entry(parsed_data, DatasetType.DPO)
        
        assert restored is not None
        assert isinstance(restored, DPOEntry)
        assert restored.prompt == entry.prompt
        assert restored.chosen == entry.chosen
        assert restored.rejected == entry.rejected
    
    @given(
        dataset_type=st.sampled_from([
            DatasetType.SFT,
            DatasetType.PRETRAIN,
            DatasetType.SFT_CONVERSATION,
            DatasetType.DPO
        ])
    )
    @settings(max_examples=20)
    def test_round_trip_preserves_validity(self, dataset_type):
        """
        Feature: document-to-dataset, Property 9: Serialization Round-Trip (Validity)
        
        Round-trip serialization should preserve entry validity.
        
        **Validates: Requirements 3.6**
        """
        # Generate appropriate entry based on type
        if dataset_type == DatasetType.SFT:
            entry = DataEntry(
                instruction="Test instruction",
                input="Test input",
                output="Test output"
            )
        elif dataset_type == DatasetType.PRETRAIN:
            entry = PretrainEntry(text="Test pre-training text content")
        elif dataset_type == DatasetType.SFT_CONVERSATION:
            entry = SFTConversationEntry(conversations=[
                ConversationMessage(role='user', content='Hello'),
                ConversationMessage(role='assistant', content='Hi there!')
            ])
        else:  # DPO
            entry = DPOEntry(
                prompt="Test prompt",
                chosen="Good response",
                rejected="Bad response"
            )
        
        # Verify original is valid
        assert validate_entry(entry, dataset_type) is True
        
        # Round-trip
        data = entry_to_dict(entry)
        json_str = json.dumps(data)
        parsed_data = json.loads(json_str)
        restored = dict_to_entry(parsed_data, dataset_type)
        
        # Verify restored is also valid
        assert restored is not None
        assert validate_entry(restored, dataset_type) is True
