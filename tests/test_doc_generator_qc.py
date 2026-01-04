"""
QC integration tests for the document dataset generator module.

This module contains tests to verify the QC (Quality Control) integration
in the DocumentDatasetGenerator for Taiwan Chinese validation.

Feature: document-to-dataset
Requirements satisfied:
- 7.5: QC flag for quality control filtering
- 7.6: Taiwan Chinese validation when QC enabled for zh-tw
"""

from unittest.mock import MagicMock, patch

from ollaforge.doc_generator import (
    DocGenerationConfig,
    DocumentDatasetGenerator,
)
from ollaforge.models import (
    ConversationMessage,
    DataEntry,
    DatasetType,
    DPOEntry,
    OutputLanguage,
    PretrainEntry,
    SFTConversationEntry,
)
from ollaforge.qc import QualityController


class TestQCIntegration:
    """
    Tests for QC integration in DocumentDatasetGenerator.

    **Validates: Requirements 7.5, 7.6**
    """

    def test_qc_controller_initialized_for_zhtw_with_qc_enabled(self):
        """
        QC controller should be initialized when qc_enabled=True and language=zh-tw.

        **Validates: Requirements 7.5, 7.6**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT,
            language=OutputLanguage.ZH_TW,
            qc_enabled=True,
            qc_confidence=0.9,
        )

        generator = DocumentDatasetGenerator(config)

        assert generator.qc_controller is not None
        assert isinstance(generator.qc_controller, QualityController)

    def test_qc_controller_not_initialized_for_english(self):
        """
        QC controller should NOT be initialized when language is English.

        **Validates: Requirements 7.5, 7.6**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT,
            language=OutputLanguage.EN,
            qc_enabled=True,
            qc_confidence=0.9,
        )

        generator = DocumentDatasetGenerator(config)

        assert generator.qc_controller is None

    def test_qc_controller_not_initialized_when_disabled(self):
        """
        QC controller should NOT be initialized when qc_enabled=False.

        **Validates: Requirements 7.5**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT,
            language=OutputLanguage.ZH_TW,
            qc_enabled=False,
            qc_confidence=0.9,
        )

        generator = DocumentDatasetGenerator(config)

        assert generator.qc_controller is None

    def test_qc_confidence_passed_to_controller(self):
        """
        QC confidence threshold should be passed to the controller.

        **Validates: Requirements 7.5**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT,
            language=OutputLanguage.ZH_TW,
            qc_enabled=True,
            qc_confidence=0.85,
        )

        generator = DocumentDatasetGenerator(config)

        assert generator.qc_controller is not None
        assert generator.qc_controller.confidence_threshold == 0.85

    def test_get_qc_stats_returns_none_when_disabled(self):
        """
        get_qc_stats should return None when QC is disabled.

        **Validates: Requirements 7.5**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT, language=OutputLanguage.EN, qc_enabled=True
        )

        generator = DocumentDatasetGenerator(config)

        assert generator.get_qc_stats() is None

    def test_get_qc_stats_returns_stats_when_enabled(self):
        """
        get_qc_stats should return stats dict when QC is enabled for zh-tw.

        **Validates: Requirements 7.5, 7.6**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT, language=OutputLanguage.ZH_TW, qc_enabled=True
        )

        generator = DocumentDatasetGenerator(config)
        stats = generator.get_qc_stats()

        assert stats is not None
        assert "total_checked" in stats
        assert "passed" in stats
        assert "failed" in stats


class TestQCFiltering:
    """
    Tests for QC filtering functionality.

    **Validates: Requirements 7.5, 7.6**
    """

    def test_apply_qc_filter_passes_valid_entries(self):
        """
        _apply_qc_filter should pass entries that pass QC validation.

        **Validates: Requirements 7.5, 7.6**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT, language=OutputLanguage.ZH_TW, qc_enabled=True
        )

        generator = DocumentDatasetGenerator(config)

        # Mock the QC controller to always pass
        generator._qc_controller = MagicMock(spec=QualityController)
        generator._qc_controller.check_entry.return_value = (True, [])

        entries = [
            DataEntry(
                instruction="這個軟體的介面設計得很好",
                input="請說明",
                output="這是台灣繁體中文",
            )
        ]

        filtered = generator._apply_qc_filter(entries)

        assert len(filtered) == 1
        assert filtered[0] == entries[0]

    def test_apply_qc_filter_removes_invalid_entries(self):
        """
        _apply_qc_filter should remove entries that fail QC validation.

        **Validates: Requirements 7.5, 7.6**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT, language=OutputLanguage.ZH_TW, qc_enabled=True
        )

        generator = DocumentDatasetGenerator(config)

        # Mock the QC controller to fail
        generator._qc_controller = MagicMock(spec=QualityController)
        generator._qc_controller.check_entry.return_value = (False, ["instruction"])

        entries = [
            DataEntry(
                instruction="这个软件的界面设计得很好",  # Mainland Chinese
                input="请说明",
                output="这是简体中文",
            )
        ]

        filtered = generator._apply_qc_filter(entries)

        assert len(filtered) == 0

    def test_apply_qc_filter_mixed_entries(self):
        """
        _apply_qc_filter should correctly filter mixed valid/invalid entries.

        **Validates: Requirements 7.5, 7.6**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT, language=OutputLanguage.ZH_TW, qc_enabled=True
        )

        generator = DocumentDatasetGenerator(config)

        # Mock the QC controller to pass first, fail second
        generator._qc_controller = MagicMock(spec=QualityController)
        generator._qc_controller.check_entry.side_effect = [
            (True, []),  # First entry passes
            (False, ["instruction"]),  # Second entry fails
            (True, []),  # Third entry passes
        ]

        entries = [
            DataEntry(instruction="台灣用語", input="輸入", output="輸出"),
            DataEntry(instruction="大陆用语", input="输入", output="输出"),
            DataEntry(instruction="另一個台灣用語", input="輸入", output="輸出"),
        ]

        filtered = generator._apply_qc_filter(entries)

        assert len(filtered) == 2
        assert filtered[0] == entries[0]
        assert filtered[1] == entries[2]

    def test_apply_qc_filter_returns_all_when_no_controller(self):
        """
        _apply_qc_filter should return all entries when QC controller is None.

        **Validates: Requirements 7.5**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT,
            language=OutputLanguage.EN,  # English, so no QC
            qc_enabled=True,
        )

        generator = DocumentDatasetGenerator(config)

        entries = [
            DataEntry(instruction="Test", input="Input", output="Output"),
            DataEntry(instruction="Test2", input="Input2", output="Output2"),
        ]

        filtered = generator._apply_qc_filter(entries)

        assert len(filtered) == 2
        assert filtered == entries


class TestQCWithDifferentEntryTypes:
    """
    Tests for QC filtering with different entry types.

    **Validates: Requirements 7.5, 7.6**
    """

    def test_qc_filter_pretrain_entries(self):
        """
        QC filtering should work with PretrainEntry.

        **Validates: Requirements 7.5, 7.6**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.PRETRAIN,
            language=OutputLanguage.ZH_TW,
            qc_enabled=True,
        )

        generator = DocumentDatasetGenerator(config)

        # Mock the QC controller
        generator._qc_controller = MagicMock(spec=QualityController)
        generator._qc_controller.check_entry.return_value = (True, [])

        entries = [PretrainEntry(text="這是台灣繁體中文的預訓練文本")]

        filtered = generator._apply_qc_filter(entries)

        assert len(filtered) == 1
        # Verify entry_to_dict was called correctly
        generator._qc_controller.check_entry.assert_called_once()

    def test_qc_filter_conversation_entries(self):
        """
        QC filtering should work with SFTConversationEntry.

        **Validates: Requirements 7.5, 7.6**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT_CONVERSATION,
            language=OutputLanguage.ZH_TW,
            qc_enabled=True,
        )

        generator = DocumentDatasetGenerator(config)

        # Mock the QC controller
        generator._qc_controller = MagicMock(spec=QualityController)
        generator._qc_controller.check_entry.return_value = (True, [])

        entries = [
            SFTConversationEntry(
                conversations=[
                    ConversationMessage(role="user", content="你好"),
                    ConversationMessage(
                        role="assistant", content="您好！有什麼可以幫助您的？"
                    ),
                ]
            )
        ]

        filtered = generator._apply_qc_filter(entries)

        assert len(filtered) == 1

    def test_qc_filter_dpo_entries(self):
        """
        QC filtering should work with DPOEntry.

        **Validates: Requirements 7.5, 7.6**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.DPO, language=OutputLanguage.ZH_TW, qc_enabled=True
        )

        generator = DocumentDatasetGenerator(config)

        # Mock the QC controller
        generator._qc_controller = MagicMock(spec=QualityController)
        generator._qc_controller.check_entry.return_value = (True, [])

        entries = [
            DPOEntry(
                prompt="什麼是軟體工程？",
                chosen="軟體工程是一門研究如何系統化開發軟體的學科。",
                rejected="软件工程是一门研究如何系统化开发软件的学科。",
            )
        ]

        filtered = generator._apply_qc_filter(entries)

        assert len(filtered) == 1


class TestQCStatsTracking:
    """
    Tests for QC statistics tracking.

    **Validates: Requirements 7.5, 7.6**
    """

    def test_qc_stats_updated_after_filtering(self):
        """
        QC stats should be updated after filtering entries.

        **Validates: Requirements 7.5, 7.6**
        """
        config = DocGenerationConfig(
            dataset_type=DatasetType.SFT, language=OutputLanguage.ZH_TW, qc_enabled=True
        )

        generator = DocumentDatasetGenerator(config)

        # Create a real QC controller but mock the actual validation
        with patch("ollaforge.qc.validate_entry_chinese") as mock_validate:
            # First entry passes, second fails
            mock_validate.side_effect = [
                (True, []),
                (False, ["instruction"]),
            ]

            entries = [
                DataEntry(instruction="台灣", input="輸入", output="輸出"),
                DataEntry(instruction="大陆", input="输入", output="输出"),
            ]

            generator._apply_qc_filter(entries)

            stats = generator.get_qc_stats()

            assert stats is not None
            assert stats["total_checked"] == 2
            assert stats["passed"] == 1
            assert stats["failed"] == 1
