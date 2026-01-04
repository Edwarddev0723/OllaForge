"""
Quality Control module for OllaForge.

This module provides language quality control for Traditional Chinese (Taiwan) datasets.
It uses a BERT-based classifier to detect and filter out Mainland Chinese expressions,
ensuring generated content uses authentic Taiwan Traditional Chinese terminology.

Model: renhehuang/bert-traditional-chinese-classifier
"""

from typing import Any, Optional

from rich.console import Console

console = Console()

# --- Configuration ---
REPO_ID = "renhehuang/bert-traditional-chinese-classifier"
LABELS = {0: "Mainland Traditional", 1: "Taiwan Traditional"}
MAX_LEN, STRIDE = 384, 128
DEFAULT_CONFIDENCE_THRESHOLD = 0.9

# Global model cache
_model = None
_tokenizer = None
_device = None
_torch = None  # Lazy loaded torch module


def _get_torch():
    """Lazy load torch module."""
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
        except ImportError:
            return None
    return _torch


def get_device(force_cpu: bool = True) -> str:
    """
    Get the device for QC model.

    Args:
        force_cpu: Force CPU usage to avoid competing with LLM for GPU/MPS memory.
                   Default True for Mac optimization - keeps MPS free for Ollama.

    Returns:
        Device string: "cpu", "mps", or "cuda"
    """
    torch = _get_torch()
    if torch is None:
        return "cpu"

    if force_cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_qc_model(force_reload: bool = False) -> tuple[Any, Any, str]:
    """
    Load the QC model and tokenizer (lazy loading with caching).

    Args:
        force_reload: Force reload the model even if cached

    Returns:
        Tuple of (model, tokenizer, device)
    """
    global _model, _tokenizer, _device

    if _model is not None and _tokenizer is not None and not force_reload:
        return _model, _tokenizer, _device

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Force CPU for BERT to keep MPS/GPU free for LLM generation
        _device = get_device(force_cpu=True)

        console.print(
            f"[dim]Loading QC model on {_device} (keeping GPU free for LLM)...[/dim]"
        )

        _tokenizer = AutoTokenizer.from_pretrained(REPO_ID, cache_dir=".cache")
        _model = AutoModelForSequenceClassification.from_pretrained(
            REPO_ID, cache_dir=".cache"
        )
        _model.to(_device).eval()

        console.print("[green]✓ QC model loaded successfully[/green]")

        return _model, _tokenizer, _device

    except ImportError:
        console.print("[yellow]⚠️ transformers not installed. QC disabled.[/yellow]")
        console.print("[dim]Install with: pip install transformers torch[/dim]")
        return None, None, None
    except Exception as e:
        console.print(f"[yellow]⚠️ Failed to load QC model: {e}[/yellow]")
        return None, None, None


def chunk_encode(
    text: str, tokenizer: Any, max_len: int = MAX_LEN, stride: int = STRIDE
) -> list[dict]:
    """
    Encode text with chunking for long texts.

    Args:
        text: Input text to encode
        tokenizer: HuggingFace tokenizer
        max_len: Maximum sequence length
        stride: Stride for overlapping chunks

    Returns:
        List of encoded chunks
    """
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)[
        "input_ids"
    ]

    if len(ids) <= max_len - 2:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return [enc]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return [
        {
            "input_ids": enc["input_ids"][i : i + 1],
            "attention_mask": enc["attention_mask"][i : i + 1],
        }
        for i in range(len(enc["input_ids"]))
    ]


def predict_language(text: str) -> Optional[dict[str, Any]]:
    """
    Predict whether text is Taiwan Traditional or Mainland Traditional Chinese.

    Args:
        text: Text to classify

    Returns:
        Dictionary with prediction results, or None if model not available
    """
    torch = _get_torch()
    if torch is None:
        return None

    import torch.nn.functional as F

    model, tokenizer, device = load_qc_model()

    if model is None or tokenizer is None:
        return None

    try:
        with torch.inference_mode():
            chunks = chunk_encode(text, tokenizer)
            probs_all = []

            for ch in chunks:
                logits = model(
                    input_ids=ch["input_ids"].to(device),
                    attention_mask=ch["attention_mask"].to(device),
                ).logits
                probs_all.append(F.softmax(logits, dim=-1).cpu())

            avg = torch.cat(probs_all, 0).mean(0)
            label_id = int(avg.argmax())

        return {
            "text_preview": (text[:100] + "...") if len(text) > 100 else text,
            "predicted_id": label_id,
            "predicted_name": LABELS[label_id],
            "is_taiwan": label_id == 1,
            "confidence": float(avg[label_id]),
            "taiwan_probability": float(avg[1]),
            "mainland_probability": float(avg[0]),
            "num_chunks": len(chunks),
            "device": device,
        }

    except Exception as e:
        console.print(f"[yellow]⚠️ QC prediction failed: {e}[/yellow]")
        return None


def is_taiwan_chinese(
    text: str, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> tuple[bool, Optional[dict[str, Any]]]:
    """
    Check if text is authentic Taiwan Traditional Chinese.

    Args:
        text: Text to check
        confidence_threshold: Minimum confidence for Taiwan classification

    Returns:
        Tuple of (is_valid, prediction_result)
    """
    result = predict_language(text)

    if result is None:
        # If QC model not available, pass through
        return True, None

    is_valid = result["is_taiwan"] and result["confidence"] >= confidence_threshold

    return is_valid, result


def validate_entry_chinese(
    entry_dict: dict[str, Any],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[bool, list[str]]:
    """
    Validate all text fields in an entry for Taiwan Chinese.

    Args:
        entry_dict: Dictionary containing entry fields
        confidence_threshold: Minimum confidence threshold

    Returns:
        Tuple of (is_valid, list of failed field names)
    """
    failed_fields = []

    # Fields to check based on entry type
    text_fields = []

    if "instruction" in entry_dict:
        text_fields.extend(["instruction", "input", "output"])
    elif "text" in entry_dict:
        text_fields.append("text")
    elif "conversations" in entry_dict:
        # Check conversation content
        for i, msg in enumerate(entry_dict.get("conversations", [])):
            if isinstance(msg, dict) and "content" in msg:
                is_valid, _ = is_taiwan_chinese(msg["content"], confidence_threshold)
                if not is_valid:
                    failed_fields.append(f"conversations[{i}].content")
        return len(failed_fields) == 0, failed_fields
    elif "prompt" in entry_dict:
        text_fields.extend(["prompt", "chosen", "rejected"])

    for field in text_fields:
        if field in entry_dict and entry_dict[field]:
            text = str(entry_dict[field])
            if text.strip():  # Only check non-empty fields
                is_valid, _ = is_taiwan_chinese(text, confidence_threshold)
                if not is_valid:
                    failed_fields.append(field)

    return len(failed_fields) == 0, failed_fields


class QualityController:
    """
    Quality Controller for managing Chinese language validation.

    Uses a "funnel" architecture: no retries, just over-request and filter.
    """

    # Default pass rate assumption for buffer calculation
    DEFAULT_PASS_RATE = 0.7
    # Minimum buffer ratio to ensure we get enough entries
    MIN_BUFFER_RATIO = 1.2
    # Maximum buffer ratio to avoid excessive requests
    MAX_BUFFER_RATIO = 3.0

    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        estimated_pass_rate: float = DEFAULT_PASS_RATE,
    ):
        """
        Initialize the Quality Controller.

        Args:
            enabled: Whether QC is enabled
            confidence_threshold: Minimum confidence for Taiwan classification
            estimated_pass_rate: Estimated pass rate for buffer calculation (0.0-1.0)
        """
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.estimated_pass_rate = max(0.1, min(1.0, estimated_pass_rate))
        self.stats = {
            "total_checked": 0,
            "passed": 0,
            "failed": 0,
            "discarded": 0,
        }

    def calculate_request_count(self, target_count: int) -> int:
        """
        Calculate how many entries to request to achieve target count after QC filtering.

        Uses buffer_ratio = 1 / estimated_pass_rate, clamped to reasonable bounds.

        Args:
            target_count: Desired number of valid entries

        Returns:
            Number of entries to request (over-request amount)
        """
        if not self.enabled:
            return target_count

        # Calculate buffer ratio based on estimated pass rate
        buffer_ratio = 1.0 / self.estimated_pass_rate

        # Clamp to reasonable bounds
        buffer_ratio = max(
            self.MIN_BUFFER_RATIO, min(self.MAX_BUFFER_RATIO, buffer_ratio)
        )

        # Calculate request count and round up
        request_count = int(target_count * buffer_ratio + 0.5)

        return request_count

    def check_entry(self, entry_dict: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Check if an entry passes QC. No retries - just pass/fail.

        Args:
            entry_dict: Entry to check

        Returns:
            Tuple of (passed, failed_fields)
        """
        if not self.enabled:
            return True, []

        self.stats["total_checked"] += 1

        passed, failed_fields = validate_entry_chinese(
            entry_dict, self.confidence_threshold
        )

        if passed:
            self.stats["passed"] += 1
        else:
            self.stats["failed"] += 1
            self.stats["discarded"] += 1

        return passed, failed_fields

    def update_pass_rate(self) -> float:
        """
        Update estimated pass rate based on actual results.

        Returns:
            Updated pass rate
        """
        if self.stats["total_checked"] > 0:
            actual_rate = self.stats["passed"] / self.stats["total_checked"]
            # Blend with previous estimate (exponential moving average)
            self.estimated_pass_rate = (
                0.7 * actual_rate + 0.3 * self.estimated_pass_rate
            )
        return self.estimated_pass_rate

    def get_stats(self) -> dict[str, Any]:
        """Get QC statistics."""
        total = self.stats["total_checked"]
        return {
            **self.stats,
            "pass_rate": (self.stats["passed"] / total * 100) if total > 0 else 0,
            "estimated_pass_rate": self.estimated_pass_rate * 100,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_checked": 0,
            "passed": 0,
            "failed": 0,
            "discarded": 0,
        }


# --- Quick test ---
if __name__ == "__main__":
    tests = [
        "這個軟件的界面設計得很好。",  # Mainland
        "這個軟體的介面設計得很好。",  # Taiwan
        "我需要下載這個程序到計算機上。",  # Mainland
        "我需要下載這個程式到電腦上。",  # Taiwan
        "請問您的網絡連接正常嗎？",  # Mainland
        "請問您的網路連線正常嗎？",  # Taiwan
    ]

    print("=" * 60)
    print("OllaForge QC Module Test")
    print("=" * 60)

    for t in tests:
        result = predict_language(t)
        if result:
            status = "✅ Taiwan" if result["is_taiwan"] else "❌ Mainland"
            print(
                f"{status} | conf={result['confidence']:.2%} | {result['text_preview']}"
            )
        else:
            print(f"⚠️ QC unavailable | {t[:50]}")
