"""
Quality Control module for OllaForge.

This module provides language quality control for Traditional Chinese (Taiwan) datasets.
It uses a BERT-based classifier to detect and filter out Mainland Chinese expressions,
ensuring generated content uses authentic Taiwan Traditional Chinese terminology.

Model: renhehuang/bert-traditional-chinese-classifier
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
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


def get_device() -> str:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_qc_model(force_reload: bool = False) -> Tuple[Any, Any, str]:
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
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        _device = get_device()
        
        console.print(f"[dim]Loading QC model on {_device}...[/dim]")
        
        _tokenizer = AutoTokenizer.from_pretrained(REPO_ID, cache_dir=".cache")
        _model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, cache_dir=".cache")
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


def chunk_encode(text: str, tokenizer: Any, max_len: int = MAX_LEN, stride: int = STRIDE) -> List[Dict]:
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
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    
    if len(ids) <= max_len - 2:
        enc = tokenizer(
            text, 
            truncation=True, 
            max_length=max_len,
            return_attention_mask=True, 
            return_tensors="pt"
        )
        return [enc]
    
    enc = tokenizer(
        text, 
        truncation=True, 
        max_length=max_len, 
        stride=stride,
        return_overflowing_tokens=True, 
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    return [
        {
            "input_ids": enc["input_ids"][i:i+1],
            "attention_mask": enc["attention_mask"][i:i+1]
        }
        for i in range(len(enc["input_ids"]))
    ]


@torch.inference_mode()
def predict_language(text: str) -> Optional[Dict[str, Any]]:
    """
    Predict whether text is Taiwan Traditional or Mainland Traditional Chinese.
    
    Args:
        text: Text to classify
        
    Returns:
        Dictionary with prediction results, or None if model not available
    """
    model, tokenizer, device = load_qc_model()
    
    if model is None or tokenizer is None:
        return None
    
    try:
        chunks = chunk_encode(text, tokenizer)
        probs_all = []
        
        for ch in chunks:
            logits = model(
                input_ids=ch["input_ids"].to(device),
                attention_mask=ch["attention_mask"].to(device)
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
    text: str, 
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Tuple[bool, Optional[Dict[str, Any]]]:
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
    entry_dict: Dict[str, Any],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Tuple[bool, List[str]]:
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
    """
    
    def __init__(
        self, 
        enabled: bool = True,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_retries: int = 3
    ):
        """
        Initialize the Quality Controller.
        
        Args:
            enabled: Whether QC is enabled
            confidence_threshold: Minimum confidence for Taiwan classification
            max_retries: Maximum retries for failed entries
        """
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.stats = {
            "total_checked": 0,
            "passed": 0,
            "failed": 0,
            "retried": 0,
        }
    
    def check_entry(self, entry_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if an entry passes QC.
        
        Args:
            entry_dict: Entry to check
            
        Returns:
            Tuple of (passed, failed_fields)
        """
        if not self.enabled:
            return True, []
        
        self.stats["total_checked"] += 1
        
        passed, failed_fields = validate_entry_chinese(
            entry_dict, 
            self.confidence_threshold
        )
        
        if passed:
            self.stats["passed"] += 1
        else:
            self.stats["failed"] += 1
        
        return passed, failed_fields
    
    def get_stats(self) -> Dict[str, Any]:
        """Get QC statistics."""
        total = self.stats["total_checked"]
        return {
            **self.stats,
            "pass_rate": (self.stats["passed"] / total * 100) if total > 0 else 0,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_checked": 0,
            "passed": 0,
            "failed": 0,
            "retried": 0,
        }


# --- Quick test ---
if __name__ == "__main__":
    tests = [
        "這個軟件的界面設計得很好。",      # Mainland
        "這個軟體的介面設計得很好。",      # Taiwan
        "我需要下載這個程序到計算機上。",  # Mainland
        "我需要下載這個程式到電腦上。",    # Taiwan
        "請問您的網絡連接正常嗎？",        # Mainland
        "請問您的網路連線正常嗎？",        # Taiwan
    ]
    
    print("=" * 60)
    print("OllaForge QC Module Test")
    print("=" * 60)
    
    for t in tests:
        result = predict_language(t)
        if result:
            status = "✅ Taiwan" if result["is_taiwan"] else "❌ Mainland"
            print(f"{status} | conf={result['confidence']:.2%} | {result['text_preview']}")
        else:
            print(f"⚠️ QC unavailable | {t[:50]}")
