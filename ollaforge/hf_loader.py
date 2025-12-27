"""
HuggingFace datasets loader for OllaForge.

This module provides functionality to load datasets directly from HuggingFace Hub
for augmentation. It supports various dataset configurations and splits.

Key features:
- Load datasets by name (e.g., "renhehuang/govQA-database-zhtw")
- Support for dataset configurations and splits
- Automatic conversion to OllaForge format
- Streaming support for large datasets
- Progress tracking during download

Example usage:
    from ollaforge.hf_loader import load_huggingface_dataset
    
    entries, fields = load_huggingface_dataset("renhehuang/govQA-database-zhtw")
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from rich.console import Console

console = Console()


class HuggingFaceLoaderError(Exception):
    """Raised when HuggingFace dataset loading fails."""
    pass


def is_huggingface_dataset(input_str: str) -> bool:
    """
    Check if the input string looks like a HuggingFace dataset identifier.
    
    HuggingFace dataset identifiers typically follow the pattern:
    - username/dataset-name
    - organization/dataset-name
    
    Args:
        input_str: Input string to check
        
    Returns:
        True if it looks like a HuggingFace dataset identifier
    """
    # Skip if it looks like a file path
    if input_str.startswith('/') or input_str.startswith('./') or input_str.startswith('..'):
        return False
    
    # Skip if it has a file extension
    common_extensions = ['.jsonl', '.json', '.csv', '.tsv', '.parquet', '.txt', '.xlsx']
    for ext in common_extensions:
        if input_str.lower().endswith(ext):
            return False
    
    # Check for HuggingFace pattern: username/dataset-name
    # Allow alphanumeric, hyphens, underscores, and dots
    hf_pattern = r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$'
    return bool(re.match(hf_pattern, input_str))


def load_huggingface_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "train",
    max_entries: Optional[int] = None,
    streaming: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    trust_remote_code: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load a dataset from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "renhehuang/govQA-database-zhtw")
        config_name: Optional dataset configuration name
        split: Dataset split to load (default: "train")
        max_entries: Maximum number of entries to load (None for all)
        streaming: Whether to use streaming mode for large datasets
        progress_callback: Optional callback(loaded, total) for progress updates
        trust_remote_code: Whether to trust remote code in the dataset
        
    Returns:
        Tuple of (entries, field_names)
        
    Raises:
        HuggingFaceLoaderError: If dataset cannot be loaded
    """
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        raise HuggingFaceLoaderError(
            "HuggingFace datasets library not installed. "
            "Install with: pip install datasets"
        )
    
    console.print(f"[cyan]📥 Loading HuggingFace dataset: {dataset_name}[/cyan]")
    
    try:
        # Try to get available configs if not specified
        if config_name is None:
            try:
                configs = get_dataset_config_names(dataset_name, trust_remote_code=trust_remote_code)
                if configs and len(configs) > 1:
                    console.print(f"[dim]Available configs: {', '.join(configs)}[/dim]")
                    console.print(f"[dim]Using default config. Specify with --config if needed.[/dim]")
            except Exception:
                pass  # Config detection failed, proceed without config
        
        # Load dataset
        load_kwargs = {
            "path": dataset_name,
            "split": split,
            "streaming": streaming,
            "trust_remote_code": trust_remote_code,
        }
        
        if config_name:
            load_kwargs["name"] = config_name
        
        console.print(f"[dim]Loading split: {split}[/dim]")
        dataset = load_dataset(**load_kwargs)
        
        # Convert to list of dicts
        entries: List[Dict[str, Any]] = []
        field_names: set = set()
        
        if streaming:
            # Streaming mode - iterate and collect
            console.print("[dim]Streaming mode enabled[/dim]")
            for i, item in enumerate(dataset):
                if max_entries and i >= max_entries:
                    break
                
                entry = dict(item)
                entries.append(entry)
                field_names.update(entry.keys())
                
                if progress_callback and (i + 1) % 100 == 0:
                    progress_callback(i + 1, max_entries or -1)
        else:
            # Non-streaming mode
            total = len(dataset)
            console.print(f"[dim]Dataset size: {total} entries[/dim]")
            
            if max_entries:
                dataset = dataset.select(range(min(max_entries, total)))
                total = len(dataset)
            
            for i, item in enumerate(dataset):
                entry = dict(item)
                entries.append(entry)
                field_names.update(entry.keys())
                
                if progress_callback and (i + 1) % 100 == 0:
                    progress_callback(i + 1, total)
        
        console.print(f"[green]✅ Loaded {len(entries)} entries from {dataset_name}[/green]")
        console.print(f"[dim]Fields: {', '.join(sorted(field_names))}[/dim]")
        
        return entries, sorted(field_names)
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful error messages
        if "doesn't exist" in error_msg.lower() or "not found" in error_msg.lower():
            raise HuggingFaceLoaderError(
                f"Dataset '{dataset_name}' not found on HuggingFace Hub. "
                f"Please check the dataset name and ensure it exists."
            )
        elif "split" in error_msg.lower():
            raise HuggingFaceLoaderError(
                f"Split '{split}' not found in dataset '{dataset_name}'. "
                f"Try using a different split (e.g., 'train', 'test', 'validation')."
            )
        elif "config" in error_msg.lower():
            raise HuggingFaceLoaderError(
                f"Configuration error for dataset '{dataset_name}'. "
                f"Try specifying a config name with --config."
            )
        else:
            raise HuggingFaceLoaderError(
                f"Failed to load dataset '{dataset_name}': {error_msg}"
            )


def get_dataset_info(dataset_name: str, trust_remote_code: bool = False) -> Dict[str, Any]:
    """
    Get information about a HuggingFace dataset without loading it.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Dictionary with dataset information
        
    Raises:
        HuggingFaceLoaderError: If dataset info cannot be retrieved
    """
    try:
        from datasets import load_dataset_builder, get_dataset_config_names
    except ImportError:
        raise HuggingFaceLoaderError(
            "HuggingFace datasets library not installed. "
            "Install with: pip install datasets"
        )
    
    try:
        # Get available configs
        configs = []
        try:
            configs = get_dataset_config_names(dataset_name, trust_remote_code=trust_remote_code)
        except Exception:
            pass
        
        # Get dataset builder info
        builder = load_dataset_builder(dataset_name, trust_remote_code=trust_remote_code)
        
        info = {
            "name": dataset_name,
            "description": builder.info.description or "No description available",
            "configs": configs,
            "features": {},
            "splits": [],
        }
        
        # Get features (field names and types)
        if builder.info.features:
            info["features"] = {
                name: str(feature) 
                for name, feature in builder.info.features.items()
            }
        
        # Get available splits
        if builder.info.splits:
            info["splits"] = list(builder.info.splits.keys())
        
        return info
        
    except Exception as e:
        raise HuggingFaceLoaderError(
            f"Failed to get info for dataset '{dataset_name}': {str(e)}"
        )


def list_dataset_splits(dataset_name: str, config_name: Optional[str] = None) -> List[str]:
    """
    List available splits for a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        config_name: Optional dataset configuration name
        
    Returns:
        List of available split names
        
    Raises:
        HuggingFaceLoaderError: If splits cannot be retrieved
    """
    try:
        from datasets import get_dataset_split_names
    except ImportError:
        raise HuggingFaceLoaderError(
            "HuggingFace datasets library not installed. "
            "Install with: pip install datasets"
        )
    
    try:
        return get_dataset_split_names(dataset_name, config_name)
    except Exception as e:
        raise HuggingFaceLoaderError(
            f"Failed to get splits for dataset '{dataset_name}': {str(e)}"
        )
