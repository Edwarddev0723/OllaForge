"""
Augmentation service for OllaForge web interface.

This module provides an async wrapper around the existing DatasetAugmentor
class, adding support for file management, progress callbacks, and task management.

Requirements satisfied:
- 2.1: Upload and validate dataset files
- 2.2: Validate target field exists in dataset
- 2.3: Preview augmentation before full processing
- 2.4: Provide download link for augmented dataset
- 2.5: Preserve original data on failure
"""

import asyncio
import uuid
import os
import tempfile
import shutil
from typing import Optional, Callable, Dict, Any, List, Tuple
from datetime import datetime
import time

from ...augmentor import DatasetAugmentor
from ...models import AugmentationConfig, OutputLanguage
from ...file_manager import read_dataset_file, FileOperationError
from ...formats import write_file, FileFormat


class AugmentationService:
    """
    Service for managing dataset augmentation tasks.
    
    Wraps the existing DatasetAugmentor with async interface,
    file management, and progress tracking support.
    """
    
    def __init__(self):
        """Initialize the augmentation service."""
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.uploaded_files: Dict[str, Dict[str, Any]] = {}
        
        # Create temp directory for uploaded files
        self.temp_dir = tempfile.mkdtemp(prefix="ollaforge_uploads_")
    
    def __del__(self):
        """Cleanup temp directory on service destruction."""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    async def upload_file(
        self,
        file_content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """
        Upload and process a dataset file.
        
        Args:
            file_content: Raw file content bytes
            filename: Original filename
            
        Returns:
            Dict containing:
                - file_id: Unique identifier for the uploaded file
                - entry_count: Number of entries in the dataset
                - fields: List of field names
                - preview: First 3 entries for preview
                
        Raises:
            FileOperationError: If file cannot be read or parsed
            
        Requirements satisfied:
        - 2.1: Upload and validate dataset files
        - 5.3: Display first 3 entries
        """
        # Generate unique file ID
        file_id = f"file_{uuid.uuid4().hex[:12]}"
        
        # Save file to temp directory
        file_path = os.path.join(self.temp_dir, f"{file_id}_{filename}")
        
        # Run file operations in thread pool
        loop = asyncio.get_event_loop()
        
        def save_and_read():
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Read and parse file
            entries, field_names = read_dataset_file(file_path)
            return entries, field_names
        
        try:
            entries, field_names = await loop.run_in_executor(None, save_and_read)
            
            # Store file info
            self.uploaded_files[file_id] = {
                "file_path": file_path,
                "filename": filename,
                "entries": entries,
                "fields": field_names,
                "entry_count": len(entries),
                "uploaded_at": datetime.utcnow().isoformat()
            }
            
            # Return upload response
            return {
                "file_id": file_id,
                "entry_count": len(entries),
                "fields": field_names,
                "preview": entries[:3]  # First 3 entries for preview
            }
            
        except Exception as e:
            # Clean up file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise FileOperationError(f"Failed to process uploaded file: {str(e)}")
    
    def get_uploaded_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get uploaded file info.
        
        Args:
            file_id: File identifier
            
        Returns:
            File info dict or None if not found
        """
        return self.uploaded_files.get(file_id)
    
    async def store_huggingface_dataset(
        self,
        entries: List[Dict[str, Any]],
        fields: List[str],
        dataset_name: str,
        split: str = "train",
    ) -> Dict[str, Any]:
        """
        Store a HuggingFace dataset for augmentation.
        
        Args:
            entries: List of dataset entries
            fields: List of field names
            dataset_name: HuggingFace dataset identifier
            split: Dataset split name
            
        Returns:
            Dict containing:
                - file_id: Unique identifier for the dataset
                - entry_count: Number of entries
                - fields: List of field names
                - preview: First 3 entries for preview
        """
        # Generate unique file ID
        file_id = f"hf_{uuid.uuid4().hex[:12]}"
        
        # Store dataset info (no file path needed for HuggingFace datasets)
        self.uploaded_files[file_id] = {
            "file_path": None,  # No local file
            "filename": f"{dataset_name.replace('/', '_')}_{split}",
            "entries": entries,
            "fields": fields,
            "entry_count": len(entries),
            "uploaded_at": datetime.utcnow().isoformat(),
            "source": "huggingface",
            "dataset_name": dataset_name,
            "split": split,
        }
        
        # Return response
        return {
            "file_id": file_id,
            "entry_count": len(entries),
            "fields": fields,
            "preview": entries[:3]  # First 3 entries for preview
        }
    
    def validate_field(
        self,
        file_id: str,
        target_field: str,
        create_new_field: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that target field exists or can be created.
        
        Args:
            file_id: File identifier
            target_field: Target field name to validate
            create_new_field: Whether to allow creating new fields
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Requirements satisfied:
        - 2.2: Validate target field exists in dataset
        """
        file_info = self.uploaded_files.get(file_id)
        if not file_info:
            return False, f"File not found: {file_id}"
        
        fields = file_info["fields"]
        
        if target_field in fields:
            return True, None
        
        if create_new_field:
            return True, None
        
        return False, f"Field '{target_field}' not found. Available fields: {', '.join(fields)}"
    
    async def preview_augmentation(
        self,
        file_id: str,
        target_field: str,
        instruction: str,
        model: str = "llama3.2",
        create_new_field: bool = False,
        context_fields: List[str] = None,
        preview_count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate preview of augmentation on sample entries.
        
        Args:
            file_id: File identifier
            target_field: Field to augment
            instruction: Augmentation instruction
            model: Ollama model name
            create_new_field: Whether to create new field
            context_fields: Fields to include as context
            preview_count: Number of entries to preview
            
        Returns:
            List of preview dicts with original and augmented entries
            
        Requirements satisfied:
        - 2.3: Preview augmentation before full processing
        - 5.2: Show before and after comparison
        """
        if context_fields is None:
            context_fields = []
        
        file_info = self.uploaded_files.get(file_id)
        if not file_info:
            raise FileOperationError(f"File not found: {file_id}")
        
        entries = file_info["entries"]
        
        # Create augmentation config for preview
        config = AugmentationConfig(
            input_file=file_info["file_path"],
            output_file="",  # Not needed for preview
            target_field=target_field,
            instruction=instruction,
            model=model,
            create_new_field=create_new_field,
            context_fields=context_fields,
            preview_count=preview_count
        )
        
        # Run preview in thread pool
        loop = asyncio.get_event_loop()
        
        def run_preview():
            augmentor = DatasetAugmentor(config)
            return augmentor.preview(entries)
        
        try:
            preview_results = await loop.run_in_executor(None, run_preview)
            
            # Format results
            previews = []
            for original, augmented in preview_results:
                previews.append({
                    "original": original,
                    "augmented": augmented
                })
            
            return previews
            
        except Exception as e:
            raise FileOperationError(f"Preview failed: {str(e)}")

    async def augment_dataset(
        self,
        file_id: str,
        target_field: str,
        instruction: str,
        model: str = "llama3.2",
        language: OutputLanguage = OutputLanguage.EN,
        create_new_field: bool = False,
        context_fields: List[str] = None,
        concurrency: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Augment entire dataset with progress updates.
        
        Args:
            file_id: File identifier
            target_field: Field to augment
            instruction: Augmentation instruction
            model: Ollama model name
            language: Output language
            create_new_field: Whether to create new field
            context_fields: Fields to include as context
            concurrency: Number of parallel workers
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            Dict containing:
                - entries: List of augmented entries
                - total: Total number of entries
                - success_count: Number of successful augmentations
                - failure_count: Number of failed augmentations
                - duration: Augmentation duration in seconds
                
        Requirements satisfied:
        - 2.4: Provide download link for augmented dataset
        - 2.5: Preserve original data on failure
        """
        if context_fields is None:
            context_fields = []
        
        file_info = self.uploaded_files.get(file_id)
        if not file_info:
            raise FileOperationError(f"File not found: {file_id}")
        
        entries = file_info["entries"]
        
        # Create output file path
        output_file = os.path.join(
            self.temp_dir,
            f"augmented_{file_id}.jsonl"
        )
        
        # Create augmentation config
        config = AugmentationConfig(
            input_file=file_info["file_path"],
            output_file=output_file,
            target_field=target_field,
            instruction=instruction,
            model=model,
            language=language,
            create_new_field=create_new_field,
            context_fields=context_fields
        )
        
        start_time = time.time()
        
        # Run augmentation in thread pool
        loop = asyncio.get_event_loop()
        
        # Wrapper for progress callback
        def sync_progress_callback(completed: int, total: int):
            if progress_callback:
                asyncio.run_coroutine_threadsafe(
                    self._async_progress_callback(progress_callback, completed, total),
                    loop
                )
        
        def run_augmentation():
            augmentor = DatasetAugmentor(config)
            result = augmentor.augment_dataset(
                entries,
                concurrency=concurrency,
                progress_callback=sync_progress_callback
            )
            augmented_entries = augmentor.get_augmented_entries()
            return result, augmented_entries
        
        try:
            result, augmented_entries = await loop.run_in_executor(
                None, run_augmentation
            )
            
            duration = time.time() - start_time
            
            return {
                "entries": augmented_entries,
                "total": result.total_entries,
                "success_count": result.success_count,
                "failure_count": result.failure_count,
                "duration": duration,
                "errors": result.errors[:10] if result.errors else []  # Limit errors
            }
            
        except Exception as e:
            raise FileOperationError(f"Augmentation failed: {str(e)}")
    
    async def _async_progress_callback(
        self,
        callback: Callable[[int, int], None],
        completed: int,
        total: int
    ):
        """Async wrapper for progress callback."""
        if asyncio.iscoroutinefunction(callback):
            await callback(completed, total)
        else:
            callback(completed, total)
    
    def create_task(self) -> str:
        """
        Create a new task and return its ID.
        
        Returns:
            Task ID string
        """
        task_id = f"aug_{uuid.uuid4().hex[:12]}"
        self.tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "total": 0,
            "result": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat()
        }
        return task_id
    
    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        total: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """
        Update task status.
        
        Args:
            task_id: Task identifier
            status: New status (pending, running, completed, failed)
            progress: Current progress count
            total: Total items count
            result: Result data
            error: Error message
        """
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        
        if status is not None:
            task["status"] = status
        if progress is not None:
            task["progress"] = progress
        if total is not None:
            task["total"] = total
        if result is not None:
            task["result"] = result
        if error is not None:
            task["error"] = error
        
        task["updated_at"] = datetime.utcnow().isoformat()
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task data dict or None if not found
        """
        return self.tasks.get(task_id)
    
    def delete_task(self, task_id: str):
        """
        Delete a task.
        
        Args:
            task_id: Task identifier
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
    
    def delete_file(self, file_id: str):
        """
        Delete an uploaded file.
        
        Args:
            file_id: File identifier
        """
        file_info = self.uploaded_files.get(file_id)
        if file_info:
            # Remove file from disk
            file_path = file_info.get("file_path")
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            
            # Remove from tracking
            del self.uploaded_files[file_id]
