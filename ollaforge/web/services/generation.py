"""
Generation service for OllaForge web interface.

This module provides an async wrapper around the existing generate_data_concurrent
function, adding support for progress callbacks and task management.

Requirements satisfied:
- 1.2: Initiate dataset generation with valid parameters
- 3.1: Display progress bar during generation
- 3.2: Real-time progress updates
"""

import asyncio
import uuid
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
import time

from ...client import generate_data_concurrent, OllamaConnectionError, OllamaGenerationError
from ...processor import process_model_response
from ...models import DatasetType, OutputLanguage, DatasetEntry
from ...qc import QualityController


class GenerationService:
    """
    Service for managing dataset generation tasks.
    
    Wraps the existing generate_data_concurrent function with async interface
    and progress tracking support.
    """
    
    def __init__(self):
        """Initialize the generation service."""
        self.tasks: Dict[str, Dict[str, Any]] = {}
    
    async def generate_dataset(
        self,
        topic: str,
        count: int,
        model: str,
        dataset_type: DatasetType = DatasetType.SFT,
        language: OutputLanguage = OutputLanguage.EN,
        qc_enabled: bool = True,
        qc_confidence: float = 0.9,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Generate dataset with progress updates.
        
        Args:
            topic: Topic description for dataset generation
            count: Number of entries to generate
            model: Ollama model name
            dataset_type: Type of dataset to generate
            language: Output language for generated content
            qc_enabled: Enable QC filtering for Traditional Chinese
            qc_confidence: QC confidence threshold
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            Dict containing:
                - entries: List of generated dataset entries
                - total: Total number of entries generated
                - duration: Generation duration in seconds
                - qc_stats: QC statistics if QC was enabled
                
        Raises:
            OllamaConnectionError: If connection to Ollama fails
            OllamaGenerationError: If generation fails
        """
        start_time = time.time()
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Wrapper for progress callback to handle async context
        def sync_progress_callback(completed: int, total: int):
            if progress_callback:
                # Schedule callback in event loop
                asyncio.run_coroutine_threadsafe(
                    self._async_progress_callback(progress_callback, completed, total),
                    loop
                )
        
        try:
            # Run the synchronous generation function in a thread pool
            responses = await loop.run_in_executor(
                None,
                lambda: generate_data_concurrent(
                    topic=topic,
                    model=model,
                    total_count=count,
                    dataset_type=dataset_type,
                    language=language,
                    progress_callback=sync_progress_callback
                )
            )
            
            # Process responses to extract valid entries
            entries = []
            for response in responses:
                if 'raw_content' in response:
                    processed = process_model_response(
                        response['raw_content'],
                        is_batch=response.get('is_batch', False),
                        dataset_type=dataset_type
                    )
                    if isinstance(processed, list):
                        entries.extend(processed)
                    elif processed is not None:
                        entries.append(processed)
            
            # Apply QC if enabled and language is Traditional Chinese
            qc_stats = None
            if qc_enabled and language == OutputLanguage.ZH_TW:
                qc = QualityController(confidence_threshold=qc_confidence)
                entries, qc_stats = await loop.run_in_executor(
                    None,
                    lambda: self._apply_qc(qc, entries)
                )
            
            duration = time.time() - start_time
            
            return {
                "entries": entries,
                "total": len(entries),
                "duration": duration,
                "qc_stats": qc_stats
            }
            
        except (OllamaConnectionError, OllamaGenerationError):
            # Re-raise these specific errors
            raise
        except Exception as e:
            # Wrap other exceptions
            raise OllamaGenerationError(f"Generation failed: {str(e)}")
    
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
    
    def _apply_qc(
        self,
        qc: QualityController,
        entries: List[DatasetEntry]
    ) -> tuple[List[DatasetEntry], Dict[str, Any]]:
        """
        Apply QC filtering to entries.
        
        Args:
            qc: QualityController instance
            entries: List of entries to filter
            
        Returns:
            Tuple of (filtered_entries, qc_stats)
        """
        # Extract text content from entries for QC
        texts = []
        for entry in entries:
            if hasattr(entry, 'text'):
                texts.append(entry.text)
            elif hasattr(entry, 'output'):
                texts.append(entry.output)
            elif hasattr(entry, 'conversations'):
                # For conversation entries, check all messages
                for msg in entry.conversations:
                    texts.append(msg.content)
        
        # Run QC check
        passed_indices = []
        failed_count = 0
        
        for i, text in enumerate(texts):
            if qc.is_taiwan_chinese(text):
                passed_indices.append(i)
            else:
                failed_count += 1
        
        # Filter entries
        filtered_entries = [entries[i] for i in passed_indices if i < len(entries)]
        
        qc_stats = {
            "total_checked": len(entries),
            "passed": len(filtered_entries),
            "failed": failed_count,
            "pass_rate": len(filtered_entries) / len(entries) if entries else 0
        }
        
        return filtered_entries, qc_stats
    
    def create_task(self) -> str:
        """
        Create a new task and return its ID.
        
        Returns:
            Task ID string
        """
        task_id = f"gen_{uuid.uuid4().hex[:12]}"
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
