"""
LLM Logger utility for capturing all LLM calls and module diagnostics.
Logs are written in a human-readable and machine-readable format (JSON).
"""
import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from contextlib import contextmanager

class LLMLogger:
    """
    Thread-safe logger for LLM calls and module diagnostics.
    Logs are written in JSON format for both human and machine readability.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, log_file_path: Optional[str] = None, enabled: bool = True):
        """
        Initialize the logger.
        
        Args:
            log_file_path: Path to the log file. If None, logging is disabled.
            enabled: Whether logging is enabled.
        """
        self.log_file_path = log_file_path
        self.enabled = enabled and (log_file_path is not None)
        self.write_lock = threading.Lock()
        self.call_counter = 0
        
        if self.enabled:
            # Ensure directory exists
            if log_file_path:
                os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else '.', exist_ok=True)
                # Initialize log file with header
                self._write_header()
    
    @classmethod
    def get_instance(cls, log_file_path: Optional[str] = None, enabled: bool = True):
        """Get singleton instance of the logger."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(log_file_path, enabled)
        return cls._instance
    
    def _write_header(self):
        """Write header to log file."""
        header = {
            "log_type": "header",
            "timestamp": datetime.now().isoformat(),
            "description": "LLM Call and Module Diagnostic Logs",
            "format": "Each log entry is a JSON object on a single line"
        }
        self._write_log_entry(header)
    
    def _write_log_entry(self, entry: Dict[str, Any]):
        """Write a log entry to the file."""
        if not self.enabled:
            return
        
        with self.write_lock:
            try:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            except Exception as e:
                # Silently fail if logging fails to avoid breaking the main application
                pass
    
    def log_llm_call(
        self,
        module_name: str,
        function_name: str,
        call_id: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
        response: Optional[Union[str, List[str]]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
        **kwargs
    ):
        """
        Log an LLM call with full input and output.
        
        Args:
            module_name: Name of the module making the call (e.g., 'memory', 'reasoning')
            function_name: Name of the function making the call
            call_id: Optional unique ID for this call
            messages: Input messages to the LLM
            model: Model name used
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            stop_strs: Stop strings parameter
            n: Number of responses requested
            response: Response from the LLM
            error: Error message if call failed
            duration_ms: Duration of the call in milliseconds
            **kwargs: Additional parameters
        """
        if not self.enabled:
            return
        
        if call_id is None:
            with self.write_lock:
                self.call_counter += 1
                call_id = self.call_counter
        
        entry = {
            "log_type": "llm_call",
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "module": module_name,
            "function": function_name,
            "input": {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop_strs": stop_strs,
                "n": n,
                **kwargs
            },
            "output": {
                "response": response,
                "error": error,
                "duration_ms": duration_ms
            }
        }
        
        self._write_log_entry(entry)
    
    def log_module_diagnostic(
        self,
        module_name: str,
        function_name: str,
        event_type: str,
        data: Dict[str, Any],
        task_id: Optional[str] = None
    ):
        """
        Log diagnostic information from modules.
        
        Args:
            module_name: Name of the module (e.g., 'memory', 'planning')
            function_name: Name of the function
            event_type: Type of event (e.g., 'memory_retrieval', 'plan_generated')
            data: Diagnostic data dictionary
            task_id: Optional task identifier
        """
        if not self.enabled:
            return
        
        entry = {
            "log_type": "module_diagnostic",
            "timestamp": datetime.now().isoformat(),
            "module": module_name,
            "function": function_name,
            "event_type": event_type,
            "task_id": task_id,
            "data": data
        }
        
        self._write_log_entry(entry)
    
    def log_simulation_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        task_id: Optional[str] = None
    ):
        """
        Log simulation-level events.
        
        Args:
            event_type: Type of event (e.g., 'task_start', 'task_complete')
            data: Event data dictionary
            task_id: Optional task identifier
        """
        if not self.enabled:
            return
        
        entry = {
            "log_type": "simulation_event",
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "task_id": task_id,
            "data": data
        }
        
        self._write_log_entry(entry)
    
    @contextmanager
    def log_llm_call_context(
        self,
        module_name: str,
        function_name: str,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
        **kwargs
    ):
        """
        Context manager for logging LLM calls with automatic timing.
        
        Usage:
            with logger.log_llm_call_context(...) as call_info:
                response = llm(...)
                call_info['response'] = response
        """
        import time
        start_time = time.time()
        call_id = None
        
        with self.write_lock:
            self.call_counter += 1
            call_id = self.call_counter
        
        call_info = {
            'call_id': call_id,
            'response': None,
            'error': None
        }
        
        try:
            yield call_info
        except Exception as e:
            call_info['error'] = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.log_llm_call(
                module_name=module_name,
                function_name=function_name,
                call_id=call_id,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_strs=stop_strs,
                n=n,
                response=call_info['response'],
                error=call_info['error'],
                duration_ms=duration_ms,
                **kwargs
            )

