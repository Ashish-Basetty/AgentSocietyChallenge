"""
Comprehensive logging utility for LLM calls and module diagnostics.
Logs are stored in JSON format for machine readability while being human-readable.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from threading import Lock
import traceback

class DiagnosticLogger:
    """Logger for LLM calls and module diagnostics"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.log_file = None
        self.enabled = True
        self.logs = []
        self.call_counter = 0
        
    def initialize(self, log_file_path: str, enabled: bool = True):
        """Initialize the logger with a file path"""
        self.log_file = log_file_path
        self.enabled = enabled
        self.logs = []
        self.call_counter = 0
        
        # Create directory if it doesn't exist
        if log_file_path:
            os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else '.', exist_ok=True)
    
    def log_llm_call(
        self,
        module_name: str,
        call_type: str,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
        response: Any,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an LLM call with full context"""
        if not self.enabled:
            return
            
        self.call_counter += 1
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "call_id": self.call_counter,
            "type": "llm_call",
            "module": module_name,
            "call_type": call_type,
            "input": {
                "messages": messages,
                "parameters": parameters
            },
            "output": {
                "response": response if isinstance(response, (str, int, float, bool, list, dict)) else str(response),
                "response_type": type(response).__name__
            },
            "error": error,
            "metadata": metadata or {}
        }
        
        self.logs.append(log_entry)
        self._write_log_entry(log_entry)
    
    def log_module_diagnostic(
        self,
        module_name: str,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log diagnostic information from modules"""
        if not self.enabled:
            return
            
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "type": "module_diagnostic",
            "module": module_name,
            "event_type": event_type,
            "data": data,
            "metadata": metadata or {}
        }
        
        self.logs.append(log_entry)
        self._write_log_entry(log_entry)
    
    def log_workflow_event(
        self,
        agent_name: str,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log workflow-level events"""
        if not self.enabled:
            return
            
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "type": "workflow_event",
            "agent": agent_name,
            "event_type": event_type,
            "data": data,
            "metadata": metadata or {}
        }
        
        self.logs.append(log_entry)
        self._write_log_entry(log_entry)
    
    def _serialize_value(self, value):
        """Recursively serialize values to be JSON-compatible"""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {str(k): self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, set):
            return list(value)  # Convert sets to lists
        else:
            return str(value)  # Convert other types to strings
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write a single log entry to file in JSON Lines format"""
        if not self.log_file:
            return
            
        try:
            # Serialize the log entry to ensure JSON compatibility
            serialized_entry = self._serialize_value(log_entry)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(serialized_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            # Fallback to stderr if file write fails
            print(f"Error writing log entry: {e}", file=__import__('sys').stderr)
    
    def flush(self):
        """Flush all logs to file"""
        if not self.log_file or not self.enabled:
            return
            
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                for log_entry in self.logs:
                    serialized_entry = self._serialize_value(log_entry)
                    f.write(json.dumps(serialized_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error flushing logs: {e}", file=__import__('sys').stderr)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of logged events"""
        if not self.enabled:
            return {}
            
        llm_calls = [log for log in self.logs if log.get("type") == "llm_call"]
        diagnostics = [log for log in self.logs if log.get("type") == "module_diagnostic"]
        workflow_events = [log for log in self.logs if log.get("type") == "workflow_event"]
        
        return {
            "total_logs": len(self.logs),
            "llm_calls": len(llm_calls),
            "diagnostics": len(diagnostics),
            "workflow_events": len(workflow_events),
            "modules_called": list(set(log.get("module", "unknown") for log in llm_calls))
        }

# Global logger instance
diagnostic_logger = DiagnosticLogger()

