# Agent Modules Documentation

This repository contains four core modules for building intelligent agents: Reasoning, Memory, Planning and ToolUse. Each module is designed to handle specific aspects of agent behavior and decision making.

## Reasoning Module

The Reasoning module processes subtasks sequentially, where each subtask and optional feedback are provided as input. The module produces solutions for individual stages, enabling systematic problem-solving across multi-step tasks.

### Overview

The module consists of multiple implementations:
1. **ReasoningBase**: Base class handling task processing and memory management
2. **ReasoningIO**: Basic reasoning implementation
3. **ReasoningCOT**: Chain-of-thought reasoning 
4. **ReasoningSelfRefine**: Self-refining reasoning
5. **ReasoningStepBack**: Step-back reasoning approach
6. **ReasoningSelfReflectiveTOT**: Self-reflective tree of thoughts
7. **ReasoningDILU**: DILU-based reasoning

### Interface

```python
class ReasoningBase:
    def __init__(self, profile_type_prompt: str, memory, llm):
        """
        Initialize reasoning base class
        
        Args:
            profile_type_prompt: Role-playing prompt for LLM
            memory: Memory module instance
            llm: LLM instance for generating reasoning
        """

    def __call__(self, task_description: str, feedback: str = ''):
        """
        Process task and generate reasoning
        
        Args:
            task_description: Description of task to process
            feedback: Optional feedback to refine reasoning
            
        Returns:
            str: Reasoning result for current step
        """
```

## Memory Module 

The Memory module provides dynamic storage and retrieval of an agent's past experiences, enabling context-aware reasoning. It systematically logs and retrieves relevant memories to support informed decision making.

### Overview

The module includes multiple implementations:
1. **MemoryBase**: Base class for memory management
2. **MemoryDILU**: DILU-based memory implementation
3. **MemoryGenerative**: Generative memory approach
4. **MemoryTP**: Task planning memory
5. **MemoryVoyager**: Voyager-style memory

### Interface

```python
class MemoryBase:
    def __init__(self, memory_type: str, llm):
        """
        Initialize memory base class
        
        Args:
            memory_type: Type of memory implementation
            llm: LLM instance for memory operations
        """

    def __call__(self, current_situation: str = ''):
        """
        Process current situation
        
        Args:
            current_situation: Current task state and trajectory
            
        Returns:
            str: Updated or retrieved memory based on situation
        """
```

## Planning Module

The Planning module decomposes complex tasks into manageable subtasks. It takes high-level task descriptions and generates structured sequences of subtasks with specific reasoning and tool-use instructions.

### Overview

The module includes multiple implementations:
1. **PlanningBase**: Base planning functionality
2. **PlanningIO**: Basic planning implementation
3. **PlanningDEPS**: Dependency-based planning
4. **PlanningTD**: Temporal dependency planning
5. **PlanningVoyager**: Voyager-style planning
6. **PlanningOPENAGI**: OpenAGI planning approach
7. **PlanningHUGGINGGPT**: HuggingGPT-style planning

### Interface

```python
class PlanningBase:
    def __init__(self, llm):
        """
        Initialize planning base class
        
        Args:
            llm: LLM instance for generating plans
        """
    
    def __call__(self, task_type: str, task_description: str, feedback: str = ''):
        """
        Generate task decomposition plan
        
        Args:
            task_type: Type of task
            task_description: Detailed task description
            feedback: Optional feedback to refine planning
            
        Returns:
            list: List of subtask dictionaries containing descriptions and instructions
        """
```

## ToolUse Module

The ToolUse module enables effective use of external tools to overcome LLM knowledge limitations. During reasoning, it selects optimal tools from a predefined pool to address specific problems.

### Overview

The module includes multiple implementations:
1. **ToolUseBase**: Base tool selection functionality
2. **ToolUseIO**: Basic tool use implementation
3. **ToolUseAnyTool**: Flexible tool selection
4. **ToolUseToolBench**: ToolBench-based approach
5. **ToolUseToolBenchFormer**: Enhanced ToolBench implementation

### Interface

```python
class ToolUseBase:
    def __init__(self, llm):
        """
        Initialize tool use base class
        
        Args:
            llm: LLM instance for tool selection
        """

    def __call__(self, task_description: str, tool_instruction: str, feedback_of_previous_tools: str = ''):
        """
        Select and use appropriate tools
        
        Args:
            task_description: Task description
            tool_instruction: Tool selection guidance
            feedback_of_previous_tools: Optional feedback on previous tool usage
            
        Returns:
            str: Tool use result
        """
```
