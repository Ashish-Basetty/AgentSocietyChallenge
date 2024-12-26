# Agent Development Guide

## 1. Overview of Agent Development

### 1.1 Track-Specific Agent Inheritance

To develop an agent, first inherit from the appropriate base class depending on your track:

- For Simulation Track: Inherit from `websocietysimulator.agent.SimulationAgent`
- For Recommendation Track: Inherit from `websocietysimulator.agent.RecommendationAgent`

### 1.2 Basic Attributes and Tools of Agents for both tracks
- `self.task`: The task (dict) to be processed (e.g., SimulationTask or RecommendationTask).
    - for SimulationTask, it contains `description`, `user_id` and `item_id`.
        - `description`: The description of the task.
        - `user_id`: The id of the user you are simulating.
        - `item_id`: The id of the item that the user is reviewing.
    - for RecommendationTask, it contains `description`, `user_id`, `candidate_category`, `candidate_list` and `loc`.
        - `description`: The description of the task.
        - `user_id`: The id of the user you are recommending items to.
        - `candidate_category`: The category of the candidate items.
        - `candidate_list`: The list of candidate item ids.
        - `loc`: The location of the user. If is [-1, -1], the user is not in a specific location (actually only for yelp dataset).
- `self.interaction_tool`: The interaction tool to interact with the environment. Supporting:
    - `get_user(user_id: Optional[str] = None) -> Optional[Dict]`: Get the user data based on user_id or task (if user_id is None).
    - `get_item(item_id: Optional[str] = None) -> Optional[Dict]`: Get the item data based on item_id or task (if item_id is None).
    - `get_reviews(item_id: Optional[str] = None, user_id: Optional[str] = None, review_id: Optional[str] = None) -> List[Dict]`: Get the reviews data based on item_id, user_id or review_id.
        - if provide `item_id`, return all reviews of the item.
        - if provide `user_id`, return all reviews of the user.
        - if provide `review_id`, return the review of the id.
- `self.llm`: The LLM to generate reasoning.

### 1.3 Implementing the Forward Method

The key step is to override the `forward()` method in your agent class. This method contains your agent's core logic.

### 1.4 Track-Specific Return Values

Different tracks require different return values from the `forward()` method:

**Simulation Track**
```python
def forward(self) -> Dict[str, Any]:
    # Must return a dictionary with:
    return {
        'stars': float,  # Rating (1.0-5.0)
        'review': str,  # Review text
    }
```

**Recommendation Track**
```python
def forward(self) -> List[Dict[str, Any]]:
    # Must return a sorted list of POI dictionaries
    return sorted_candidate_list
```

### 1.5 Example Implementations
Example implementations for both tracks can be found in the `example` folder:

- Simulation Track: `example/userBehaviorSimulation.py`
- Recommendation Track: `example/recommendationExample.py`


## 2. LLM Client Integration

### 2.1 Available LLM Clients

The framework provides a base LLM class and two implementations:

```python
# Base LLM class
class LLMBase:
    def __call__(self, messages: List[Dict[str, str]], 
                 temperature: float = 0.0,
                 max_tokens: int = 500) -> str:
        pass

# Available implementations
class DeepseekLLM(LLMBase):
    # Deepseek API implementation
    pass

class OpenAILLM(LLMBase):
    # OpenAI API implementation
    pass
```

### 2.2 Custom LLM Implementation

You can implement your own LLM client by inheriting from `LLMBase`. Note that during evaluation, we will use a standardized LLM client to ensure fair comparison.

Example:
```python
class CustomLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "custom-model"):
        super().__init__(model)
        self.client = CustomAPIClient(api_key)
        
    def __call__(self, messages, temperature=0.0, max_tokens=500):
        # Implement your LLM call logic here
        return response_text
```

## 3. Agent Modules Documentation
We provide several standardized modules to accelerate development, which are included in `websocietysimulator.agent.modules`. This repository contains four core modules for building intelligent agents: Reasoning, Memory, Planning and ToolUse. Each module is designed to handle specific aspects of agent behavior and decision making.

### 3.1 Reasoning Module

The Reasoning module processes subtasks sequentially, where each subtask and optional feedback are provided as input. The module produces solutions for individual stages, enabling systematic problem-solving across multi-step tasks.

#### Overview

The module consists of multiple implementations:
1. **ReasoningBase**: Base class handling task processing and memory management
2. **ReasoningIO**[1]
3. **ReasoningCOT**[2]
4. **ReasoningCOTSC**[3]
5. **ReasoningTOT**[4]
6. **ReasoningSelfRefine**[5]
7. **ReasoningStepBack**[6]
8. **ReasoningSelfReflectiveTOT**[4, 6]
9. **ReasoningDILU**[7]

#### Interface

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

### 3.2 Memory Module 

The Memory module provides dynamic storage and retrieval of an agent's past experiences, enabling context-aware reasoning. It systematically logs and retrieves relevant memories to support informed decision making.

#### Overview

The module includes multiple implementations:
1. **MemoryBase**: Base class for memory management
2. **MemoryDILU**[7]
3. **MemoryGenerative**[8]
4. **MemoryTP**[9]
5. **MemoryVoyager**[10]

#### Interface

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

### 3.3 Planning Module

The Planning module decomposes complex tasks into manageable subtasks. It takes high-level task descriptions and generates structured sequences of subtasks with specific reasoning and tool-use instructions.

#### Overview
The module includes multiple implementations:
1. **PlanningBase**: Base planning functionality
2. **PlanningIO**
3. **PlanningDEPS**[11]
4. **PlanningVoyager**[10]
5. **PlanningOPENAGI**[12]
6. **PlanningHUGGINGGPT**[13]

#### Interface

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

### 3.4 ToolUse Module

The ToolUse module enables effective use of external tools to overcome LLM knowledge limitations. During reasoning, it selects optimal tools from a predefined pool to address specific problems.

#### Overview

The module includes multiple implementations:
1. **ToolUseBase**: Base tool selection functionality
2. **ToolUseIO**
3. **ToolUseAnyTool**[14]
4. **ToolUseToolBench**[15]
5. **ToolUseToolFormer**[16]

#### Interface

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

## References:
[1] Kojima et al. (2022). Zero-Shot Reasoning with Large Language Models. arXiv:2205.11916
[2] Wei et al. (2022). Chain of Thought Prompting Elicits Reasoning in Large Language Models. arXiv:2201.11903
[3] Wang et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171
[4] Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arXiv:2305.10601
[5] Zhang et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. arXiv:2303.17651
[6] Zheng et al. (2023). Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models. arXiv:2310.06117
[7] Wen et al. (2023). DILU: A Knowledge-Driven Approach to Turn LLMs into Intelligent Agents. arXiv:2310.09819
[8] Park et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. arXiv:2304.03442
[9] Yu et al. (2023). Thought Propagation: An Analogical Approach to Complex Reasoning with Large Language Models. arXiv:2310.03965
[10] Wang et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv:2305.16291
[11] Xu et al. (2023). DEPS: A Framework for Dependency-based Planning with LLMs. arXiv:2305.16291
[12] Wang et al. (2023). OpenAGI: When LLM Meets Domain Experts. arXiv:2304.04370
[13] Shen et al. (2023). HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face. arXiv:2303.17580
[14] Qin et al. (2023). AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls. arXiv:2308.10848
[15] Qin et al. (2023). ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arXiv:2307.16789
[16] Schick et al. (2023). ToolFormer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761