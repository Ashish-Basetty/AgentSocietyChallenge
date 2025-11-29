# Planning Module Recommendations for Yelp User Simulation

## Overview

This document analyzes the available planning modules and recommends the **most optimal** one for Yelp user behavior simulation tasks.

## Available Planning Modules

### 1. **PlanningHUGGINGGPT** ⭐ **RECOMMENDED - BEST CHOICE**

**Why it's optimal:**
- ✅ **Explicitly considers dependencies and order** - Critical for sequential tasks
- ✅ **Minimizes tasks while ensuring completeness** - Efficient and focused
- ✅ **Step-by-step thinking** - Breaks down complex tasks systematically
- ✅ **Balances structure with flexibility** - Good for Yelp simulation workflow

**Best for:** Structured sequential tasks where order and dependencies matter (like Yelp user simulation)

**Prompt emphasizes:** "Think step by step about all the tasks needed... Pay attention to the dependencies and order among tasks."

---

### 2. **PlanningTD** ⭐⭐ **SECOND CHOICE**

**Why it's good:**
- ✅ **Explicit temporal dependencies** - Ensures logical sequencing
- ✅ **Order-focused** - Perfect for tasks with clear execution order
- ⚠️ **More verbose** - May generate more detailed plans than needed

**Best for:** Tasks with strict temporal ordering requirements

**Prompt emphasizes:** "Divides task into several subtasks with explicit temporal dependencies... Consider the order of actions"

---

### 3. **PlanningOPENAGI** ⭐⭐⭐ **THIRD CHOICE**

**Why it's good:**
- ✅ **Concise todo lists** - Very efficient
- ✅ **Short and relevant** - Minimizes unnecessary steps
- ⚠️ **May miss nuances** - Might oversimplify complex tasks

**Best for:** Simple, straightforward tasks that need quick planning

**Prompt emphasizes:** "Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence"

---

### 4. **PlanningDEPS**

**Characteristics:**
- Focuses on multi-hop reasoning
- Good for complex reasoning chains
- May be overkill for Yelp simulation

**Best for:** Complex multi-hop reasoning tasks

---

### 5. **PlanningVoyager**

**Characteristics:**
- Open-ended exploration
- Good for discovery tasks
- Less structured than needed for simulation

**Best for:** Exploration and discovery tasks

---

### 6. **PlanningIO**

**Characteristics:**
- Basic planning
- Simple and straightforward
- Less sophisticated than other options

**Best for:** Simple tasks requiring basic planning

---

## Yelp User Simulation Task Flow

The optimal planning module needs to handle this sequential workflow:

1. **Get User Information** → Understand user preferences and review history
2. **Get Business Information** → Understand restaurant characteristics
3. **Get Existing Reviews** → Understand context and common themes
4. **Analyze & Generate** → Create rating and review matching user's style

## Recommendation Summary

### 🏆 **Use PlanningHUGGINGGPT**

**Rationale:**
1. The Yelp simulation task is **structured and sequential** with clear dependencies
2. We need **efficient planning** that doesn't miss critical steps
3. **Order matters** - can't generate review without user and business data
4. PlanningHUGGINGGPT balances **structure, efficiency, and completeness** perfectly

## Implementation

I've created `optimized_simulation_agent.py` which uses PlanningHUGGINGGPT by default, but you can easily switch between modules.

### Usage:

```python
from optimized_simulation_agent import OptimizedSimulationAgent
from websocietysimulator.agent.modules.planning_modules import PlanningHUGGINGGPT

# Use default (PlanningHUGGINGGPT)
agent = OptimizedSimulationAgent(llm=your_llm)

# Or specify a different module
agent = OptimizedSimulationAgent(llm=your_llm, planning_module=PlanningTD)
```

### Testing Different Modules:

```bash
# Test a specific module
python test_planning_modules.py --module HUGGINGGPT --num-tasks 3

# Compare all modules
python test_planning_modules.py --module all --num-tasks 3
```

## Expected Improvements

Using PlanningHUGGINGGPT over the baseline should improve:

1. **Preference Estimation** - Better understanding of task dependencies leads to more accurate rating prediction
2. **Review Generation** - More structured planning results in reviews that better match user style
3. **Overall Quality** - Better planning = better overall simulation quality

## Next Steps

1. **Test PlanningHUGGINGGPT** with a small set of tasks
2. **Compare results** with your baseline agent
3. **Experiment with PlanningTD** if you need even more explicit temporal ordering
4. **Fine-tune the few-shot example** in `OptimizedPlanningWrapper` for your specific use case

