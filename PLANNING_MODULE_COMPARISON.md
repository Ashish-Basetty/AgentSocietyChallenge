# Planning Module Comparison for Yelp User Simulation

## Goal

Optimize the agent for simulating a Yelp user who rates and writes reviews for restaurants. We're starting by experimenting with **planning modules** to find the most optimal one.

## Available Planning Modules

The codebase provides 6 different planning modules:

1. **PlanningHUGGINGGPT** - Step-by-step planning with attention to dependencies and order
2. **PlanningTD** - Explicit temporal dependencies and logical sequencing  
3. **PlanningOPENAGI** - Concise todo lists optimized for efficiency
4. **PlanningDEPS** - Multi-hop reasoning with sequences of sub-goals
5. **PlanningVoyager** - Subgoal generation for task completion
6. **PlanningIO** - Basic planning approach

## Testing Setup

I've created a comprehensive test script (`test_planning_modules.py`) that:

- Tests each planning module with the same set of tasks
- Runs tests in isolated subprocesses to avoid PyTorch state issues
- Generates actual evaluation metrics for comparison
- Creates a comparison table showing:
  - Preference Estimation (how well the rating matches ground truth)
  - Review Generation (quality of generated reviews)
  - Overall Quality (combined metric)

## Fixes Applied

1. **Increased max_tokens** in planning modules from default (500) to 2000 to handle longer planning outputs
2. **Fixed memory module** to handle `hnswlib` import errors gracefully (falls back to in-memory mode)
3. **Created TestSimulationAgent** that can use any planning module
4. **Subprocess isolation** ensures each test runs with clean PyTorch state

## How to Run

### Test all planning modules:
```bash
python test_planning_modules.py --module all --num-tasks 3
```

### Test a specific module:
```bash
python test_planning_modules.py --module OPENAGI --num-tasks 3
```

## Expected Output

The script will generate:
1. **Console output** with a comparison table showing actual values
2. **JSON files** in `results/` directory:
   - `planning_comparison.json` - Summary of all modules
   - `{module_name}_results.json` - Detailed results for each module

## Previous Results

Based on previous tests (from `results/planning_comparison.json`):
- **OPENAGI**: Overall Quality: 0.6122 (Best!)
- **HUGGINGGPT**: Overall Quality: 0.5979
- **TD**: Overall Quality: 0.5979

However, some modules (DEPS, Voyager, IO) failed due to token limits and memory issues, which have now been fixed.

## Next Steps

After running the comparison, we can:
1. Identify the optimal planning module
2. Experiment with other modules (reasoning, memory, tool use)
3. Fine-tune prompts for better performance


