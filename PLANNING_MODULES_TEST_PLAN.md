# Planning Modules Test Plan

## Overview
This document outlines the plan to test different planning modules with the TOTSimulationAgent and record their performance.

## Available Planning Modules

1. **PlanningBaseline** (Current)
   - Simple hardcoded plan: fetch user info, then business info
   - No LLM calls, deterministic
   - Baseline for comparison

2. **PlanningIO**
   - Input-Output planning approach
   - Uses LLM to generate subtasks with reasoning and tool instructions

3. **PlanningDEPS**
   - Dependency-based planning
   - Focuses on multi-hop questions and dependencies

4. **PlanningTD**
   - Temporal Dependencies planning
   - Explicitly considers order and dependencies of actions

5. **PlanningVoyager**
   - Voyager-style planning
   - Generates subgoals to complete the task

6. **PlanningOPENAGI**
   - OPENAGI-style planning
   - Creates concise todo lists

7. **PlanningHUGGINGGPT**
   - HuggingGPT-style planning
   - Minimizes number of tasks while ensuring completeness

## Test Strategy

### Phase 1: Setup
- [x] Create modified TOTSimulationAgent that accepts planning modules
- [x] Create adapter to make standard planning modules compatible
- [x] Create test script with result tracking

### Phase 2: Individual Module Testing
- Run each planning module separately with 30 tasks
- Record:
  - Preference Estimation score
  - Review Generation score
  - Overall Quality score
  - Execution time
  - Any errors or issues

### Phase 3: Comparison Analysis
- Generate comparison report
- Identify best performing module
- Analyze trade-offs (performance vs. complexity)

## Expected Challenges

1. **Interface Compatibility**
   - PlanningBaseline uses `__call__(task_description)` (dict)
   - Standard modules use `__call__(task_type, task_description, feedback, few_shot)`
   - **Solution**: Created PlanningAdapter wrapper

2. **Plan Format**
   - Need to ensure all modules return plans in expected format:
     ```python
     [
         {
             'description': '...',
             'reasoning instruction': '...',
             'tool use instruction': {user_id or item_id}
         },
         ...
     ]
     ```

3. **LLM Call Failures**
   - Some planning modules may fail to generate valid plans
   - **Solution**: Fallback to PlanningBaseline on errors

4. **Performance Variation**
   - Different modules may have different execution times
   - Some may require more LLM calls

## Success Metrics

- All modules can be tested without crashes
- Results are recorded consistently
- Comparison report is generated
- Best performing module is identified

## Usage

### Test all modules:
```bash
python test_planning_modules.py --module all --num-tasks 30
```

### Test specific module:
```bash
python test_planning_modules.py --module io --num-tasks 30
```

### Custom output directory:
```bash
python test_planning_modules.py --module all --output-dir my_results
```

## Results Location

Results will be saved in:
- `planning_test_results/` (or custom directory)
  - `{module_name}_results.json` - Evaluation results
  - `{module_name}_history.json` - Evaluation history
  - `{module_name}_llm_logs.jsonl` - LLM call logs
  - `comparison_report.md` - Comparison report

## Next Steps After Testing

1. Analyze results to identify best module
2. Consider hybrid approaches
3. Optimize best performing module
4. Document findings in ImprovementNotes.md

