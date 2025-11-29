# Planning Modules Testing Guide

## Quick Start

### Test All Planning Modules
```bash
python test_planning_modules.py --module all --num-tasks 30
```

### Test a Specific Module
```bash
python test_planning_modules.py --module io --num-tasks 30
```

### Custom Output Directory
```bash
python test_planning_modules.py --module all --output-dir my_test_results
```

## What This Does

1. **Tests 7 different planning modules:**
   - `baseline` - Simple hardcoded plan (current)
   - `io` - Input-Output planning
   - `deps` - Dependency-based planning
   - `td` - Temporal Dependencies planning
   - `voyager` - Voyager-style planning
   - `openagi` - OPENAGI-style planning
   - `hugginggpt` - HuggingGPT-style planning

2. **Records metrics for each:**
   - Preference Estimation score
   - Review Generation score
   - Overall Quality score
   - Execution time
   - Error logs (if any)

3. **Generates comparison report:**
   - Markdown table comparing all modules
   - Detailed results for each module
   - Saved in `comparison_report.md`

## Output Structure

```
planning_test_results/
├── baseline_results.json
├── baseline_history.json
├── baseline_llm_logs.jsonl
├── io_results.json
├── io_history.json
├── io_llm_logs.jsonl
├── ... (for each module)
└── comparison_report.md
```

## Understanding Results

### Metrics Explained

- **Preference Estimation**: How well the agent estimates user preferences (0-1, higher is better)
- **Review Generation**: Quality of generated reviews (0-1, higher is better)
- **Overall Quality**: Combined metric (0-1, higher is better)

### Interpreting the Comparison

The comparison report shows:
- Which module performs best on each metric
- Trade-offs between modules
- Execution time differences
- Any errors or failures

## Troubleshooting

### Module Fails to Generate Plan
- The adapter falls back to PlanningBaseline if a module fails
- Check the LLM logs for specific errors
- Some modules may need better few-shot examples

### Low Performance
- Some modules may need tuning
- Check if the generated plans make sense
- Review the LLM logs to see what plans were generated

### Out of Memory
- Reduce `--num-tasks` if testing with limited resources
- Use `cache=True` in simulator (modify test script)

## Next Steps

After testing:
1. Review `comparison_report.md`
2. Identify best performing module
3. Consider optimizing the best module
4. Update `ImprovementNotes.md` with findings

