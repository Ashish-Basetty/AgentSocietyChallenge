# Reasoning Modules Comparison Report

Generated: 2025-11-29 12:22:01

## Overview

This report compares different reasoning modules tested with TOTSimulationAgent on Yelp user simulation tasks.
All tests use PlanningBaseline as the fixed planning module to isolate reasoning module performance.

## Results Summary

### Successful Modules (Ranked by Overall Quality)

| Rank | Module | Preference Estimation | Review Generation | Overall Quality | Execution Time |
|------|--------|----------------------|-------------------|-----------------|----------------|
| 1 | TOT | 0.6000 | 0.5958 | 0.5979 | 1.5s |
| 2 | IO | 0.6000 | 0.5958 | 0.5979 | 2.5s |
| 3 | COT | 0.6000 | 0.5958 | 0.5979 | 2.6s |
| 4 | COTSC | 0.6000 | 0.5958 | 0.5979 | 1.8s |
| 5 | DILU | 0.6000 | 0.5958 | 0.5979 | 1.7s |
| 6 | SelfRefine | 0.6000 | 0.5958 | 0.5979 | 2.0s |
| 7 | StepBack | 0.6000 | 0.5958 | 0.5979 | 1.6s |

### 🏆 Best Performing Module: TOT

- **Overall Quality**: 0.5979
- **Preference Estimation**: 0.6000
- **Review Generation**: 0.5958

## Module Descriptions

- **TOT** (Tree of Thoughts): Current baseline, generates multiple reasoning candidates and votes on best
- **IO** (Input-Output): Direct input-output reasoning with examples
- **COT** (Chain of Thought): Step-by-step reasoning process
- **COTSC** (COT Self-Consistency): Multiple COT outputs, selects most common
- **DILU** (DILU-style): Uses system messages with examples
- **SelfRefine**: Self-refinement with error checking and revision
- **StepBack**: Step-back reasoning that extracts principles first

## Detailed Results

See individual result files:
- `tot_results.json`
- `io_results.json`
- `cot_results.json`
- `cotsc_results.json`
- `dilu_results.json`
- `selfrefine_results.json`
- `stepback_results.json`
