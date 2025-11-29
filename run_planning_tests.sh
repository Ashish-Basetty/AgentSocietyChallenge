#!/bin/bash
# Simple script to run planning module tests

cd "$(dirname "$0")"

echo "Starting Planning Modules Test..."
echo "================================"

# Test with baseline first (quick test)
echo "Testing baseline module (5 tasks)..."
python test_planning_modules.py --module baseline --num-tasks 5 --output-dir planning_test_results

# If successful, test all modules
if [ $? -eq 0 ]; then
    echo ""
    echo "Baseline test successful. Testing all modules (30 tasks each)..."
    python test_planning_modules.py --module all --num-tasks 30 --output-dir planning_test_results
else
    echo "Baseline test failed. Please check the error above."
    exit 1
fi

echo ""
echo "Tests completed! Check planning_test_results/comparison_report.md for results."

