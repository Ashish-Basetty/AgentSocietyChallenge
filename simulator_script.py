from websocietysimulator import Simulator
from websocietysimulator.agent import TOTSimulationAgent, BaselineSimulationAgent
from websocietysimulator.llm import GeminiLLM
from websocietysimulator.utils import LLMLogger
from google import genai
from dotenv import load_dotenv
import os
import json
import argparse

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run simulation and save results.")
    parser.add_argument(
        "--output",
        type=str,
        default="./",
        help="Directory or file prefix for saving results (default: current directory)"
    )
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        help="Disable LLM call and diagnostic logging (default: logging is enabled)"
    )
    args = parser.parse_args()
    output_path = args.output

    # Determine if logging should be enabled (enabled by default unless --disable-logging is set)
    enable_logging = not args.disable_logging

    # Ensure the output directory exists if a directory is provided
    if os.path.isdir(output_path):
        results_file = os.path.join(output_path, "evaluation_results.json")
        history_file = os.path.join(output_path, "evaluation_history.json")
        log_file = os.path.join(output_path, "llm_logs.jsonl")
    else:
        # If it's a file prefix, append suffixes
        results_file = f"{output_path}_results.json"
        history_file = f"{output_path}_history.json"
        log_file = f"{output_path}_llm_logs.jsonl"
    
    # Initialize logger
    logger = LLMLogger.get_instance(
        log_file_path=log_file if enable_logging else None,
        enabled=enable_logging
    )
    
    if enable_logging:
        print(f"Logging enabled. Logs will be saved to {log_file}")
    else:
        print("Logging disabled.")

    task_set = "yelp"
    simulator = Simulator(data_dir="dataset", device="auto", cache=False)

    simulator.set_task_and_groundtruth(
        task_dir=f"example/track1/{task_set}/tasks",
        groundtruth_dir=f"example/track1/{task_set}/groundtruth"
    )

    simulator.set_agent(BaselineSimulationAgent)

    # Create LLM with logger
    llm = GeminiLLM(api_key=os.getenv("GEMINI_API_KEY"), logger=logger)
    simulator.set_llm(llm)
    
    # Set logger in simulator for passing to agents
    simulator.logger = logger

    agent_outputs = simulator.run_simulation(number_of_tasks=3, enable_threading=True, max_workers=10)

    evaluation_results = simulator.evaluate()
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    evaluation_history = simulator.get_evaluation_history()
    with open(history_file, 'w') as f:
        json.dump(evaluation_history, f, indent=4)

    print(f"Results saved to {results_file} and {history_file}")
    if enable_logging:
        print(f"LLM logs saved to {log_file}")

if __name__ == "__main__":
    main()