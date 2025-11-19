from websocietysimulator import Simulator
from websocietysimulator.agent import TOTSimulationAgent, BaselineSimulationAgent
from websocietysimulator.llm import GeminiLLM
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
    args = parser.parse_args()
    output_path = args.output

    # Ensure the output directory exists if a directory is provided
    if os.path.isdir(output_path):
        results_file = os.path.join(output_path, "evaluation_results.json")
        history_file = os.path.join(output_path, "evaluation_history.json")
    else:
        # If it's a file prefix, append suffixes
        results_file = f"{output_path}_results.json"
        history_file = f"{output_path}_history.json"

    task_set = "yelp"
    simulator = Simulator(data_dir="dataset", device="auto", cache=False)

    simulator.set_task_and_groundtruth(
        task_dir=f"example/track1/{task_set}/tasks",
        groundtruth_dir=f"example/track1/{task_set}/groundtruth"
    )

    simulator.set_agent(TOTSimulationAgent)

    simulator.set_llm(GeminiLLM(api_key=os.getenv("GEMINI_API_KEY")))

    agent_outputs = simulator.run_simulation(number_of_tasks=30, enable_threading=True, max_workers=10)

    evaluation_results = simulator.evaluate()
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    evaluation_history = simulator.get_evaluation_history()
    with open(history_file, 'w') as f:
        json.dump(evaluation_history, f, indent=4)

    print(f"Results saved to {results_file} and {history_file}")

if __name__ == "__main__":
    main()