from websocietysimulator import Simulator
from example.ModelingAgent_baseline import MySimulationAgent
from websocietysimulator.llm import GeminiLLM
from google import genai
from dotenv import load_dotenv
import os
import json

def main():
    load_dotenv()

    # Initialize Simulator
    task_set = "yelp"

    simulator = Simulator(data_dir="dataset", device="auto", cache=False)
    # The cache parameter controls whether to use cache for interaction tool.
    # If you want to use cache, you can set cache=True. When using cache, the simulator will only load data into memory when it is needed, which saves a lot of memory.
    # If you want to use normal interaction tool, you can set cache=False. Notice that, normal interaction tool will load all data into memory at the beginning, which needs a lot of memory (20GB+).

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"example/track1/{task_set}/tasks", groundtruth_dir=f"example/track1/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(MySimulationAgent)

    # Set LLM client
    simulator.set_llm(GeminiLLM(api_key=os.getenv("GEMINI_API_KEY")))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=10, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
     
    with open(f'./evaluation_results_track1_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()

    with open(f'./evaluation_history_track1_{task_set}.json', 'w') as f:
        json.dump(evaluation_history, f, indent=4)

if __name__ == "__main__":
    main()