from websocietysimulator import Simulator
from example.ModelingAgent_baseline import MySimulationAgent
from websocietysimulator.llm import GeminiLLM
from google import genai
from dotenv import load_dotenv
import os

def main():
    load_dotenv()

    # Initialize Simulator
    simulator = Simulator(data_dir="dataset", device="auto", cache=False)
    # The cache parameter controls whether to use cache for interaction tool.
    # If you want to use cache, you can set cache=True. When using cache, the simulator will only load data into memory when it is needed, which saves a lot of memory.
    # If you want to use normal interaction tool, you can set cache=False. Notice that, normal interaction tool will load all data into memory at the beginning, which needs a lot of memory (20GB+).

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir="example/track1/yelp/tasks", groundtruth_dir="example/track1/yelp/groundtruth")

    # Set your custom agent
    simulator.set_agent(MySimulationAgent)

    # Set LLM client
    simulator.set_llm(GeminiLLM(api_key=os.getenv("GEMINI_API_KEY")))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()

    print(f"The evaluation_results are :{evaluation_results}")

if __name__ == "__main__":
    main()