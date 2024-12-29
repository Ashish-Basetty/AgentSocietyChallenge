import json
from websocietysimulator import Simulator
from websocietysimulator.agent.recommendation_agent import RecommendationAgent
from typing import List, Dict, Any

from websocietysimulator.llm import InfinigenceLLM

import logging
logging.basicConfig(level=logging.INFO)


class CustomRecommendationAgent(RecommendationAgent):
    """
    Example of a custom RecommendationAgent for the Recommendation Track.
    """

    def workflow(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations by sorting candidate POIs based on their ratings.
        Returns:
            List[Dict[str, Any]]: A sorted list of candidate POIs (business dictionaries).
        """
        # Access task data
        if not self.task:
            raise RuntimeError("No task has been set. Please call insert_task before forward.")
        candidate_list = self.task['candidate_list']

        return candidate_list


if __name__ == "__main__":
    task_set = "amazon" # "goodreads" or "yelp"
    # Set the data
    simulator = Simulator(data_dir="your data dir", device="auto")
    simulator.set_task_and_groundtruth(task_dir=f"./track2/{task_set}/tasks", groundtruth_dir=f"./track2/{task_set}/groundtruth")

    # Set the agent and LLM
    simulator.set_agent(CustomRecommendationAgent)
    simulator.set_llm(InfinigenceLLM(api_key="Your API Key"))

    # Run the simulation
    outputs = simulator.run_simulation()

    # Evaluate the agent
    evaluation_results = simulator.evaluate()       
    with open(f'./evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    # 获取评估历史
    evaluation_history = simulator.get_evaluation_history()