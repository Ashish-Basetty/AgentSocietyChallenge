from websocietysimulator import Simulator
from websocietysimulator.agents.recommendation_agent import RecommendationAgent
from typing import List, Dict, Any


class CustomRecommendationAgent(RecommendationAgent):
    """
    Example of a custom RecommendationAgent for the Recommendation Track.
    """

    def forward(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations by sorting candidate POIs based on their ratings.
        Returns:
            List[Dict[str, Any]]: A sorted list of candidate POIs (business dictionaries).
        """
        # Access scenario data
        if not self.scenario:
            raise RuntimeError("No scenario has been set. Please call insert_scenario before forward.")
        candidate_poi = self.scenario['candidate_poi']

        # Example sorting logic: sort businesses by "stars" (highest rating first)
        sorted_poi = sorted(candidate_poi, key=lambda poi: poi.get('stars', 0), reverse=True)

        return sorted_poi


if __name__ == "__main__":
    # Initialize the simulator with the dataset directory
    simulator = Simulator(data_dir="path/to/dataset", groundtruth_file="path/to/groundtruth")

    # Set scenarios (e.g., load predefined recommendation scenarios)
    simulator.set_scenario(scenario_dir="./recommendation_scenarios")

    # Use the custom agent for recommendation
    simulator.set_agent(CustomRecommendationAgent)

    # 运行模拟
    outputs = simulator.run_simulation()

    # 评估结果
    evaluation_results = simulator.evaluate()

    # 获取评估历史
    evaluation_history = simulator.get_evaluation_history()