from yelpsimulator import Simulator
from yelpsimulator.agents.recommendation_agent import RecommendationAgent
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
    simulator = Simulator(data_dir="path/to/dataset")

    # Set scenarios (e.g., load predefined recommendation scenarios)
    simulator.set_scenario(scenario_dir="./recommendation_scenarios")

    # Use the custom agent for recommendation
    simulator.set_agent(CustomRecommendationAgent)

    # Run the simulation and collect results
    results = simulator.run_simulation()

    # Output the results
    for result in results:
        scenario = result['scenario']
        output = result.get('output', result.get('error'))
        print(f"Scenario: {scenario}")
        print(f"Recommendation Output: {output}")