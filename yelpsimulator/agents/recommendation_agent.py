from typing import List, Dict, Any
from agents.agent import Agent

class RecommendationAgent(Agent):
    def __init__(self, data_dir: str):
        """
        RecommendationAgent initialization.
        Args:
            data_dir: Directory containing Yelp dataset files.
        """
        super().__init__(data_dir)
        self.scenario = None

    def insert_scenario(self, scenario):
        """
        Insert a recommendation scenario.
        Args:
            scenario: An instance of RecommendationScenario.
        """
        if not scenario:
            raise ValueError("The scenario cannot be None.")
        self.scenario = scenario.to_dict()
        self.interaction_tool.set_scenario(self.scenario)

    def forward(self) -> List[Dict[str, Any]]:
        """
        Abstract forward method for RecommendationAgent.
        Participants must override this method to provide a sorted list of candidate POIs.
        Returns:
            list: A sorted list of candidate POIs (business dictionaries).
        """
        raise NotImplementedError("Forward method must be implemented by the participant.")