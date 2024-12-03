from typing import Tuple
from agents.agent import Agent

class SimulationAgent(Agent):
    def __init__(self, data_dir: str):
        """
        SimulationAgent initialization.
        Args:
            data_dir: Directory containing Yelp dataset files.
        """
        super().__init__(data_dir)
        self.scenario = None

    def insert_scenario(self, scenario):
        """
        Insert a simulation scenario.
        Args:
            scenario: An instance of SimulationScenario.
        """
        if not scenario:
            raise ValueError("The scenario cannot be None.")
        self.scenario = scenario.to_dict()
        self.interaction_tool.set_scenario(self.scenario)

    def forward(self) -> Tuple[float, str, Tuple[int, int, int]]:
        """
        Abstract forward method for SimulationAgent.
        Participants must override this method to provide:
            - star (float): Simulated rating
            - review_text (str): Simulated review text
            - behavior_metrics (tuple): Simulated behavior metrics (int, int, int)
        """
        raise NotImplementedError("Forward method must be implemented by the participant.")