from typing import Dict, Any
from .agent import Agent

class SimulationAgent(Agent):
    def __init__(self):
        """
        SimulationAgent initialization.
        """
        super().__init__()
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

    def forward(self) -> Dict[str, Any]:
        """
        Abstract forward method for SimulationAgent.
        Participants must override this method to provide:
            - star (float): Simulated rating
            - review (str): Simulated review text
            - useful (int): Simulated useful count
            - cool (int): Simulated cool count
            - funny (int): Simulated funny count
        """
        result = {
            'star': 0,
            'review': '',
            'useful': 0,
            'cool': 0,
            'funny': 0
        }
        return result