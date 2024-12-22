from typing import Dict, Any
from .agent import Agent

class SimulationAgent(Agent):
    def __init__(self):
        """
        SimulationAgent initialization.
        """
        super().__init__()
        self.task = None

    def insert_task(self, task):
        """
        Insert a simulation task.
        Args:
            task: An instance of SimulationTask.
        """
        if not task:
            raise ValueError("The task cannot be None.")
        self.task = task.to_dict()
        self.interaction_tool.set_task(self.task)

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