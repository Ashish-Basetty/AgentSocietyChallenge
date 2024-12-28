from typing import Dict, Any
from .agent import Agent
from ..llm import LLMBase
class SimulationAgent(Agent):
    def __init__(self, llm: LLMBase):
        """
        SimulationAgent initialization.
        """
        super().__init__(llm=llm)
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

    def forward(self) -> Dict[str, Any]:
        """
        Abstract forward method for SimulationAgent.
        Participants must override this method to provide:
            - stars (float): Simulated rating
            - review (str): Simulated review text
            - useful (int): Simulated useful count
            - cool (int): Simulated cool count
            - funny (int): Simulated funny count
        """
        result = {
            'stars': 0,
            'review': '',
            'useful': 0,
            'funny': 0,
            'cool': 0
        }
        return result