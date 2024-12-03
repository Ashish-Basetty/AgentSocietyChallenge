from abc import ABC, abstractmethod
from typing import Any
from tools.interaction_tool import InteractionTool

class Agent(ABC):
    def __init__(self, data_dir: str):
        """
        Abstract base class for agents.
        Args:
            data_dir: Directory containing Yelp dataset files.
        """
        self.interaction_tool = InteractionTool(data_dir)

    @abstractmethod
    def insert_scenario(self, scenario):
        """Insert a scenario for the agent."""
        pass

    @abstractmethod
    def forward(self) -> Any:
        """Abstract forward method for evaluation."""
        pass