from abc import ABC, abstractmethod
from typing import Any, Dict

class Scenario(ABC):
    def __init__(self, time: str):
        """
        Base Scenario class.
        Args:
            time: The time parameter for the scenario.
        """
        self.time = time

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the scenario to a dictionary."""
        pass