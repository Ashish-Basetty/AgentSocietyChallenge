from .scenario import Scenario
from typing import Dict

class SimulationScenario(Scenario):
    def __init__(self, time: str, user: str, business: str):
        """
        Simulation Scenario for the SimulationAgent.
        Args:
            time: The time parameter to limit InteractionTool behavior.
            user: The user being simulated.
            business: The business receiving the simulated review.
        """
        super().__init__(time)
        self.user = user
        self.business = business

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the scenario to a dictionary.
        Returns:
            dict: The scenario in dictionary format.
        """
        return {
            "time": self.time,
            "user": self.user,
            "business": self.business
        }