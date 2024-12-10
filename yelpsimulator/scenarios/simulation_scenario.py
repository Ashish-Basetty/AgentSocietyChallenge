from typing import Dict

class SimulationScenario():
    def __init__(self, date: str, user_id: str, business_id: str):
        """
        Simulation Scenario for the SimulationAgent.
        Args:
            time: The time parameter to limit InteractionTool behavior.
            user: The user being simulated.
            business: The business receiving the simulated review.
        """
        self.date = date
        self.user_id = user_id
        self.business_id = business_id

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the scenario to a dictionary.
        Returns:
            dict: The scenario in dictionary format.
        """
        return {
            "date": self.date,
            "user_id": self.user_id,
            "business_id": self.business_id
        }