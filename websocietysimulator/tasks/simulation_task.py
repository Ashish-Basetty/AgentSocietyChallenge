from typing import Dict

class SimulationTask:
    def __init__(self, user_id: str, business_id: str):
        """
        Simulation Task for the SimulationAgent.
        Args:
            time: The time parameter to limit InteractionTool behavior.
            user: The user being simulated.
            business: The business receiving the simulated review.
        """
        self.user_id = user_id
        self.business_id = business_id

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the task to a dictionary.
        Returns:
            dict: The task in dictionary format.
        """
        return {
            "user_id": self.user_id,
            "business_id": self.business_id
        }