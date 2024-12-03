from scenarios.scenario import Scenario
from typing import Any, Dict, List

class RecommendationScenario(Scenario):
    def __init__(self, time: str, context: Any, candidate_poi: List[Dict[str, Any]]):
        """
        Recommendation Scenario for the RecommendationAgent.
        Args:
            time: The time parameter to limit InteractionTool behavior.
            context: Contextual information for recommendations.
            candidate_poi: List of candidate points of interest (businesses).
        """
        super().__init__(time)
        self.context = context
        self.candidate_poi = candidate_poi

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the scenario to a dictionary.
        Returns:
            dict: The scenario in dictionary format.
        """
        return {
            "time": self.time,
            "context": self.context,
            "candidate_poi": self.candidate_poi
        }