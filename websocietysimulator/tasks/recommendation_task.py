from typing import Any, Dict, List

class RecommendationTask:
    def __init__(self, user_id: str,  
                 candidate_list: List[str]):
        """
        Recommendation Task for the RecommendationAgent.
        Args:
            user_id: The ID of the user requesting recommendations.
            candidate_list: List of candidate business IDs.
            loc: User's location as [latitude, longitude].
        """
        self.user_id = user_id
        self.candidate_list = candidate_list

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        Returns:
            dict: The task in dictionary format.
        """
        return {
            "user_id": self.user_id,
            "candidate_list": self.candidate_list,
        }