import os
import json
import pandas as pd
from typing import Optional, Dict, List, Any

class InteractionTool:
    def __init__(self, data_dir: str):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
        """
        self.data_dir = data_dir
        self.business_data = self._load_data('business.json')
        self.review_data = self._load_data('review.json')
        self.user_data = self._load_data('user.json')
        self.tip_data = self._load_data('tip.json')
        self.checkin_data = self._load_data('checkin.json')
        self.scenario = None

    def _load_data(self, filename: str) -> pd.DataFrame:
        """Load a dataset as a Pandas DataFrame."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)

    def set_scenario(self, scenario: Dict[str, Any]):
        """
        Update the context of the tool based on a scenario.
        Args:
            scenario: Scenario dictionary with context parameters.
        """
        self.scenario = scenario

    def _ensure_scenario(self):
        """Ensure that a scenario has been set before any action."""
        if not self.scenario:
            raise RuntimeError("No scenario has been set. Please set a scenario before interacting.")

    def get_user(self, user_id: Optional[str] = None) -> Optional[Dict]:
        """Fetch user data based on user_id or scenario with time-limited reviews."""
        self._ensure_scenario()  # Ensure scenario is set
        
        # Determine user_id from parameter or scenario
        user_id = user_id or self.scenario.get('user') if self.scenario else None
        if not user_id:
            return None
        
        # Fetch the user entity
        user = self.user_data[self.user_data['user_id'] == user_id]
        if user.empty:
            return None
        user = user.to_dict(orient='records')[0]
        
        # Filter reviews based on the time limitation
        if 'time' in self.scenario: # type: ignore
            time_limit = self.scenario['time'] # type: ignore
            reviews = self.review_data[
                (self.review_data['user_id'] == user_id) & (self.review_data['date'] < time_limit)
            ]
            user['reviews'] = reviews.to_dict(orient='records')
        else:
            # Include all reviews if no time constraint exists
            reviews = self.review_data[self.review_data['user_id'] == user_id]
            user['reviews'] = reviews.to_dict(orient='records')
        
        return user

    def get_business(self, business_id: Optional[str] = None) -> Optional[Dict]:
        """Fetch business data based on business_id or scenario."""
        self._ensure_scenario()  # Ensure scenario is set
        business_id = business_id or self.scenario.get('business') if self.scenario else None
        if not business_id:
            return None
        business = self.business_data[self.business_data['business_id'] == business_id]
        return business.to_dict(orient='records')[0] if not business.empty else None

    def get_reviews(
        self, 
        business_id: Optional[str] = None, 
        user_id: Optional[str] = None, 
        review_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch reviews limited by time or filter by review_id, business_id, or user_id.
        """
        self._ensure_scenario()  # Ensure scenario is set
        
        # Start with all reviews
        reviews = self.review_data

        # Filter by review_id if provided
        if review_id:
            reviews = reviews[reviews['review_id'] == review_id]
        else:
            # Filter by business_id and user_id
            business_id = business_id or (self.scenario.get('business') if self.scenario else None)
            user_id = user_id or (self.scenario.get('user') if self.scenario else None)
            if business_id:
                reviews = reviews[reviews['business_id'] == business_id]
            if user_id:
                reviews = reviews[reviews['user_id'] == user_id]
        
        # Apply time constraint from the scenario
        if self.scenario and 'time' in self.scenario:
            reviews = reviews[reviews['date'] < self.scenario['time']]
        
        return reviews.to_dict(orient='records')

    def get_tips(self, business_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict]:
        """Fetch tips limited by time."""
        self._ensure_scenario()  # Ensure scenario is set
        business_id = business_id or (self.scenario.get('business') if self.scenario else None)
        user_id = user_id or (self.scenario.get('user') if self.scenario else None)
        tips = self.tip_data
        if business_id:
            tips = tips[tips['business_id'] == business_id]
        if user_id:
            tips = tips[tips['user_id'] == user_id]
        if self.scenario and 'time' in self.scenario:
            tips = tips[tips['date'] < self.scenario['time']]
        return tips.to_dict(orient='records')

    def get_checkins(self, business_id: Optional[str] = None) -> List[Dict]:
        """Fetch checkins limited by time."""
        self._ensure_scenario()  # Ensure scenario is set
        business_id = business_id or (self.scenario.get('business') if self.scenario else None)
        if not business_id:
            return []
        checkins = self.checkin_data
        checkins = checkins[checkins['business_id'] == business_id]
        if self.scenario and 'time' in self.scenario:
            checkins = checkins[checkins['date'] < self.scenario['time']]
        return checkins.to_dict(orient='records')