import os
import json
import pandas as pd
from typing import Optional, Dict, List, Any

class InteractionTool:
    def __init__(self, data_dir: str, groundtruth_data: Optional[List[Dict]] = None):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
            groundtruth_data: List of groundtruth data for Track2 evaluation.
        """
        self.data_dir = data_dir
        self.groundtruth_data = groundtruth_data
        self.groundtruth_businesses = set()
        
        # 如果有groundtruth数据，提取需要过滤的business_ids
        if groundtruth_data and 'groundtruth' in groundtruth_data[0]:
            self.groundtruth_businesses = {
                item['groundtruth'] for item in groundtruth_data
            }
        
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
        """Fetch user data based on user_id or scenario."""
        self._ensure_scenario()
        
        user_id = user_id or self.scenario.get('user') if self.scenario else None
        if not user_id:
            return None
        
        user = self.user_data[self.user_data['user_id'] == user_id]
        if user.empty:
            return None
        user = user.to_dict(orient='records')[0]
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
        """Fetch reviews filtered by various parameters."""
        self._ensure_scenario()
        
        reviews = self.review_data

        if review_id:
            reviews = reviews[reviews['review_id'] == review_id]
        else:
            business_id = business_id or (self.scenario.get('business') if self.scenario else None)
            user_id = user_id or (self.scenario.get('user') if self.scenario else None)
            if business_id:
                reviews = reviews[reviews['business_id'] == business_id]
            if user_id:
                reviews = reviews[reviews['user_id'] == user_id]

        if 'date' in self.scenario:
            date_limit = self.scenario['date']
            reviews = self.review_data[
                (self.review_data['user_id'] == user_id) & 
                (self.review_data['date'] < date_limit)
            ]
        elif 'loc' in self.scenario:
            candidate_list = self.scenario.get('candidate_list', [])
            # 找到groundtruth中在candidate_list中的business_id
            groundtruth_business = next(
                (item['groundtruth'] for item in self.groundtruth_data 
                 if item['groundtruth'] in candidate_list),
                None
            )
            if groundtruth_business:
                reviews = reviews[reviews['business_id'] != groundtruth_business]
        return reviews.to_dict(orient='records')

    def get_tips(self, business_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict]:
        """Fetch tips with date filter."""
        self._ensure_scenario()
        business_id = business_id or (self.scenario.get('business') if self.scenario else None)
        user_id = user_id or (self.scenario.get('user') if self.scenario else None)
        tips = self.tip_data
        if business_id:
            tips = tips[tips['business_id'] == business_id]
        if user_id:
            tips = tips[tips['user_id'] == user_id]
        if self.scenario and 'date' in self.scenario:
            tips = tips[tips['date'] <= self.scenario['date']]
        return tips.to_dict(orient='records')

    def get_checkins(self, business_id: Optional[str] = None) -> List[Dict]:
        """Fetch checkins with date filter."""
        self._ensure_scenario()
        business_id = business_id or (self.scenario.get('business') if self.scenario else None)
        if not business_id:
            return []
        checkins = self.checkin_data
        checkins = checkins[checkins['business_id'] == business_id]
        if self.scenario and 'date' in self.scenario:
            checkins = checkins[checkins['date'] <= self.scenario['date']]
        return checkins.to_dict(orient='records')