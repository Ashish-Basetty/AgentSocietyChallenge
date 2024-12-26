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
        self.task = None

    def _load_data(self, filename: str) -> pd.DataFrame:
        """Load a dataset as a Pandas DataFrame."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)

    def set_task(self, task: Dict[str, Any]):
        """
        Update the context of the tool based on a task.
        Args:
            task: Task dictionary with context parameters.
        """
        self.task = task

    def _ensure_task(self):
        """Ensure that a task has been set before any action."""
        if not self.task:
            raise RuntimeError("No task has been set. Please set a task before interacting.")

    def get_user(self, user_id: Optional[str] = None) -> Optional[Dict]:
        """Fetch user data based on user_id or task."""
        self._ensure_task()
        
        user_id = user_id or self.task.get('user_id') if self.task else None
        if not user_id:
            return None
        
        user = self.user_data[self.user_data['user_id'] == user_id]
        if user.empty:
            return None
        user = user.to_dict(orient='records')[0]
        return user

    def get_business(self, business_id: Optional[str] = None) -> Optional[Dict]:
        """Fetch business data based on business_id or task."""
        self._ensure_task()  # Ensure task is set
        business_id = business_id or self.task.get('business_id') if self.task else None
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
        self._ensure_task()
        
        reviews = self.review_data

        if review_id:
            reviews = reviews[reviews['review_id'] == review_id]
        else:
            business_id = business_id or (self.task.get('business_id') if self.task else None)
            user_id = user_id or (self.task.get('user_id') if self.task else None)
            if business_id:
                reviews = reviews[reviews['business_id'] == business_id]
            if user_id:
                reviews = reviews[reviews['user_id'] == user_id]
        return reviews.to_dict(orient='records')

    def get_tips(self, business_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict]:
        """Fetch tips with date filter."""
        self._ensure_task()
        business_id = business_id or (self.task.get('business') if self.task else None)
        user_id = user_id or (self.task.get('user') if self.task else None)
        tips = self.tip_data
        if business_id:
            tips = tips[tips['business_id'] == business_id]
        if user_id:
            tips = tips[tips['user_id'] == user_id]
        return tips.to_dict(orient='records')

    def get_checkins(self, business_id: Optional[str] = None) -> List[Dict]:
        """Fetch checkins with date filter."""
        self._ensure_task()
        business_id = business_id or (self.task.get('business') if self.task else None)
        if not business_id:
            return []   
        checkins = self.checkin_data
        checkins = checkins[checkins['business_id'] == business_id]
        return checkins.to_dict(orient='records')
