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

    def _load_data(self, filename: str) -> pd.DataFrame:
        """Load a dataset as a Pandas DataFrame."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Fetch user data based on user_id."""  
        user = self.user_data[self.user_data['user_id'] == user_id]
        if user.empty:
            return None
        user = user.to_dict(orient='records')[0]
        return user

    def get_business(self, business_id: str) -> Optional[Dict]:
        """Fetch business data based on business_id."""
        business = self.business_data[self.business_data['business_id'] == business_id]
        return business.to_dict(orient='records')[0] if not business.empty else None

    def get_reviews(
        self, 
        business_id: Optional[str] = None, 
        user_id: Optional[str] = None, 
        review_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch reviews filtered by various parameters."""
        if business_id is None and user_id is None and review_id is None:
            return []
        reviews = self.review_data
        if review_id:
            reviews = reviews[reviews['review_id'] == review_id]
        else:
            if business_id:
                reviews = reviews[reviews['business_id'] == business_id]
            if user_id:
                reviews = reviews[reviews['user_id'] == user_id]
        return reviews.to_dict(orient='records')

    def get_tips(self, business_id: str, user_id: str) -> List[Dict]:
        """Fetch tips with date filter."""
        tips = self.tip_data
        if business_id:
            tips = tips[tips['business_id'] == business_id]
        if user_id:
            tips = tips[tips['user_id'] == user_id]
        return tips.to_dict(orient='records')

    def get_checkins(self, business_id: str) -> List[Dict]:
        """Fetch checkins with date filter."""
        checkins = self.checkin_data
        checkins = checkins[checkins['business_id'] == business_id]
        return checkins.to_dict(orient='records')
