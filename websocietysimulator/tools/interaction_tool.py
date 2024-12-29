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
        self.item_data = self._load_data('item.json')
        self.review_data = self._load_data('review.json')
        self.user_data = self._load_data('user.json')

    def _load_data(self, filename: str) -> pd.DataFrame:
        """Load a dataset as a Pandas DataFrame."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Fetch user data based on user_id."""  
        user = self.user_data[self.user_data['user_id'] == user_id]
        if user.empty:
            return None
        user = user.to_dict(orient='records')[0]
        return user

    def get_item(self, item_id: str = None) -> Optional[Dict]:
        """Fetch item data based on item_id or scenario."""
        if not item_id:
            return None
        item = self.item_data[self.item_data['item_id'] == item_id]
        return item.to_dict(orient='records')[0] if not item.empty else None

    def get_reviews(
        self, 
        item_id: Optional[str] = None, 
        user_id: Optional[str] = None, 
        review_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch reviews filtered by various parameters."""
        if item_id is None and user_id is None and review_id is None:
            return []
        reviews = self.review_data
        if review_id:
            reviews = reviews[reviews['review_id'] == review_id]
        else:
            if item_id:
                reviews = reviews[reviews['item_id'] == item_id]
            if user_id:
                reviews = reviews[reviews['user_id'] == user_id]
        return reviews.to_dict(orient='records')
