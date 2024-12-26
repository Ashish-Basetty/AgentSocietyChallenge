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
        """Fetch user data based on user_id or scenario."""
        self._ensure_task()
        
        user_id = user_id or self.task.get('user') if self.task else None
        if not user_id:
            return None
        
        user = self.user_data[self.user_data['user_id'] == user_id]
        if user.empty:
            return None
        user = user.to_dict(orient='records')[0]
        return user

    def get_item(self, item_id: Optional[str] = None) -> Optional[Dict]:
        """Fetch item data based on item_id or scenario."""
        self._ensure_task()  # Ensure scenario is set
        item_id = item_id or self.task.get('item') if self.task else None
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
        self._ensure_task()
        
        reviews = self.review_data

        if review_id:
            reviews = reviews[reviews['review_id'] == review_id]
        else:
            item_id = item_id or (self.task.get('item') if self.task else None)
            user_id = user_id or (self.task.get('user') if self.task else None)
            if item_id:
                reviews = reviews[reviews['item_id'] == item_id]
            if user_id:
                reviews = reviews[reviews['user_id'] == user_id]
        return reviews.to_dict(orient='records')
