import json
import logging
import numpy as np
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import nltk
nltk.download('vader_lexicon')

@dataclass
class RecommendationMetrics:
    hr_at_1: float
    hr_at_3: float
    hr_at_5: float
    total_scenarios: int
    hits_at_1: int
    hits_at_3: int
    hits_at_5: int

@dataclass
class SimulationMetrics:
    star_rmse: float
    sentiment_rmse: float
    useful_rmse: float
    cool_rmse: float
    funny_rmse: float
    overall_rmse: float
    sentiment_details: Dict[str, float]

class BaseEvaluator:
    """Base class for evaluation tools"""
    def __init__(self):
        self.metrics_history: List[Union[RecommendationMetrics, SimulationMetrics]] = []

    def save_metrics(self, metrics: Union[RecommendationMetrics, SimulationMetrics]):
        """Save metrics to history"""
        self.metrics_history.append(metrics)

    def get_metrics_history(self):
        """Get all historical metrics"""
        return self.metrics_history

class RecommendationEvaluator(BaseEvaluator):
    """Evaluator for recommendation tasks"""
    
    def __init__(self):
        super().__init__()
        self.n_values = [1, 3, 5]  # 预定义的n值数组

    def calculate_hr_at_n(
        self,
        ground_truth: List[str],
        predictions: List[List[str]]
    ) -> RecommendationMetrics:
        """Calculate Hit Rate at different N values"""
        total = len(ground_truth)
        hits = {n: 0 for n in self.n_values}
        
        for gt, pred in zip(ground_truth, predictions):
            for n in self.n_values:
                if gt in pred[:n]:
                    hits[n] += 1
        
        metrics = RecommendationMetrics(
            hr_at_1=hits[1] / total if total > 0 else 0,
            hr_at_3=hits[3] / total if total > 0 else 0,
            hr_at_5=hits[5] / total if total > 0 else 0,
            total_scenarios=total,
            hits_at_1=hits[1],
            hits_at_3=hits[3],
            hits_at_5=hits[5]
        )
        
        self.save_metrics(metrics)
        return metrics

class SimulationEvaluator(BaseEvaluator):
    """Evaluator for simulation tasks"""
    
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = self._get_device(device)
        
        pipeline_device = self.device
        st_device = "cuda" if self.device == 0 else "cpu" 
        
        self.sia = SentimentIntensityAnalyzer()
        self.emotion_classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=5,
            device=pipeline_device
        )
        self.topic_model = SentenceTransformer(
            'paraphrase-MiniLM-L6-v2',
            device=st_device
        )
        
    def _get_device(self, device: str) -> int:
        """Parse device from string"""
        if device == "gpu":
            if torch.cuda.is_available():
                return 0  # GPU
            else:
                logging.warning("GPU is not available, falling back to CPU")
                return -1  # CPU
        elif device == "cpu":
            return -1  # CPU
        elif device == "auto":
            return 0 if torch.cuda.is_available() else -1
        else:
            raise ValueError("Device type must be 'cpu', 'gpu' or 'auto'")

    def calculate_metrics(
        self,
        simulated_data: Dict,
        real_data: Dict
    ) -> SimulationMetrics:
        """Calculate all simulation metrics"""
        # Calculate basic metrics
        star_rmse = np.sqrt(np.mean((simulated_data['stars'] - real_data['stars']) ** 2))
        useful_rmse = np.sqrt(np.mean((simulated_data['useful'] - real_data['useful']) ** 2))
        cool_rmse = np.sqrt(np.mean((simulated_data['cool'] - real_data['cool']) ** 2))
        funny_rmse = np.sqrt(np.mean((simulated_data['funny'] - real_data['funny']) ** 2))

        # Calculate sentiment metrics
        sentiment_details = self._calculate_sentiment_metrics(
            simulated_data['review'],
            real_data['review']
        )
        sentiment_rmse = sentiment_details['overall_similarity']

        # Calculate overall RMSE
        overall_rmse = np.mean([
            star_rmse,
            sentiment_rmse,
            useful_rmse,
            cool_rmse,
            funny_rmse
        ])

        metrics = SimulationMetrics(
            star_rmse=star_rmse,
            sentiment_rmse=sentiment_rmse,
            useful_rmse=useful_rmse,
            cool_rmse=cool_rmse,
            funny_rmse=funny_rmse,
            overall_rmse=overall_rmse,
            sentiment_details=sentiment_details
        )

        self.save_metrics(metrics)
        return metrics

    def _calculate_sentiment_metrics(
        self,
        text1: str,
        text2: str
    ) -> Dict[str, float]:
        """Calculate detailed sentiment metrics between two texts"""
        # Polarity analysis
        polarity1 = self.sia.polarity_scores(text1)['compound']
        polarity2 = self.sia.polarity_scores(text2)['compound']
        polarity_similarity = 1 - abs(polarity1 - polarity2) / 2

        # Emotion analysis
        emotions1 = self.emotion_classifier(text1)[0]
        emotions2 = self.emotion_classifier(text2)[0]
        emotion_similarity = self._calculate_emotion_similarity(emotions1, emotions2)

        # Topic analysis
        embeddings = self.topic_model.encode([text1, text2])
        topic_similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                               (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))

        return {
            'polarity_similarity': polarity_similarity,
            'emotion_similarity': emotion_similarity,
            'topic_similarity': topic_similarity,
            'overall_similarity': np.mean([
                polarity_similarity,
                emotion_similarity,
                topic_similarity
            ])
        }

    def _calculate_emotion_similarity(
        self,
        emotions1: List[Dict],
        emotions2: List[Dict]
    ) -> float:
        """Calculate similarity between two emotion distributions"""
        # Convert emotions to vectors
        emotion_dict1 = {e['label']: e['score'] for e in emotions1}
        emotion_dict2 = {e['label']: e['score'] for e in emotions2}
        
        # Get all unique emotions
        all_emotions = set(emotion_dict1.keys()) | set(emotion_dict2.keys())
        
        # Create vectors
        vec1 = np.array([emotion_dict1.get(e, 0) for e in all_emotions])
        vec2 = np.array([emotion_dict2.get(e, 0) for e in all_emotions])
        
        # Calculate cosine similarity
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))) 