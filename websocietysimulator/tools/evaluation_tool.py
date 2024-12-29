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
    average_hr: float
    total_scenarios: int
    hits_at_1: int
    hits_at_3: int
    hits_at_5: int

@dataclass
class SimulationMetrics:
    star_mape: float
    polarity_mape: float
    emotion_mape: float
    topic_mape: float
    overall_mape: float

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
        
        hr_at_1 = hits[1] / total if total > 0 else 0
        hr_at_3 = hits[3] / total if total > 0 else 0
        hr_at_5 = hits[5] / total if total > 0 else 0
        average_hr = (hr_at_1 + hr_at_3 + hr_at_5) / 3
        metrics = RecommendationMetrics(
            hr_at_1=hr_at_1,
            hr_at_3=hr_at_3,
            hr_at_5=hr_at_5,
            average_hr=average_hr,
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
        simulated_data: List[Dict],
        real_data: List[Dict]
    ) -> SimulationMetrics:
        """Calculate all simulation metrics"""
        # Calculate stars metrics
        simulated_stars = [item['stars'] for item in simulated_data]
        real_stars = [item['stars'] for item in real_data]
        star_mape = 0
        for sim_star, real_star in zip(simulated_stars, real_stars):
            star_mape += np.abs(sim_star - real_star) / real_star
        star_mape = star_mape / len(real_stars)

        # Calculate review metrics
        simulated_reviews = [item['review'] for item in simulated_data]
        real_reviews = [item['review'] for item in real_data]
        review_details = self._calculate_review_metrics(
            simulated_reviews,
            real_reviews
        )

        polarity_mape = review_details['polarity_mape']
        emotion_mape = review_details['emotion_mape']
        topic_mape = review_details['topic_mape']
        # Calculate overall MAPE
        overall_mape = np.mean([
            star_mape,
            polarity_mape,
            emotion_mape,
            topic_mape,
        ])

        metrics = SimulationMetrics(
            star_mape=star_mape,
            polarity_mape=polarity_mape,
            emotion_mape=emotion_mape,
            topic_mape=topic_mape,
            overall_mape=overall_mape,
        )

        self.save_metrics(metrics)
        return metrics

    def _calculate_review_metrics(
        self,
        simulated_reviews: List[str],
        real_reviews: List[str]
    ) -> Dict[str, float]:
        """Calculate detailed review metrics between two texts"""
        # Polarity analysis
        polarity_mape = 0
        emotion_mape = 0
        topic_mape = 0
        for simulated_review, real_review in zip(simulated_reviews, real_reviews):
            # Polarity analysis
            polarity1 = self.sia.polarity_scores(simulated_review)['compound']
            polarity2 = self.sia.polarity_scores(real_review)['compound']
            polarity_error = abs(polarity1 - polarity2) / polarity2
            polarity_mape += polarity_error

            # Emotion analysis
            emotions1 = self.emotion_classifier(simulated_review)[0]
            emotions2 = self.emotion_classifier(real_review)[0]
            emotion_error = self._calculate_emotion_similarity(emotions1, emotions2)
            emotion_mape += emotion_error

            # Topic analysis
            embeddings = self.topic_model.encode([simulated_review, real_review])
            topic_error = np.mean(np.abs(embeddings[0] - embeddings[1]) / (embeddings[1] + 1e-10))
            topic_mape += topic_error

        polarity_mape = polarity_mape / len(real_reviews)
        emotion_mape = emotion_mape / len(real_reviews)
        topic_mape = topic_mape / len(real_reviews)
        return {
            'polarity_mape': polarity_mape,
            'emotion_mape': emotion_mape,
            'topic_mape': topic_mape,
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
        
        # Calculate error between emotion vectors
        error = np.mean(np.abs(vec1 - vec2) / (vec2 + 1e-10))
        return float(error)