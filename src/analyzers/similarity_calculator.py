"""
Accurate similarity calculation modules for SEO content analysis.
"""

import numpy as np
from difflib import SequenceMatcher
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """Handles accurate similarity calculations for SEO content."""
    
    @staticmethod
    def calculate_string_similarity(text1: str, text2: str) -> float:
        """
        Calculate accurate string similarity using multiple methods.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score (0-100)
        """
        if not text1 or not text2:
            return 0.0
        
        text1 = str(text1).lower().strip()
        text2 = str(text2).lower().strip()
        
        if not text1 or not text2:
            return 0.0
        
        # Use SequenceMatcher for direct string comparison
        similarity = SequenceMatcher(None, text1, text2).ratio()
        
        return similarity * 100
    
    @staticmethod
    def calculate_jaccard_similarity(set1: List[str], set2: List[str]) -> float:
        """
        Calculate Jaccard similarity between keyword sets.
        
        Args:
            set1: First set of keywords
            set2: Second set of keywords
            
        Returns:
            Jaccard similarity score (0-100)
        """
        if not set1 or not set2:
            return 0.0
        
        # Clean and normalize keywords
        clean_set1 = {str(k).lower().strip() for k in set1 if k and str(k).strip()}
        clean_set2 = {str(k).lower().strip() for k in set2 if k and str(k).strip()}
        
        if not clean_set1 or not clean_set2:
            return 0.0
        
        intersection = clean_set1.intersection(clean_set2)
        union = clean_set1.union(clean_set2)
        
        return len(intersection) / len(union) * 100 if union else 0.0
    
    @classmethod
    def calculate_similarity_matrix(cls, texts: List[str]) -> np.ndarray:
        """
        Calculate similarity matrix for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of similarity scores
        """
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = cls.calculate_string_similarity(
                    texts[i], texts[j]
                )
        
        return similarity_matrix
    
    @classmethod
    def calculate_composite_score(cls, similarities: Dict[str, float], 
                                weights: Dict[str, float]) -> float:
        """
        Calculate weighted composite similarity score.
        
        Args:
            similarities: Dictionary of similarity scores
            weights: Dictionary of weights for each similarity type
            
        Returns:
            Weighted composite score
        """
        total_score = 0.0
        total_weight = 0.0
        
        for key, score in similarities.items():
            if key in weights:
                total_score += score * weights[key]
                total_weight += weights[key]
        
        return total_score / total_weight if total_weight > 0 else 0.0

class IntentDetector:
    """Detects content intent from titles and content."""
    
    @staticmethod
    def detect_intent(title: str) -> str:
        """
        Detect content intent from title patterns.
        
        Args:
            title: Page title
            
        Returns:
            Intent classification
        """
        if not title:
            return 'unknown'
        
        title_lower = str(title).lower()
        
        # Intent patterns with priorities
        intent_patterns = {
            'how-to': ['how to', 'guide', 'tutorial', 'step by step', 'steps to'],
            'comparison': ['best', 'top', 'review', 'vs', 'versus', 'comparison', 'compare'],
            'informational': ['what is', 'definition', 'meaning', 'explained', 'overview'],
            'transactional': ['buy', 'price', 'cost', 'cheap', 'deal', 'discount', 'purchase'],
            'question': ['?', 'why', 'when', 'where', 'who', 'what', 'how']
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in title_lower for pattern in patterns):
                return intent
        
        return 'general'
    
    @classmethod
    def detect_intents(cls, titles: List[str]) -> List[str]:
        """Detect intents for multiple titles."""
        return [cls.detect_intent(title) for title in titles]

class RiskAssessor:
    """Assesses cannibalization risk based on similarity scores."""
    
    @staticmethod
    def assess_risk(composite_score: float, same_intent: bool) -> str:
        """
        Assess cannibalization risk level.
        
        Args:
            composite_score: Overall similarity score
            same_intent: Whether pages have same intent
            
        Returns:
            Risk level: 'High', 'Medium', or 'Low'
        """
        if composite_score > 80 and same_intent:
            return 'High'
        elif composite_score > 60:
            return 'Medium'
        else:
            return 'Low'
    
    @classmethod
    def assess_risks(cls, scores: List[float], intents: List[bool]) -> List[str]:
        """Assess risks for multiple URL pairs."""
        return [cls.assess_risk(score, intent) for score, intent in zip(scores, intents)]
