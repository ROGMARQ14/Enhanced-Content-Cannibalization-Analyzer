"""
Data processing pipeline for cannibalization analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from ..utils.data_loader import DataLoader
from ..analyzers.similarity_calculator import (
    SimilarityCalculator, 
    IntentDetector, 
    RiskAssessor
)

logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing pipeline for cannibalization analysis."""
    
    def __init__(self):
        self.weights = {
            'title': 0.35,
            'h1': 0.25,
            'meta': 0.15,
            'keywords': 0.15,
            'semantic': 0.10
        }
    
    def process_internal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process internal SEO data."""
        validation = DataLoader.validate_seo_data(df, 'internal')
        
        if not validation['valid']:
            raise ValueError(f"Invalid internal data: {validation['issues']}")
        
        # Extract relevant columns
        processed = pd.DataFrame()
        processed['URL'] = df[validation['url_column']]
        processed['Title'] = df[validation['title_column']].fillna('').astype(str)
        processed['H1'] = df[validation['h1_column']].fillna('').astype(str)
        
        if validation['meta_column']:
            processed['Meta'] = df[validation['meta_column']].fillna('').astype(str)
        else:
            processed['Meta'] = ''
        
        if validation['embedding_column']:
            processed['Embedding'] = df[validation['embedding_column']]
        
        # Clean URLs
        processed['URL'] = processed['URL'].astype(str).str.strip()
        
        # Detect intents
        processed['Intent'] = IntentDetector.detect_intents(processed['Title'].tolist())
        
        return processed
    
    def process_gsc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Google Search Console data."""
        # Find URL and Query columns
        url_col = None
        query_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'url' in col_lower or 'page' in col_lower or 'landing' in col_lower:
                url_col = col
            elif 'query' in col_lower or 'keyword' in col_lower:
                query_col = col
        
        if not url_col or not query_col:
            raise ValueError("Could not find URL and Query columns in GSC data")
        
        # Group queries by URL
        gsc_processed = df.groupby(url_col)[query_col].apply(list).reset_index()
        gsc_processed.columns = ['URL', 'Queries']
        
        return gsc_processed
    
    def merge_data(self, internal_df: pd.DataFrame, gsc_df: pd.DataFrame) -> pd.DataFrame:
        """Merge internal and GSC data."""
        merged = pd.merge(
            internal_df, 
            gsc_df, 
            on='URL', 
            how='left'
        )
        
        # Fill missing queries with empty list
        merged['Queries'] = merged['Queries'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        return merged
    
    def calculate_all_similarities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all similarity scores for URL pairs."""
        urls = df['URL'].tolist()
        titles = df['Title'].tolist()
        h1s = df['H1'].tolist()
        metas = df['Meta'].tolist()
        queries = df['Queries'].tolist()
        intents = df['Intent'].tolist()
        
        results = []
        
        # Calculate pairwise similarities
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                similarities = {
                    'Title_Similarity': SimilarityCalculator.calculate_string_similarity(
                        titles[i], titles[j]
                    ),
                    'H1_Similarity': SimilarityCalculator.calculate_string_similarity(
                        h1s[i], h1s[j]
                    ),
                    'Meta_Similarity': SimilarityCalculator.calculate_string_similarity(
                        metas[i], metas[j]
                    ),
                    'Keyword_Overlap': SimilarityCalculator.calculate_jaccard_similarity(
                        queries[i], queries[j]
                    )
                }
                
                # Calculate composite score
                composite_score = SimilarityCalculator.calculate_composite_score(
                    similarities, self.weights
                )
                
                # Check if same intent
                same_intent = intents[i] == intents[j]
                
                # Assess risk
                risk_level = RiskAssessor.assess_risk(composite_score, same_intent)
                
                results.append({
                    'URL_1': urls[i],
                    'URL_2': urls[j],
                    'Title_1': titles[i],
                    'Title_2': titles[j],
                    'H1_1': h1s[i],
                    'H1_2': h1s[j],
                    'Meta_1': metas[i],
                    'Meta_2': metas[j],
                    'Intent_1': intents[i],
                    'Intent_2': intents[j],
                    'Same_Intent': same_intent,
                    **similarities,
                    'Composite_Score': composite_score,
                    'Risk_Level': risk_level
                })
        
        return pd.DataFrame(results)
    
    def process_complete_analysis(self, internal_df: pd.DataFrame, 
                                gsc_df: pd.DataFrame) -> pd.DataFrame:
        """Run complete analysis pipeline."""
        logger.info("Starting complete analysis pipeline")
        
        # Process data
        internal_processed = self.process_internal_data(internal_df)
        gsc_processed = self.process_gsc_data(gsc_df)
        
        # Merge data
        merged = self.merge_data(internal_processed, gsc_processed)
        
        # Calculate similarities
        results = self.calculate_all_similarities(merged)
        
        # Sort by composite score descending
        results = results.sort_values('Composite_Score', ascending=False)
        
        logger.info(f"Analysis complete. Found {len(results)} URL pairs")
        
        return results
    
    def get_summary_stats(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            'total_pairs': len(results_df),
            'high_risk': len(results_df[results_df['Risk_Level'] == 'High']),
            'medium_risk': len(results_df[results_df['Risk_Level'] == 'Medium']),
            'low_risk': len(results_df[results_df['Risk_Level'] == 'Low']),
            'avg_composite_score': results_df['Composite_Score'].mean(),
            'max_composite_score': results_df['Composite_Score'].max(),
            'intent_distribution': results_df['Intent_1'].value_counts().to_dict()
        }
