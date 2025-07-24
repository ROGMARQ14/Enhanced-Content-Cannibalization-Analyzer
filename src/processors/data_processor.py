"""
Data processing pipeline for cannibalization analysis - FIXED VERSION
"""

import pandas as pd
from typing import Dict, Any
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
            'title': 0.45,
            'h1': 0.30,
            'meta': 0.0,  # Removed from analysis
            'keywords': 0.15,
            'semantic': 0.10
        }

    def process_internal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process internal SEO data."""
        validation = DataLoader.validate_seo_data(df, 'internal')

        if not validation['valid']:
            raise ValueError(
                f"Invalid internal data: {validation['issues']}"
            )

        # Extract relevant columns
        processed = pd.DataFrame()
        processed['URL'] = df[validation['url_column']]
        processed['Title'] = df[
            validation['title_column']
        ].fillna('').astype(str)
        processed['H1'] = df[
            validation['h1_column']
        ].fillna('').astype(str)

        if validation['meta_column']:
            processed['Meta'] = df[
                validation['meta_column']
            ].fillna('').astype(str)
        else:
            processed['Meta'] = ''

        # Clean URLs and remove duplicates
        processed['URL'] = processed['URL'].astype(str).str.strip()
        processed = processed.drop_duplicates(subset=['URL'])

        # Remove empty URLs
        processed = processed[processed['URL'].str.len() > 0]

        # Detect intents
        processed['Intent'] = IntentDetector.detect_intents(
            processed['Title'].tolist()
        )

        logger.info(f"Processed {len(processed)} unique URLs")
        return processed

    def process_gsc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Google Search Console data."""
        # Find URL and Query columns
        url_col = None
        query_col = None

        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['url', 'page', 'landing']):
                url_col = col
            elif any(x in col_lower for x in ['query', 'keyword']):
                query_col = col

        if not url_col or not query_col:
            raise ValueError(
                "Could not find URL and Query columns in GSC data"
            )

        # Clean URLs and group queries
        df[url_col] = df[url_col].astype(str).str.strip()

        # Group queries by URL and clean
        gsc_processed = df.groupby(url_col)[query_col].apply(
            lambda x: [
                str(q).strip() for q in x if str(q).strip()
            ]
        ).reset_index()
        gsc_processed.columns = ['URL', 'Queries']

        # Remove URLs with no queries
        gsc_processed = gsc_processed[
            gsc_processed['Queries'].apply(len) > 0
        ]

        logger.info(f"Processed {len(gsc_processed)} URLs with queries")
        return gsc_processed

    def merge_data(self, internal_df: pd.DataFrame,
                   gsc_df: pd.DataFrame) -> pd.DataFrame:
        """Merge internal and GSC data."""
        merged = pd.merge(
            internal_df,
            gsc_df,
            on='URL',
            how='inner'  # Only keep URLs that exist in both datasets
        )

        # Fill missing queries with empty list
        merged['Queries'] = merged['Queries'].apply(
            lambda x: x if isinstance(x, list) else []
        )

        logger.info(
            f"Merged data: {len(merged)} URLs with both internal and GSC data"
        )
        return merged

    def calculate_all_similarities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate similarities with intelligent filtering."""
        results = []

        # Get data
        urls = df['URL'].tolist()
        titles = df['Title'].tolist()
        h1s = df['H1'].tolist()
        metas = df['Meta'].tolist()
        queries = df['Queries'].tolist()
        intents = df['Intent'].tolist()

        # Intelligent filtering
        logger.info("Calculating similarities with intelligent filtering...")

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                # Skip if both titles are empty
                if not titles[i].strip() and not titles[j].strip():
                    continue

                # Skip if both H1s are empty
                if not h1s[i].strip() and not h1s[j].strip():
                    continue

                # Calculate keyword overlap for filtering
                keyword_overlap = SimilarityCalculator.calculate_jaccard_similarity(
                    queries[i], queries[j]
                )

                # Quick title similarity check
                quick_title_sim = SimilarityCalculator.calculate_string_similarity(
                    titles[i][:50], titles[j][:50]
                )

                # Filter pairs
                if (intents[i] == intents[j] or
                        keyword_overlap > 10 or
                        quick_title_sim > 20):

                    # Calculate full similarities
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
                        'Keyword_Overlap': keyword_overlap
                    }

                    # Calculate composite score
                    composite_score = SimilarityCalculator.calculate_composite_score(
                        similarities, self.weights
                    )

                    # Skip very low similarity pairs
                    if composite_score < 5:
                        continue

                    # Check if same intent
                    same_intent = intents[i] == intents[j]

                    # Assess risk
                    risk_level = RiskAssessor.assess_risk(
                        composite_score, same_intent
                    )

                    # Clean data for export
                    def clean_text(text):
                        return str(text).strip() if pd.notna(
                            text
                        ) and str(text).strip() else ''

                    results.append({
                        'URL_1': clean_text(urls[i]),
                        'URL_2': clean_text(urls[j]),
                        'Title_1': clean_text(titles[i]),
                        'Title_2': clean_text(titles[j]),
                        'H1_1': clean_text(h1s[i]),
                        'H1_2': clean_text(h1s[j]),
                        'Meta_1': clean_text(metas[i]),
                        'Meta_2': clean_text(metas[j]),
                        'Intent_1': clean_text(intents[i]),
                        'Intent_2': clean_text(intents[j]),
                        'Same_Intent': same_intent,
                        **similarities,
                        'Composite_Score': round(composite_score, 2),
                        'Risk_Level': risk_level
                    })

        if not results:
            logger.warning("No significant cannibalization pairs found")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Sort by composite score descending
        results_df = results_df.sort_values(
            'Composite_Score', ascending=False
        )

        # Remove any rows with NaN values
        results_df = results_df.dropna()

        logger.info(
            f"Analysis complete. Found {len(results_df)} relevant URL pairs"
        )
        return results_df

    def process_complete_analysis(self, internal_df: pd.DataFrame,
                                  gsc_df: pd.DataFrame) -> pd.DataFrame:
        """Run complete analysis pipeline with improved filtering."""
        logger.info("Starting complete analysis pipeline")

        # Process data
        internal_processed = self.process_internal_data(internal_df)
        gsc_processed = self.process_gsc_data(gsc_df)

        if len(internal_processed) == 0 or len(gsc_processed) == 0:
            raise ValueError("No valid data to analyze after processing")

        # Merge data
        merged = self.merge_data(internal_processed, gsc_processed)

        if len(merged) < 2:
            raise ValueError("Need at least 2 URLs with data to analyze")

        # Calculate similarities with filtering
        results = self.calculate_all_similarities(merged)

        if len(results) == 0:
            logger.warning("No significant cannibalization detected")
            return pd.DataFrame()

        logger.info(
            f"Analysis complete. Found {len(results)} relevant URL pairs"
        )
        return results

    def get_summary_stats(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics."""
        if results_df.empty:
            return {
                'total_pairs': 0,
                'high_risk': 0,
                'medium_risk': 0,
                'low_risk': 0,
                'avg_composite_score': 0.0,
                'max_composite_score': 0.0,
                'intent_distribution': {}
            }

        return {
            'total_pairs': len(results_df),
            'high_risk': len(
                results_df[results_df['Risk_Level'] == 'High']
            ),
            'medium_risk': len(
                results_df[results_df['Risk_Level'] == 'Medium']
            ),
            'low_risk': len(
                results_df[results_df['Risk_Level'] == 'Low']
            ),
            'avg_composite_score': round(
                results_df['Composite_Score'].mean(), 2
            ),
            'max_composite_score': round(
                results_df['Composite_Score'].max(), 2
            ),
            'intent_distribution': results_df[
                'Intent_1'
            ].value_counts().to_dict()
        }
