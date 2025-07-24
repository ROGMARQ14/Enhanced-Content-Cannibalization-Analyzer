"""
Report generation and export utilities.
"""

import pandas as pd
import io
from datetime import datetime
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Handles report generation and export functionality."""
    
    @staticmethod
    def generate_summary_report(results_df: pd.DataFrame, 
                              summary_stats: Dict[str, Any]) -> str:
        """Generate a text summary report."""
        
        report = f"""
CONTENT CANNIBALIZATION ANALYSIS SUMMARY
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

OVERVIEW:
- Total URL pairs analyzed: {summary_stats['total_pairs']:,}
- High risk pairs: {summary_stats['high_risk']}
- Medium risk pairs: {summary_stats['medium_risk']}
- Low risk pairs: {summary_stats['low_risk']}
- Average composite score: {summary_stats['avg_composite_score']:.1f}%
- Maximum composite score: {summary_stats['max_composite_score']:.1f}%

RISK DISTRIBUTION:
"""
        
        # Add risk distribution
        for level in ['High', 'Medium', 'Low']:
            count = summary_stats.get(f'{level.lower()}_risk', 0)
            if count > 0:
                report += f"- {level} Risk: {count} pairs\n"
        
        # Add top risks
        report += "\nTOP 10 CANNIBALIZATION RISKS:\n"
        top_risks = results_df.head(10)
        
        for idx, (_, row) in enumerate(top_risks.iterrows(), 1):
            report += f"\n{idx}. {row['URL_1']}\n"
            report += f"   vs {row['URL_2']}\n"
            report += f"   Score: {row['Composite_Score']:.1f}% | Risk: {row['Risk_Level']}\n"
            report += f"   Intent: {row['Intent_1']} vs {row['Intent_2']}\n"
        
        return report
    
    @staticmethod
    def generate_csv_data(results_df: pd.DataFrame) -> str:
        """Generate CSV data for download."""
        buffer = io.StringIO()
        results_df.to_csv(buffer, index=False)
        return buffer.getvalue()
    
    @staticmethod
    def generate_json_report(results_df: pd.DataFrame, 
                           summary_stats: Dict[str, Any]) -> str:
        """Generate JSON report for API integration."""
        
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_pairs': summary_stats['total_pairs'],
                'risk_distribution': {
                    'high': summary_stats['high_risk'],
                    'medium': summary_stats['medium_risk'],
                    'low': summary_stats['low_risk']
                }
            },
            'summary': summary_stats,
            'top_risks': results_df.head(20).to_dict('records'),
            'all_data': results_df.to_dict('records')
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    @staticmethod
    def generate_priority_actions(results_df: pd.DataFrame) -> pd.DataFrame:
        """Generate priority action table for high-risk pairs."""
        
        high_risk = results_df[results_df['Risk_Level'] == 'High'].copy()
        
        if high_risk.empty:
            return pd.DataFrame()
        
        # Add recommendations
        high_risk['Recommended_Action'] = high_risk.apply(
            lambda row: ReportGenerator._get_recommendation(row), axis=1
        )
        
        high_risk['Priority'] = range(1, len(high_risk) + 1)
        
        return high_risk[[
            'Priority', 'URL_1', 'URL_2', 'Composite_Score', 
            'Recommended_Action', 'Same_Intent'
        ]]
    
    @staticmethod
    def _get_recommendation(row: pd.Series) -> str:
        """Generate specific recommendation for a URL pair."""
        
        if row['Same_Intent'] and row['Composite_Score'] > 90:
            return "Consider consolidating content or implementing canonical tags"
        elif row['Title_Similarity'] > 85:
            return "Differentiate titles to target distinct search intents"
        elif row['H1_Similarity'] > 80:
            return "Update H1 tags to better distinguish page focus"
        elif row['Keyword_Overlap'] > 70:
            return "Review keyword targeting and adjust content strategy"
        else:
            return "Monitor and consider minor content differentiation"
    
    @staticmethod
    def generate_consolidation_plan(results_df: pd.DataFrame) -> pd.DataFrame:
        """Generate content consolidation recommendations."""
        
        high_risk = results_df[results_df['Risk_Level'] == 'High'].copy()
        
        if high_risk.empty:
            return pd.DataFrame()
        
        # Group by URL to find clusters
        url_groups = {}
        
        for _, row in high_risk.iterrows():
            url1, url2 = row['URL_1'], row['URL_2']
            
            if url1 not in url_groups:
                url_groups[url1] = []
            if url2 not in url_groups:
                url_groups[url2] = []
            
            url_groups[url1].append((url2, row['Composite_Score']))
            url_groups[url2].append((url1, row['Composite_Score']))
        
        # Create consolidation recommendations
        recommendations = []
        processed = set()
        
        for url, conflicts in url_groups.items():
            if url in processed:
                continue
            
            # Find the best URL to keep (highest authority/traffic)
            all_urls = [url] + [u for u, _ in conflicts]
            scores = [100] + [s for _, s in conflicts]  # Give base URL high score
            
            # Simple heuristic: keep URL with highest score
            best_idx = scores.index(max(scores))
            canonical_url = all_urls[best_idx]
            urls_to_redirect = [u for u in all_urls if u != canonical_url]
            
            recommendations.append({
                'Canonical_URL': canonical_url,
                'URLs_to_Redirect': ', '.join(urls_to_redirect),
                'Justification': f'Cluster of {len(all_urls)} competing pages'
            })
            
            processed.update(all_urls)
        
        return pd.DataFrame(recommendations)
    
    @classmethod
    def export_all_formats(cls, results_df: pd.DataFrame, 
                         summary_stats: Dict[str, Any]) -> Dict[str, str]:
        """Export all report formats."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            'csv': cls.generate_csv_data(results_df),
            'summary': cls.generate_summary_report(results_df, summary_stats),
            'json': cls.generate_json_report(results_df, summary_stats),
            'priority_actions': cls.generate_priority_actions(results_df).to_csv(index=False),
            'consolidation_plan': cls.generate_consolidation_plan(results_df).to_csv(index=False),
            'filenames': {
                'csv': f'cannibalization_analysis_{timestamp}.csv',
                'summary': f'analysis_summary_{timestamp}.txt',
                'json': f'analysis_report_{timestamp}.json',
                'priority': f'priority_actions_{timestamp}.csv',
                'consolidation': f'consolidation_plan_{timestamp}.csv'
            }
        }
