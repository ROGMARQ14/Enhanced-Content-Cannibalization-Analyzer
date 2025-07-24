"""
Enhanced Content Cannibalization Analyzer - FIXED VERSION
Robust CSV parsing with comprehensive error handling
"""

import streamlit as st
import numpy as np
import logging
from datetime import datetime
import io
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.data_loader import DataLoader
from processors.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Enhanced Content Cannibalization Analyzer - FIXED",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Enhanced Content Cannibalization Analyzer - FIXED")
st.markdown("""
This **FIXED VERSION** resolves the pandas ParserError issues with robust CSV parsing.

**Key improvements:**
- ✅ Robust CSV parsing with encoding detection
- ✅ Automatic delimiter detection
- ✅ Comprehensive error messages
- ✅ Better data validation
- ✅ Enhanced user feedback
- ✅ Persistent download buttons (no app reset)
- ✅ Streamlined analysis settings
- ✅ Clean reports without NaN values
""")


def load_and_validate_files(internal_file, gsc_file):
    """Load and validate both files with comprehensive error handling."""
    
    try:
        # Load internal data
        st.info("Loading internal SEO data...")
        internal_df = DataLoader.load_csv(internal_file, "Internal SEO Report")
        
        # Validate internal data
        validation = DataLoader.validate_seo_data(internal_df, 'internal')
        if not validation['valid']:
            st.error(
                f"❌ Internal data validation failed: "
                f"{', '.join(validation['issues'])}"
            )
            return None, None
        
        # Load GSC data
        st.info("Loading Google Search Console data...")
        gsc_df = DataLoader.load_csv(gsc_file, "GSC Report")
        
        # Validate GSC data
        gsc_validation = DataLoader.validate_seo_data(gsc_df, 'gsc')
        if not gsc_validation['valid']:
            st.error(
                f"❌ GSC data validation failed: "
                f"{', '.join(gsc_validation['issues'])}"
            )
            return None, None
        
        st.success("✅ Both files loaded successfully!")
        return internal_df, gsc_df
        
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        logger.error(f"File loading error: {str(e)}")
        return None, None


def clean_dataframe_for_export(df):
    """Clean dataframe by removing NaN values and formatting for export."""
    # Create a copy to avoid modifying original
    clean_df = df.copy()
    
    # Replace NaN with empty strings
    clean_df = clean_df.fillna('')
    
    # Convert float columns to strings with proper formatting
    float_cols = clean_df.select_dtypes(include=[np.float64, np.float32]).columns
    for col in float_cols:
        clean_df[col] = clean_df[col].apply(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x)
        )
    
    # Ensure all data is string for consistent export
    clean_df = clean_df.astype(str)
    
    return clean_df


def main():
    """Main application function."""
    
    # Initialize session state for results persistence
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_summary' not in st.session_state:
        st.session_state.analysis_summary = None
    
    # Sidebar for file uploads
    with st.sidebar:
        st.markdown("### 📁 Upload Files")
        
        internal_file = st.file_uploader(
            "Internal SEO Report (CSV)",
            type=['csv', 'txt'],
            help="Upload your internal SEO crawl data (Screaming Frog, etc.)"
        )
        
        gsc_file = st.file_uploader(
            "Google Search Console Report (CSV)",
            type=['csv', 'txt'],
            help="Upload your GSC query report"
        )
        
        st.markdown("---")
        
        # Analysis settings
        st.markdown("### ⚙️ Analysis Settings")
        
        # Updated weights without Meta Description
        title_weight = st.slider("Title Similarity", 0.0, 1.0, 0.45, 0.05)
        h1_weight = st.slider("H1 Similarity", 0.0, 1.0, 0.30, 0.05)
        keyword_weight = st.slider("Keyword Overlap", 0.0, 1.0, 0.15, 0.05)
        semantic_weight = st.slider("Semantic Similarity", 0.0, 1.0, 0.10, 0.05)
        
        # Auto-adjust weights to sum to 1.0
        total_weight = title_weight + h1_weight + keyword_weight + semantic_weight
        if total_weight > 0:
            # Normalize weights
            title_weight = title_weight / total_weight
            h1_weight = h1_weight / total_weight
            keyword_weight = keyword_weight / total_weight
            semantic_weight = semantic_weight / total_weight
        
        weights = {
            'title': title_weight,
            'h1': h1_weight,
            'meta': 0.0,
            'keywords': keyword_weight,
            'semantic': semantic_weight
        }
    
    # Main content area
    if internal_file and gsc_file:
        
        # Load files
        internal_df, gsc_df = load_and_validate_files(internal_file, gsc_file)
        
        if internal_df is not None and gsc_df is not None:
            
            # Display file info
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Internal Data Preview")
                st.dataframe(internal_df.head())
                st.caption(f"Loaded {len(internal_df)} rows")
            
            with col2:
                st.markdown("### 📈 GSC Data Preview")
                st.dataframe(gsc_df.head())
                st.caption(f"Loaded {len(gsc_df)} rows")
            
            # Run analysis
            if st.button("🔍 Run Cannibalization Analysis", type="primary"):
                
                with st.spinner("Analyzing content cannibalization..."):
                    try:
                        # Initialize processor with custom weights
                        processor = DataProcessor()
                        processor.weights = weights
                        
                        # Run analysis
                        results = processor.process_complete_analysis(
                            internal_df, gsc_df
                        )
                        summary = processor.get_summary_stats(results)
                        
                        # Clean results before storing
                        results = results.fillna('')
                        
                        # Store results in session state for persistence
                        st.session_state.analysis_results = results
                        st.session_state.analysis_summary = summary
                        
                        st.success("✅ Analysis complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
                        logger.error(f"Analysis error: {str(e)}")
                        st.exception(e)
            
            # Display results if available (persistent across interactions)
            if st.session_state.analysis_results is not None:
                results = st.session_state.analysis_results
                summary = st.session_state.analysis_summary
                
                # Summary metrics
                st.markdown("### 📋 Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Pairs", summary['total_pairs'])
                with col2:
                    st.metric("High Risk", summary['high_risk'])
                with col3:
                    st.metric("Medium Risk", summary['medium_risk'])
                with col4:
                    st.metric("Avg Score", f"{summary['avg_composite_score']:.1f}%")
                
                # Results table
                st.markdown("### 📊 Detailed Results")
                
                # Filter options
                risk_filter = st.multiselect(
                    "Filter by Risk Level",
                    options=['High', 'Medium', 'Low'],
                    default=['High', 'Medium']
                )
                
                filtered_results = results[results['Risk_Level'].isin(risk_filter)]
                
                # Display results - clean display without NaN
                display_df = filtered_results[[
                    'URL_1', 'URL_2', 'Composite_Score', 
                    'Risk_Level', 'Same_Intent'
                ]].head(50).copy()
                
                # Clean display
                display_df = display_df.fillna('')
                
                st.dataframe(
                    display_df,
                    use_container_width=True
                )
                
                # Download section - Persistent across interactions
                st.markdown("### 📥 Download Results")
                
                # Clean data for export
                clean_results = clean_dataframe_for_export(results)
                
                # Generate all reports with cleaned data
                try:
                    # CSV export
                    csv_buffer = io.StringIO()
                    clean_results.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    # Summary report
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    summary_text = f"""
CONTENT CANNIBALIZATION ANALYSIS SUMMARY
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

OVERVIEW:
- Total URL pairs analyzed: {summary['total_pairs']:,}
- High risk pairs: {summary['high_risk']}
- Medium risk pairs: {summary['medium_risk']}
- Low risk pairs: {summary['low_risk']}
- Average composite score: {summary['avg_composite_score']:.1f}%

TOP CANNIBALIZATION RISKS:
"""
                    
                    # Add top risks (cleaned)
                    top_risks = results[results['Risk_Level'].isin(['High', 'Medium'])].head(10)
                    for _, row in top_risks.iterrows():
                        url1 = str(row['URL_1']).strip()
                        url2 = str(row['URL_2']).strip()
                        score = str(row['Composite_Score']).strip()
                        risk = str(row['Risk_Level']).strip()
                        if url1 and url2 and url1 != 'nan' and url2 != 'nan':
                            summary_text += (
                                f"\n{url1}\n  vs {url2}\n"
                                f"  Score: {score}% | Risk: {risk}\n"
                            )
                    
                    # Priority actions (cleaned)
                    priority_actions = results[results['Risk_Level'] == 'High'].head(20)
                    priority_clean = clean_dataframe_for_export(priority_actions)
                    
                    # Create download buttons with persistent keys
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="📊 Download Full Analysis (CSV)",
                            data=csv_data,
                            file_name=f"enhanced_cannibalization_analysis_{timestamp}.csv",
                            mime="text/csv",
                            key="download_csv_persistent"
                        )
                    
                    with col2:
                        st.download_button(
                            label="📄 Download Summary (TXT)",
                            data=summary_text,
                            file_name=f"cannibalization_summary_{timestamp}.txt",
                            mime="text/plain",
                            key="download_summary_persistent"
                        )
                    
                    with col3:
                        if not priority_clean.empty:
                            priority_csv = io.StringIO()
                            priority_clean.to_csv(priority_csv, index=False)
                            st.download_button(
                                label="📋 Download Priority Actions (CSV)",
                                data=priority_csv.getvalue(),
                                file_name=f"priority_actions_{timestamp}.csv",
                                mime="text/csv",
                                key="download_priority_persistent"
                            )
                
                except Exception as e:
                    st.error(f"❌ Error generating downloads: {str(e)}")
                    logger.error(f"Download generation error: {str(e)}")
    
    else:
        # Instructions
        st.info("""
        👆 **Please upload both files to begin analysis:**
        
        ### 📁 Required Files:
        
        **1. Internal SEO Report (CSV)**
        - From Screaming Frog, Sitebulb, or similar crawler
        - Must contain columns: URL, Title, H1, Meta Description
        
        **2. Google Search Console Report (CSV)**
        - From GSC Performance report
        - Must contain: Landing Page and Query columns
        
        ### 🛠️ **Troubleshooting:**
        
        **If you get parsing errors:**
        - ✅ Try saving your file as UTF-8 encoded CSV
        - ✅ Check for special characters in URLs/titles
        - ✅ Ensure consistent delimiter (comma, semicolon, or tab)
        
        **Column detection is automatic** - the tool will find the right columns
        even if they have different names.
        """)
        
        with st.expander("📖 **Detailed Instructions & Examples**"):
            st.markdown("""
            ### **Internal SEO Report Format:**
            Your file should look like this:
            ```
            Address,Title 1,H1-1,Meta Description 1
            https://example.com/page1,Page Title,H1 Text,Meta description...
            https://example.com/page2,Another Title,Different H1,Another description...
            ```
            
            ### **GSC Report Format:**
            Your file should look like this:
            ```
            Landing page,Query
            https://example.com/page1,keyword research
            https://example.com/page1,seo tools
            https://example.com/page2,content marketing
            ```
            
            ### **Analysis Weights (Updated):**
            - **Title Similarity (45%)**: Most important for SEO
            - **H1 Similarity (30%)**: Key on-page signal
            - **Keyword Overlap (15%)**: Actual search competition
            - **Semantic Similarity (10%)**: Overall content theme
            - **Meta Description (0%)**: Removed from analysis settings
            """)


# Footer
st.markdown("---")
st.markdown("🎯 **Enhanced Content Cannibalization Analyzer - FIXED VERSION**")

if __name__ == "__main__":
    main()
