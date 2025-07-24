"""
Enhanced Content Cannibalization Analyzer - FIXED VERSION
Robust CSV parsing with comprehensive error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime
import io

# Import our new modules
from src.utils.data_loader import DataLoader
from src.processors.data_processor import DataProcessor
from src.reports.report_generator import ReportGenerator

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
""")

# Helper functions
@st.cache_data
def load_and_validate_files(internal_file, gsc_file):
    """Load and validate both files with comprehensive error handling."""
    
    try:
        # Load internal data
        st.info("Loading internal SEO data...")
        internal_df = DataLoader.load_csv(internal_file, "Internal SEO Report")
        
        # Validate internal data
        validation = DataLoader.validate_seo_data(internal_df, 'internal')
        if not validation['valid']:
            st.error(f"❌ Internal data validation failed: {', '.join(validation['issues'])}")
            return None, None
        
        # Load GSC data
        st.info("Loading Google Search Console data...")
        gsc_df = DataLoader.load_csv(gsc_file, "GSC Report")
        
        # Validate GSC data
        gsc_validation = DataLoader.validate_seo_data(gsc_df, 'gsc')
        if not gsc_validation['valid']:
            st.error(f"❌ GSC data validation failed: {', '.join(gsc_validation['issues'])}")
            return None, None
        
        st.success("✅ Both files loaded successfully!")
        return internal_df, gsc_df
        
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        logger.error(f"File loading error: {str(e)}")
        return None, None

def main():
    """Main application function."""
    
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
        
        # Weights configuration
        st.markdown("**Similarity Weights:**")
        title_weight = st.slider("Title Similarity", 0.0, 1.0, 0.35, 0.05)
        h1_weight = st.slider("H1 Similarity", 0.0, 1.0, 0.25, 0.05)
        meta_weight = st.slider("Meta Description Similarity", 0.0, 1.0, 0.15, 0.05)
        keyword_weight = st.slider("Keyword Overlap", 0.0, 1.0, 0.15, 0.05)
        semantic_weight = st.slider("Semantic Similarity", 0.0, 1.0, 0.10, 0.05)
        
        # Validate weights sum to 1.0
        total_weight = title_weight + h1_weight + meta_weight + keyword_weight + semantic_weight
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_weight:.2f}, not 1.0. Adjusting...")
        
        weights = {
            'title': title_weight,
            'h1': h1_weight,
            'meta': meta_weight,
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
                        results = processor.process_complete_analysis(internal_df, gsc_df)
                        summary = processor.get_summary_stats(results)
                        
                        # Display results
                        st.success("✅ Analysis complete!")
                        
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
                        
                        # Display results
                        st.dataframe(
                            filtered_results[[
                                'URL_1', 'URL_2', 'Composite_Score', 
                                'Risk_Level', 'Same_Intent'
                            ]].head(50),
                            use_container_width=True
                        )
                        
                        # Download section
                        st.markdown("### 📥 Download Results")
                        
                        # Generate all reports
                        reports = ReportGenerator.export_all_formats(results, summary)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                label="📊 Download Full Analysis (CSV)",
                                data=reports['csv'],
                                file_name=reports['filenames']['csv'],
                                mime="text/csv"
                            )
                        
                        with col2:
                            st.download_button(
                                label="📄 Download Summary (TXT)",
                                data=reports['summary'],
                                file_name=reports['filenames']['summary'],
                                mime="text/plain"
                            )
                        
                        with col3:
                            st.download_button(
                                label="📋 Download Priority Actions (CSV)",
                                data=reports['priority_actions'],
                                file_name=reports['filenames']['priority'],
                                mime="text/csv"
                            )
                        
                        # Additional insights
                        if summary['high_risk'] > 0:
                            st.warning(f"⚠️ Found {summary['high_risk']} high-risk cannibalization issues!")
                            
                            priority_actions = ReportGenerator.generate_priority_actions(results)
                            if not priority_actions.empty:
                                st.markdown("### 🚨 Priority Actions")
                                st.dataframe(priority_actions, use_container_width=True)
                        
                        # Consolidation recommendations
                        if summary['high_risk'] > 2:
                            consolidation = ReportGenerator.generate_consolidation_plan(results)
                            if not consolidation.empty:
                                st.markdown("### 🔄 Content Consolidation Plan")
                                st.dataframe(consolidation, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
                        logger.error(f"Analysis error: {str(e)}")
                        st.exception(e)
    
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
            
            ### **Analysis Weights:**
            - **Title Similarity (35%)**: Most important for SEO
            - **H1 Similarity (25%)**: Key on-page signal
            - **Meta Description (15%)**: SERP presentation
            - **Keyword Overlap (15%)**: Actual search competition
            - **Semantic Similarity (10%)**: Overall content theme
            """)

# Footer
st.markdown("---")
st.markdown("🎯 **Enhanced Content Cannibalization Analyzer - FIXED VERSION**")

if __name__ == "__main__":
    main()
