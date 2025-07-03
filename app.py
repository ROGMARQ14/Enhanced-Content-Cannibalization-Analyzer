import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from datetime import datetime
import re
from collections import Counter
from urllib.parse import urlparse, unquote

# Set page config
st.set_page_config(
    page_title="Enhanced Content Cannibalization Analyzer",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Enhanced Content Cannibalization Analyzer")
st.markdown("""
This advanced tool analyzes content similarity across multiple dimensions:
- **Title & H1 similarity** (weighted heavily for SEO impact)
- **Meta description similarity** (SERP competition)
- **Keyword/query overlap** (search intent matching)
- **Semantic similarity** (overall content theme)
- **Composite cannibalization score** (smart weighted average)
""")

# Helper functions
def clean_text(text):
    """Clean and normalize text for comparison"""
    if pd.isna(text) or text is None:
        return ""
    return str(text).lower().strip()

def safe_string_format(text, max_length=100):
    """
    Safely format strings for reports, handling special characters and length
    
    Args:
        text: Input text that might contain special characters
        max_length: Maximum length before truncation
    
    Returns:
        str: Safe, formatted string
    """
    if pd.isna(text) or text is None:
        return "N/A"
    
    try:
        # Convert to string and handle URL encoding
        text_str = str(text)
        
        # Decode URL encoding if present
        try:
            text_str = unquote(text_str)
        except:
            pass
        
        # Remove or escape problematic characters
        text_str = text_str.replace('{', '{{').replace('}', '}}')
        
        # Truncate if too long
        if len(text_str) > max_length:
            text_str = text_str[:max_length-3] + "..."
        
        # Remove newlines and tabs that could break formatting
        text_str = text_str.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        
        # Remove excessive whitespace
        text_str = ' '.join(text_str.split())
        
        return text_str
        
    except Exception:
        return "Error formatting text"

def generate_safe_summary_report(results_df):
    """
    Generate a summary report with robust error handling
    
    Args:
        results_df: DataFrame with analysis results
    
    Returns:
        str: Safely formatted summary report
    """
    try:
        # Calculate basic statistics with error handling
        total_pairs = len(results_df) if not results_df.empty else 0
        high_risk_count = len(results_df[results_df['Risk_Level'] == 'High']) if not results_df.empty else 0
        medium_risk_count = len(results_df[results_df['Risk_Level'] == 'Medium']) if not results_df.empty else 0
        avg_score = results_df['Composite_Score'].mean() if not results_df.empty else 0
        
        # Build report with safe string formatting
        report_lines = [
            "CONTENT CANNIBALIZATION ANALYSIS SUMMARY",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "OVERVIEW:",
            f"- Total URL pairs analyzed: {total_pairs:,}",
            f"- High risk pairs: {high_risk_count}",
            f"- Medium risk pairs: {medium_risk_count}",
            f"- Average composite score: {avg_score:.1f}%",
            "",
            "TOP CANNIBALIZATION RISKS:",
            "-" * 30
        ]
        
        # Add top issues with safe formatting
        if not results_df.empty:
            top_issues = results_df.head(10)
            for idx, (_, row) in enumerate(top_issues.iterrows(), 1):
                try:
                    url1 = safe_string_format(row.get('URL_1', 'N/A'), 80)
                    url2 = safe_string_format(row.get('URL_2', 'N/A'), 80)
                    score = row.get('Composite_Score', 0)
                    risk = row.get('Risk_Level', 'Unknown')
                    
                    report_lines.extend([
                        f"",
                        f"ISSUE #{idx}:",
                        f"URL 1: {url1}",
                        f"URL 2: {url2}",
                        f"Score: {score}% | Risk: {risk}",
                        "-" * 30
                    ])
                except Exception as e:
                    report_lines.extend([
                        f"",
                        f"ISSUE #{idx}: Error processing data - {str(e)[:50]}",
                        "-" * 30
                    ])
        else:
            report_lines.append("No issues found in the analysis.")
        
        # Add recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "1. Review high-risk pairs for content consolidation opportunities",
            "2. Differentiate titles and H1s for similar content",
            "3. Optimize meta descriptions to reduce SERP competition",
            "4. Consider internal linking strategy for competing pages",
            "",
            "Generated by Enhanced Content Cannibalization Analyzer"
        ])
        
        return "\n".join(report_lines)
        
    except Exception as e:
        # Fallback report if everything fails
        return f"""
CONTENT CANNIBALIZATION ANALYSIS SUMMARY
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

ERROR: Unable to generate full report due to data formatting issues.
Error details: {str(e)[:100]}

Please check your data for special characters or contact support.

Basic Statistics:
- Total pairs processed: {len(results_df) if 'results_df' in locals() else 0}
- Analysis completed successfully

Generated by Enhanced Content Cannibalization Analyzer
"""

def calculate_text_similarity(texts1, texts2):
    """Calculate cosine similarity between text pairs using TF-IDF"""
    if not texts1 or not texts2:
        return np.zeros((len(texts1), len(texts2)))

    # Combine all texts for vectorization
    all_texts = texts1 + texts2
    # Handle empty texts
    all_texts = [text if text else "empty" for text in all_texts]

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        # Split back into two sets
        matrix1 = tfidf_matrix[:len(texts1)]
        matrix2 = tfidf_matrix[len(texts1):]
        # Calculate similarity
        return cosine_similarity(matrix1, matrix2)
    except:
        return np.zeros((len(texts1), len(texts2)))

def calculate_keyword_overlap(keywords1, keywords2):
    """Calculate Jaccard similarity between keyword sets"""
    if not keywords1 or not keywords2:
        return 0.0
    
    set1 = set(keywords1)
    set2 = set(keywords2)
    
    if not set1 or not set2:
        return 0.0
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

def detect_intent_from_title(title):
    """Detect content intent from title patterns with safe error handling"""
    if title is None or pd.isna(title):
        return 'unknown'
    
    try:
        title_lower = str(title).lower().strip()
        if not title_lower:
            return 'unknown'
        
        # Intent patterns - safe string operations
        if any(word in title_lower for word in ['how to', 'guide', 'tutorial', 'step']):
            return 'how-to'
        elif any(word in title_lower for word in ['best', 'top', 'review', 'vs', 'versus', 'comparison']):
            return 'comparison'
        elif any(word in title_lower for word in ['what is', 'definition', 'meaning', 'explained']):
            return 'informational'
        elif any(word in title_lower for word in ['buy', 'price', 'cost', 'cheap', 'deal', 'discount']):
            return 'transactional'
        elif '?' in title_lower:
            return 'question'
        else:
            return 'general'
    except Exception:
        return 'unknown'

# File upload section
col1, col2 = st.columns(2)

with col1:
    internal_file = st.file_uploader(
        "Upload Internal HTML Report (CSV)",
        type=['csv'],
        help="Should contain URLs, titles, H1s, meta descriptions, etc."
    )

with col2:
    gsc_file = st.file_uploader(
        "Upload GSC Report (CSV)",
        type=['csv'],
        help="Should contain queries and landing pages"
    )

if internal_file and gsc_file:
    with st.spinner('Loading files...'):
        # Load internal HTML data with error handling
        try:
            # Try semicolon delimiter first (common in SEO tools)
            internal_df = pd.read_csv(internal_file, sep=';', encoding='utf-8')
        except:
            try:
                # Fallback to comma delimiter
                internal_df = pd.read_csv(internal_file, encoding='utf-8')
            except Exception as e:
                st.error(f"❌ Error loading Internal HTML file: {str(e)}")
                st.stop()
        
        # Load GSC data with error handling
        try:
            gsc_df = pd.read_csv(gsc_file, encoding='utf-8')
        except Exception as e:
            st.error(f"❌ Error loading GSC file: {str(e)}")
            st.stop()
    
    st.success(f"✅ Files loaded successfully!")
    
    # Display basic info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total URLs (Internal HTML)", len(internal_df))
    with col2:
        st.metric("Total Queries (GSC)", len(gsc_df))

    # Process GSC data to aggregate queries by URL
    with st.spinner('Processing GSC data...'):
        # Clean column names
        gsc_df.columns = gsc_df.columns.str.strip()
        
        # Aggregate queries by landing page
        url_queries = {}
        for _, row in gsc_df.iterrows():
            try:
                url = row.get('Landing Page', row.get('URL', ''))
                query = row.get('Query', '')
                clicks = row.get('Clicks', row.get(' Clicks ', 0))
                
                if url and query:
                    if url not in url_queries:
                        url_queries[url] = []
                    url_queries[url].append({
                        'query': query,
                        'clicks': clicks
                    })
            except Exception:
                continue  # Skip problematic rows

    # Identify column names dynamically
    with st.spinner('Analyzing data structure...'):
        # Find URL column
        url_column = None
        for col in internal_df.columns:
            if 'address' in col.lower() or 'url' in col.lower():
                url_column = col
                break

        # Find other relevant columns
        title_column = next((col for col in internal_df.columns if 'Title 1' in col or 'title' in col.lower()), None)
        h1_column = next((col for col in internal_df.columns if 'H1-1' in col or 'h1' in col.lower()), None)
        meta_column = next((col for col in internal_df.columns if 'Meta Description' in col or 'meta description' in col.lower()), None)
        embedding_column = next((col for col in internal_df.columns if 'embedding' in col.lower()), None)

        st.info(f"Detected columns - URL: {url_column}, Title: {title_column}, H1: {h1_column}")

    # Filter for valid URLs
    if url_column:
        # Keep only HTTP/HTTPS URLs with error handling
        try:
            internal_df = internal_df[internal_df[url_column].str.contains('http', na=False)]
            internal_df = internal_df[~internal_df[url_column].str.contains('redirect', na=False, case=False)]
            st.metric("Valid URLs for analysis", len(internal_df))
        except Exception as e:
            st.error(f"❌ Error filtering URLs: {str(e)}")
            st.stop()

        # Process similarity analysis
        with st.spinner('Calculating multi-dimensional similarity...'):
            results = []
            urls = internal_df[url_column].tolist()
            n = len(urls)

            # Prepare data for each URL with error handling
            url_data = {}
            for idx, row in internal_df.iterrows():
                try:
                    url = row[url_column]
                    url_data[url] = {
                        'title': clean_text(row.get(title_column, '')) if title_column else '',
                        'h1': clean_text(row.get(h1_column, '')) if h1_column else '',
                        'meta': clean_text(row.get(meta_column, '')) if meta_column else '',
                        'queries': [q['query'] for q in url_queries.get(url, [])],
                        'intent': detect_intent_from_title(row.get(title_column, '')) if title_column else 'unknown',
                        'embedding': row.get(embedding_column, '') if embedding_column else ''
                    }
                except Exception:
                    continue  # Skip problematic rows

            # Calculate similarity matrices
            titles = [url_data[url]['title'] for url in urls if url in url_data]
            h1s = [url_data[url]['h1'] for url in urls if url in url_data]
            metas = [url_data[url]['meta'] for url in urls if url in url_data]

            # Text-based similarities
            title_sim_matrix = calculate_text_similarity(titles, titles)
            h1_sim_matrix = calculate_text_similarity(h1s, h1s)
            meta_sim_matrix = calculate_text_similarity(metas, metas)

            # Embedding similarity (if available)
            embedding_sim_matrix = None
            if embedding_column:
                embeddings = []
                valid_embedding_indices = []
                
                for idx, url in enumerate(urls):
                    try:
                        if url in url_data:
                            embedding_str = url_data[url]['embedding']
                            if embedding_str and not pd.isna(embedding_str):
                                embedding = np.array([float(x) for x in str(embedding_str).split(',')])
                                embeddings.append(embedding)
                                valid_embedding_indices.append(idx)
                    except:
                        pass

                if embeddings:
                    embeddings = np.array(embeddings)
                    embedding_sim_matrix = cosine_similarity(embeddings)

            # Calculate results for each pair with progress tracking
            progress_bar = st.progress(0)
            total_pairs = (len(urls) * (len(urls) - 1)) // 2
            pair_count = 0
            
            for i in range(len(urls)):
                for j in range(i + 1, len(urls)):
                    try:
                        url1 = urls[i]
                        url2 = urls[j]

                        # Skip if URLs not in processed data
                        if url1 not in url_data or url2 not in url_data:
                            continue

                        # Skip if same URL
                        if url1 == url2:
                            continue

                        # Get similarities safely
                        title_sim = title_sim_matrix[i, j] * 100 if i < len(title_sim_matrix) and j < len(title_sim_matrix[0]) else 0
                        h1_sim = h1_sim_matrix[i, j] * 100 if i < len(h1_sim_matrix) and j < len(h1_sim_matrix[0]) else 0
                        meta_sim = meta_sim_matrix[i, j] * 100 if i < len(meta_sim_matrix) and j < len(meta_sim_matrix[0]) else 0

                        # Keyword overlap
                        queries1 = url_data[url1]['queries']
                        queries2 = url_data[url2]['queries']
                        keyword_overlap = calculate_keyword_overlap(queries1, queries2) * 100

                        # Embedding similarity
                        embedding_sim = 0
                        if embedding_sim_matrix is not None and i in valid_embedding_indices and j in valid_embedding_indices:
                            try:
                                idx_i = valid_embedding_indices.index(i)
                                idx_j = valid_embedding_indices.index(j)
                                embedding_sim = embedding_sim_matrix[idx_i, idx_j] * 100
                            except:
                                embedding_sim = 0

                        # Calculate composite score
                        weights = {
                            'title': 0.35,
                            'h1': 0.25,
                            'meta': 0.15,
                            'keyword': 0.15,
                            'embedding': 0.10
                        }

                        composite_score = (
                            weights['title'] * title_sim +
                            weights['h1'] * h1_sim +
                            weights['meta'] * meta_sim +
                            weights['keyword'] * keyword_overlap +
                            weights['embedding'] * embedding_sim
                        )

                        # Intent matching
                        intent1 = url_data[url1]['intent']
                        intent2 = url_data[url2]['intent']
                        same_intent = intent1 == intent2

                        results.append({
                            'URL_1': url1,
                            'URL_2': url2,
                            'Composite_Score': round(composite_score, 1),
                            'Title_Similarity': round(title_sim, 1),
                            'H1_Similarity': round(h1_sim, 1),
                            'Meta_Similarity': round(meta_sim, 1),
                            'Keyword_Overlap': round(keyword_overlap, 1),
                            'Embedding_Similarity': round(embedding_sim, 1),
                            'Intent_1': intent1,
                            'Intent_2': intent2,
                            'Same_Intent': same_intent,
                            'Risk_Level': 'High' if composite_score > 80 and same_intent else 'Medium' if composite_score > 60 else 'Low'
                        })
                        
                        # Update progress
                        pair_count += 1
                        if pair_count % 100 == 0:
                            progress_bar.progress(min(pair_count / total_pairs, 1.0))
                            
                    except Exception:
                        continue  # Skip problematic pairs

            progress_bar.progress(1.0)
            
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Composite_Score', ascending=False)
                st.success(f"✅ Analysis complete! Analyzed {len(results_df):,} URL pairs")
            else:
                st.error("❌ No valid results generated. Please check your data.")
                st.stop()

            # Summary statistics
            st.markdown("### 📊 Cannibalization Risk Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                high_risk = len(results_df[results_df['Risk_Level'] == 'High'])
                st.metric("High Risk Pairs", high_risk, help="Same intent + >80% similarity")

            with col2:
                medium_risk = len(results_df[results_df['Risk_Level'] == 'Medium'])
                st.metric("Medium Risk Pairs", medium_risk, help="60-80% similarity")

            with col3:
                avg_composite = results_df['Composite_Score'].mean()
                st.metric("Avg Composite Score", f"{avg_composite:.1f}%")

            with col4:
                same_intent_high = len(results_df[(results_df['Same_Intent']) & (results_df['Composite_Score'] > 70)])
                st.metric("Same Intent >70%", same_intent_high)

            # Detailed Analysis Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["🚨 High Risk Pairs", "📊 All Results", "🎯 By Intent", "📥 Download"])

            with tab1:
                st.markdown("### High Risk Cannibalization Candidates")
                high_risk_df = results_df[results_df['Risk_Level'] == 'High'].head(50)
                
                if len(high_risk_df) > 0:
                    # Format for display
                    display_df = high_risk_df[['URL_1', 'URL_2', 'Composite_Score', 'Title_Similarity',
                                             'H1_Similarity', 'Keyword_Overlap', 'Intent_1', 'Intent_2']].copy()
                    
                    # Add percentage signs
                    for col in ['Composite_Score', 'Title_Similarity', 'H1_Similarity', 'Keyword_Overlap']:
                        display_df[col] = display_df[col].astype(str) + '%'
                    
                    st.dataframe(display_df, use_container_width=True, height=400)
                else:
                    st.info("No high-risk cannibalization pairs found. Your content is well-differentiated!")

            with tab2:
                st.markdown("### All URL Pairs Analysis")
                
                # Filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_composite = st.slider("Min Composite Score %", 0, 100, 50)
                with col2:
                    risk_filter = st.multiselect("Risk Levels", ['High', 'Medium', 'Low'], default=['High', 'Medium'])
                with col3:
                    intent_filter = st.checkbox("Same intent only", value=False)

                # Apply filters
                filtered_df = results_df[
                    (results_df['Composite_Score'] >= min_composite) &
                    (results_df['Risk_Level'].isin(risk_filter))
                ]

                if intent_filter:
                    filtered_df = filtered_df[filtered_df['Same_Intent']]

                # Display
                st.metric("Filtered Results", len(filtered_df))
                display_cols = ['URL_1', 'URL_2', 'Composite_Score', 'Title_Similarity',
                               'H1_Similarity', 'Meta_Similarity', 'Keyword_Overlap', 'Risk_Level']
                display_df = filtered_df[display_cols].head(100).copy()
                
                for col in ['Composite_Score', 'Title_Similarity', 'H1_Similarity', 'Meta_Similarity', 'Keyword_Overlap']:
                    display_df[col] = display_df[col].astype(str) + '%'
                
                st.dataframe(display_df, use_container_width=True, height=400)

            with tab3:
                st.markdown("### Analysis by Intent Type")
                
                # Intent distribution
                intent_counts = pd.concat([
                    results_df['Intent_1'].value_counts(),
                    results_df['Intent_2'].value_counts()
                ]).groupby(level=0).sum().sort_values(ascending=False)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("**Intent Distribution:**")
                    for intent, count in intent_counts.items():
                        st.write(f"- {intent}: {count}")

                with col2:
                    # Same intent high similarity
                    same_intent_issues = results_df[
                        (results_df['Same_Intent']) &
                        (results_df['Composite_Score'] > 70)
                    ].groupby('Intent_1').size().sort_values(ascending=False)
                    
                    st.markdown("**Cannibalization by Intent:**")
                    for intent, count in same_intent_issues.items():
                        st.write(f"- {intent}: {count} high-similarity pairs")

            with tab4:
                st.markdown("### Download Complete Analysis")
                
                # Prepare download data with error handling
                try:
                    download_df = results_df.copy()
                    
                    # Convert to CSV with safe encoding
                    csv_buffer = io.StringIO()
                    download_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                    csv_data = csv_buffer.getvalue()

                    # Generate filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"enhanced_cannibalization_analysis_{timestamp}.csv"

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Full Analysis (CSV)",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv",
                            help="Contains all URL pairs with individual similarity scores and risk levels"
                        )

                    with col2:
                        # Create safe summary report
                        try:
                            summary = generate_safe_summary_report(results_df)
                            
                            st.download_button(
                                label="📄 Download Summary Report (TXT)",
                                data=summary,
                                file_name=f"cannibalization_summary_{timestamp}.txt",
                                mime="text/plain",
                                help="Safe summary report with error handling"
                            )
                        except Exception as e:
                            st.error(f"❌ Error generating summary report: {str(e)}")
                            
                            # Provide a basic fallback summary
                            basic_summary = f"""
Content Cannibalization Analysis Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Total pairs analyzed: {len(results_df)}
High risk pairs: {len(results_df[results_df['Risk_Level'] == 'High'])}
Medium risk pairs: {len(results_df[results_df['Risk_Level'] == 'Medium'])}

Note: Detailed report generation encountered formatting issues.
Please use the CSV download for complete data.
"""
                            st.download_button(
                                label="📄 Download Basic Summary (TXT)",
                                data=basic_summary,
                                file_name=f"basic_summary_{timestamp}.txt",
                                mime="text/plain"
                            )
                
                except Exception as e:
                    st.error(f"❌ Error preparing download files: {str(e)}")

            # Insights section
            st.markdown("### 💡 Key Insights")
            
            # Calculate insights with error handling
            try:
                very_high_title_sim = len(results_df[results_df['Title_Similarity'] > 90])
                high_keyword_overlap = len(results_df[results_df['Keyword_Overlap'] > 70])
                insights = []

                if very_high_title_sim > 0:
                    insights.append(f"⚠️ Found {very_high_title_sim} URL pairs with >90% title similarity - consider differentiating titles")

                if high_keyword_overlap > 0:
                    insights.append(f"🎯 {high_keyword_overlap} URL pairs target very similar keywords - review search intent")

                if len(results_df[results_df['Risk_Level'] == 'High']) > 10:
                    insights.append("🚨 Multiple high-risk cannibalization issues detected - prioritize content consolidation or differentiation")

                if insights:
                    for insight in insights:
                        st.info(insight)
                else:
                    st.success("✅ Your content appears well-differentiated with minimal cannibalization risk!")
                    
            except Exception:
                st.info("💡 Analysis completed successfully. Check the results in the tabs above.")

else:
    # Instructions
    st.info("""
    👆 Please upload both files to begin analysis:

    1. **Internal HTML Report** (from Screaming Frog or similar)
       - Must contain: URLs, Titles, H1s, Meta Descriptions
       - Optional: Embeddings, Word Count, etc.

    2. **GSC Report** (from Google Search Console)
       - Must contain: Queries and Landing Pages
       - Helps identify keyword cannibalization
    """)

    with st.expander("📖 How this enhanced analysis works"):
        st.markdown("""
        This tool goes beyond simple embedding similarity by analyzing:

        **1. Title Similarity (35% weight)**
        - Most important for SEO cannibalization
        - High similarity = competing for same SERP positions

        **2. H1 Similarity (25% weight)**
        - Key on-page signal for topic focus
        - Should be unique across pages

        **3. Meta Description Similarity (15% weight)**
        - Indicates SERP presentation overlap
        - Important for click-through rates

        **4. Keyword/Query Overlap (15% weight)**
        - Based on actual GSC data
        - Shows real search competition

        **5. Semantic Similarity (10% weight)**
        - Overall content theme alignment
        - Least weighted for niche sites

        **Risk Levels:**
        - **High**: Same intent + >80% composite score
        - **Medium**: 60-80% composite score
        - **Low**: <60% composite score
        """)

# Footer
st.markdown("---")
st.markdown("🎯 Enhanced Content Cannibalization Analyzer - Built for SEO Professionals")
