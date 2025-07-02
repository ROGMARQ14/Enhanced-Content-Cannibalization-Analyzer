import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import re

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

✅ **New**: Automatic URL normalization to prevent false positives from parameter variations
""")

# Helper functions
def clean_text(text):
    """Clean and normalize text for comparison"""
    if pd.isna(text) or text is None:
        return ""
    return str(text).lower().strip()

def normalize_url(url):
    """
    Normalize URL by removing parameters and fragments that can cause false positives
    
    Args:
        url (str): Original URL
    
    Returns:
        str: Normalized URL without parameters, fragments, or tracking codes
    """
    if pd.isna(url) or not url:
        return ""
    
    url = str(url).strip()
    
    # Handle URL encoding issues (like %20 for spaces)
    try:
        from urllib.parse import unquote
        url = unquote(url)
    except:
        pass
    
    # Remove common tracking parameters and fragments
    try:
        parsed = urlparse(url)
        
        # Remove query parameters (everything after ?)
        # This removes &, ?, = parameters automatically
        clean_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            '',  # Remove params
            '',  # Remove query
            ''   # Remove fragment (# parameters)
        ))
        
        # Clean up the path
        clean_url = clean_url.rstrip('/')  # Remove trailing slashes
        clean_url = clean_url.strip()      # Remove any whitespace
        
        # Convert to lowercase for consistency
        clean_url = clean_url.lower()
        
        return clean_url
        
    except Exception:
        # Fallback: manual cleaning for malformed URLs
        url = url.lower().strip()
        
        # Remove fragments (#)
        if '#' in url:
            url = url.split('#')[0]
        
        # Remove query parameters (?)
        if '?' in url:
            url = url.split('?')[0]
        
        # Remove trailing slash
        url = url.rstrip('/')
        
        return url

def deduplicate_urls(df, url_column):
    """
    Remove duplicate URLs after normalization, keeping the first occurrence
    
    Args:
        df (pd.DataFrame): DataFrame with URLs
        url_column (str): Name of the URL column
    
    Returns:
        tuple: (cleaned_df, removed_count, duplicate_report)
    """
    original_count = len(df)
    
    # Add normalized URL column
    df['normalized_url'] = df[url_column].apply(normalize_url)
    
    # Track duplicates for reporting
    duplicate_groups = df.groupby('normalized_url').agg({
        url_column: list,
        'normalized_url': 'first'
    }).reset_index(drop=True)
    
    # Find groups with multiple URLs (duplicates)
    duplicates = duplicate_groups[duplicate_groups[url_column].apply(len) > 1]
    
    # Keep only the first occurrence of each normalized URL
    df_cleaned = df.drop_duplicates(subset=['normalized_url'], keep='first')
    
    # Create duplicate report
    duplicate_report = []
    for _, group in duplicates.iterrows():
        original_urls = group[url_column]
        normalized = group['normalized_url']
        duplicate_report.append({
            'normalized_url': normalized,
            'original_urls': original_urls,
            'count': len(original_urls)
        })
    
    removed_count = original_count - len(df_cleaned)
    
    return df_cleaned, removed_count, duplicate_report

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
    """Detect content intent from title patterns"""
    if title is None or pd.isna(title):
        return 'unknown'
    
    title_lower = str(title).strip().lower()
    if not title_lower:
        return 'unknown'
    
    # Intent patterns
    try:
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
        # Load internal HTML data
        try:
            # Try semicolon delimiter first (common in SEO tools)
            internal_df = pd.read_csv(internal_file, sep=';', encoding='utf-8')
        except:
            # Fallback to comma delimiter
            internal_df = pd.read_csv(internal_file, encoding='utf-8')
        
        # Load GSC data
        gsc_df = pd.read_csv(gsc_file, encoding='utf-8')
    
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
        
        # Normalize GSC URLs as well
        if 'Landing Page' in gsc_df.columns:
            gsc_df['normalized_landing_page'] = gsc_df['Landing Page'].apply(normalize_url)
        elif 'URL' in gsc_df.columns:
            gsc_df['normalized_landing_page'] = gsc_df['URL'].apply(normalize_url)
        
        # Aggregate queries by normalized landing page
        url_queries = {}
        for _, row in gsc_df.iterrows():
            url = row.get('normalized_landing_page', normalize_url(row.get('Landing Page', row.get('URL', ''))))
            query = row.get('Query', '')
            clicks = row.get('Clicks', row.get(' Clicks ', 0))
            
            if url and query:
                if url not in url_queries:
                    url_queries[url] = []
                url_queries[url].append({
                    'query': query,
                    'clicks': clicks
                })

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

    # Filter for valid URLs and normalize
    if url_column:
        # Keep only HTTP/HTTPS URLs
        internal_df = internal_df[internal_df[url_column].str.contains('http', na=False)]
        internal_df = internal_df[~internal_df[url_column].str.contains('redirect', na=False, case=False)]
        
        # URL Normalization and Deduplication
        with st.spinner('🧹 Normalizing URLs and removing duplicates...'):
            cleaned_df, removed_count, duplicate_report = deduplicate_urls(internal_df, url_column)
        
        # Show deduplication results
        if removed_count > 0:
            st.warning(f"🧹 **URL Cleaning Results**: Removed {removed_count} duplicate URLs with parameters")
            
            with st.expander(f"📋 View {len(duplicate_report)} duplicate groups"):
                for i, dup in enumerate(duplicate_report[:10]):  # Show first 10
                    st.write(f"**Group {i+1}:** `{dup['normalized_url']}`")
                    for orig_url in dup['original_urls']:
                        st.write(f"  - {orig_url}")
                    st.write("---")
                
                if len(duplicate_report) > 10:
                    st.info(f"... and {len(duplicate_report) - 10} more duplicate groups")
        else:
            st.success("✅ No URL duplicates found - your data is already clean!")
        
        # Use cleaned data for analysis
        internal_df = cleaned_df
        
        st.metric("Valid URLs for analysis (after deduplication)", len(internal_df))

        # Process similarity analysis
        with st.spinner('Calculating multi-dimensional similarity...'):
            results = []
            # Use normalized URLs for the analysis
            urls = internal_df['normalized_url'].tolist()
            original_urls = internal_df[url_column].tolist()  # Keep originals for display
            n = len(urls)

            # Prepare data for each URL
            url_data = {}
            for idx, row in internal_df.iterrows():
                normalized_url = row['normalized_url']
                original_url = row[url_column]
                
                url_data[normalized_url] = {
                    'original_url': original_url,  # Store original for display
                    'title': clean_text(row.get(title_column, '')) if title_column else '',
                    'h1': clean_text(row.get(h1_column, '')) if h1_column else '',
                    'meta': clean_text(row.get(meta_column, '')) if meta_column else '',
                    'queries': [q['query'] for q in url_queries.get(normalized_url, [])],
                    'intent': detect_intent_from_title(row.get(title_column, '')) if title_column else 'unknown',
                    'embedding': row.get(embedding_column, '') if embedding_column else ''
                }

            # Calculate similarity matrices
            titles = [url_data[url]['title'] for url in urls]
            h1s = [url_data[url]['h1'] for url in urls]
            metas = [url_data[url]['meta'] for url in urls]

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

            # Calculate results for each pair
            progress_bar = st.progress(0)
            total_pairs = (n * (n - 1)) // 2
            pair_count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    url1 = urls[i]
                    url2 = urls[j]

                    # Skip if same URL (shouldn't happen after deduplication, but safety check)
                    if url1 == url2:
                        continue

                    # Get similarities
                    title_sim = title_sim_matrix[i, j] * 100
                    h1_sim = h1_sim_matrix[i, j] * 100
                    meta_sim = meta_sim_matrix[i, j] * 100

                    # Keyword overlap
                    queries1 = url_data[url1]['queries']
                    queries2 = url_data[url2]['queries']
                    keyword_overlap = calculate_keyword_overlap(queries1, queries2) * 100

                    # Embedding similarity
                    embedding_sim = 0
                    if embedding_sim_matrix is not None and i in valid_embedding_indices and j in valid_embedding_indices:
                        idx_i = valid_embedding_indices.index(i)
                        idx_j = valid_embedding_indices.index(j)
                        embedding_sim = embedding_sim_matrix[idx_i, idx_j] * 100

                    # Calculate composite score
                    # Weighted heavily towards title and H1 for SEO impact
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
                        'URL_1': url_data[url1]['original_url'],  # Use original URLs for display
                        'URL_2': url_data[url2]['original_url'],  # Use original URLs for display
                        'Normalized_URL_1': url1,  # Keep normalized for debugging
                        'Normalized_URL_2': url2,  # Keep normalized for debugging
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

            progress_bar.progress(1.0)
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Composite_Score', ascending=False)

            st.success(f"✅ Analysis complete! Analyzed {len(results_df):,} URL pairs (after removing duplicates)")

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
                
                # Prepare download data
                download_df = results_df.copy()
                
                # Convert to CSV
                csv_buffer = io.StringIO()
                download_df.to_csv(csv_buffer, index=False)
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
                    # Create summary report
                    summary = f"""
CONTENT CANNIBALIZATION ANALYSIS SUMMARY
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

URL CLEANING RESULTS:
- Original URLs: {len(internal_df) + removed_count}
- Duplicate URLs removed: {removed_count}
- Clean URLs analyzed: {len(internal_df)}

ANALYSIS OVERVIEW:
- Total URL pairs analyzed: {len(results_df):,}
- High risk pairs: {len(results_df[results_df['Risk_Level'] == 'High'])}
- Medium risk pairs: {len(results_df[results_df['Risk_Level'] == 'Medium'])}
- Average composite score: {results_df['Composite_Score'].mean():.1f}%

TOP CANNIBALIZATION RISKS:
"""
                    for idx, row in results_df.head(10).iterrows():
                        summary += f"\n{row['URL_1']}\n vs {row['URL_2']}\n Score: {row['Composite_Score']}% | Risk: {row['Risk_Level']}\n"

                    st.download_button(
                        label="📄 Download Summary Report (TXT)",
                        data=summary,
                        file_name=f"cannibalization_summary_{timestamp}.txt",
                        mime="text/plain"
                    )

            # Insights section
            st.markdown("### 💡 Key Insights")
            
            # Calculate insights
            very_high_title_sim = len(results_df[results_df['Title_Similarity'] > 90])
            high_keyword_overlap = len(results_df[results_df['Keyword_Overlap'] > 70])
            insights = []

            if removed_count > 0:
                insights.append(f"🧹 Removed {removed_count} duplicate URLs with parameters - analysis is now more accurate")

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

        **🧹 URL Normalization (NEW):**
        - Automatically removes URL parameters (&, ?, =, #)
        - Prevents false positives from tracking codes
        - Deduplicates similar URLs before analysis

        **Risk Levels:**
        - **High**: Same intent + >80% composite score
        - **Medium**: 60-80% composite score
        - **Low**: <60% composite score
        """)

    with st.expander("🛠️ URL Parameter Examples"):
        st.markdown("""
        **URLs that will be normalized:**
        
        ✅ **Before normalization:**
        - `https://example.com/page?utm_source=google&utm_medium=cpc`
        - `https://example.com/page#section1`
        - `https://example.com/page?ref=homepage&track=123`
        - `https://example.com/page%20` (with URL encoding)
        
        ✅ **After normalization:**
        - `https://example.com/page`
        - `https://example.com/page`
        - `https://example.com/page`
        - `https://example.com/page`
        
        **Result:** Only one comparison instead of false positives between similar URLs.
        """)

# Footer
st.markdown("---")
st.markdown("🎯 Enhanced Content Cannibalization Analyzer - Built for SEO Professionals")
