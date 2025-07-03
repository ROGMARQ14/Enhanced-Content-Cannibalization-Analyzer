import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from datetime import datetime
import re
from collections import Counter
import json

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

# OpenAI API Key section
with st.sidebar:
    st.markdown("### 🤖 AI-Powered Analysis")
    openai_api_key = st.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        help="Enter your OpenAI API key to get AI-powered insights and recommendations"
    )
    
    if openai_api_key:
        st.success("✅ API Key provided - AI insights will be available")
    else:
        st.info("💡 Add OpenAI API key for detailed recommendations")

# Helper functions
def clean_text(text):
    """Clean and normalize text for comparison"""
    if pd.isna(text) or text is None:
        return ""
    return str(text).lower().strip()

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

def get_ai_insights(results_df, openai_api_key):
    """Generate AI-powered insights using OpenAI"""
    try:
        import openai
        
        # Configure OpenAI
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Prepare data for analysis
        high_risk = results_df[results_df['Risk_Level'] == 'High'].head(20)
        
        # Create a summary for the AI
        analysis_data = {
            "total_pairs": len(results_df),
            "high_risk_count": len(results_df[results_df['Risk_Level'] == 'High']),
            "medium_risk_count": len(results_df[results_df['Risk_Level'] == 'Medium']),
            "avg_composite_score": float(results_df['Composite_Score'].mean()),
            "top_issues": []
        }
        
        for _, row in high_risk.head(10).iterrows():
            analysis_data["top_issues"].append({
                "url1": row['URL_1'],
                "url2": row['URL_2'],
                "composite_score": float(row['Composite_Score']),
                "title_similarity": float(row['Title_Similarity']),
                "keyword_overlap": float(row['Keyword_Overlap']),
                "same_intent": row['Same_Intent']
            })
        
        # Create prompt
        prompt = f"""
        You are an expert SEO Strategist specializing in identifying and resolving content cannibalization issues. Your goal is to transform raw cannibalization data into a strategic, actionable executive report that a marketing team can implement to improve organic search performance.

Website Context:

Business Goal: [e.g., "Generate qualified B2B leads," "Drive e-commerce sales for consumer electronics," "Increase ad revenue through pageviews"]

Target Audience: [e.g., "CMOs at mid-size tech companies," "DIY home improvement enthusiasts," "Students seeking financial aid advice"]

Input Data:
Here is the content cannibalization analysis data, presented in JSON format. Each entry represents a keyword for which multiple URLs from our site are competing.

json
{json.dumps(analysis_data, indent=2)}
Task: Generate a comprehensive SEO report with the following sections:

1. Executive Summary:

Provide a 2-3 sentence overview of the key findings.

Assess the severity of the content cannibalization issue and its current impact on SEO performance (e.g., suppressed rankings, diluted authority, poor user experience).

Summarize the high-level strategic direction recommended in this report.

2. Top 5 Priority Actions:

Present the five most critical cannibalization issues to address first.

Prioritize based on a combination of keyword commercial value, search volume, and the potential for significant performance uplift.

Format this as a table with the following columns: Priority, Keyword, Competing URLs, Recommended Action, Rationale & Expected Outcome.

3. Content Consolidation & Pruning Plan:

Identify specific clusters of pages that should be merged or redirected.

For each cluster, recommend a single "canonical" (winner) URL to become the primary target.

List the "loser" URLs that should be 301 redirected or have their content merged into the winner.

Format this as a table with the following columns: Canonical URL (Winner), URLs to Consolidate/Redirect (Losers), Justification for Choice.

4. Quick Wins (Low-Effort, High-Impact):

List at least three immediate fixes that require minimal resources (e.g., can be done in under an hour).

Focus on actions like:

Optimizing title tags and meta descriptions to differentiate intent.

Adjusting internal links to signal the most important page to search engines.

Slightly de-optimizing a competing page for the target keyword.

5. Long-Term Strategy & Prevention:

Provide strategic recommendations to prevent future content cannibalization.

Include guidelines for the content creation workflow, such as:

A process for checking existing content before creating new articles.

A keyword-to-URL mapping strategy.

Best practices for internal linking to reinforce topical authority.

Tone and Formatting:

Tone: Be professional, data-driven, and authoritative.

Formatting: Use Markdown for clarity. Employ bolding for key terms and use tables where requested to ensure the report is scannable and actionable.

Constraint: Do not simply restate the data from the JSON. Your value is in the interpretation, strategic insights, and actionable recommendations.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert SEO consultant specializing in content strategy and cannibalization issues."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating AI insights: {str(e)}"

def detect_intent_from_title(title):
    """Detect content intent from title patterns"""
    if pd.isna(title) or title is None:
        return 'unknown'
    
    title_lower = str(title).lower()
    
    # Intent patterns
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

def is_clean_url(url):
    """Check if URL is clean (no parameters)"""
    if pd.isna(url) or not url:
        return False
    
    url_str = str(url)
    
    # Check for common parameter indicators
    parameter_chars = ['?', '&', '=', '#', '%5B', '%5D', 'utm_', 'fbclid', 'gclid']
    
    for char in parameter_chars:
        if char in url_str:
            return False
    
    return True

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
        
        # Aggregate queries by landing page
        url_queries = {}
        for _, row in gsc_df.iterrows():
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
        # Keep only HTTP/HTTPS URLs
        internal_df = internal_df[internal_df[url_column].str.contains('http', na=False)]
        internal_df = internal_df[~internal_df[url_column].str.contains('redirect', na=False, case=False)]
        
        # Remove URLs with parameters
        internal_df['is_clean'] = internal_df[url_column].apply(is_clean_url)
        excluded_urls = len(internal_df[~internal_df['is_clean']])
        internal_df = internal_df[internal_df['is_clean']].drop('is_clean', axis=1)
        
        st.metric("Valid URLs for analysis", len(internal_df))
        if excluded_urls > 0:
            st.warning(f"Excluded {excluded_urls} URLs with parameters (?, &, #, etc.)")
    
    # Process similarity analysis
    with st.spinner('Calculating multi-dimensional similarity...'):
        results = []
        urls = internal_df[url_column].tolist()
        n = len(urls)
        
        # Prepare data for each URL
        url_data = {}
        for idx, row in internal_df.iterrows():
            url = row[url_column]
            url_data[url] = {
                'title': clean_text(row.get(title_column, '')) if title_column else '',
                'h1': clean_text(row.get(h1_column, '')) if h1_column else '',
                'meta': clean_text(row.get(meta_column, '')) if meta_column else '',
                'queries': [q['query'] for q in url_queries.get(url, [])],
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
        for i in range(n):
            for j in range(i + 1, n):
                url1 = urls[i]
                url2 = urls[j]
                
                # Skip if same domain path
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
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Composite_Score', ascending=False)
    
    st.success(f"✅ Analysis complete! Analyzed {len(results_df):,} URL pairs")
    
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
    if openai_api_key:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚨 High Risk Pairs", "📊 All Results", "🎯 By Intent", "🤖 AI Insights", "📥 Download"])
    else:
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
    
    # AI Insights tab (if API key provided)
    if openai_api_key:
        with tab4:
            st.markdown("### 🤖 AI-Powered Analysis & Recommendations")
            
            if st.button("Generate AI Insights", type="primary"):
                with st.spinner("Analyzing your content with AI... This may take a moment."):
                    ai_insights = get_ai_insights(results_df, openai_api_key)
                    
                    if "Error" in ai_insights:
                        st.error(ai_insights)
                    else:
                        # Display AI insights in a nice format
                        st.markdown(ai_insights)
                        
                        # Option to download AI insights
                        st.download_button(
                            label="📥 Download AI Insights",
                            data=ai_insights,
                            file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
            
            st.info("""
            💡 **What the AI analyzes:**
            - Your top cannibalization issues
            - Content consolidation opportunities
            - Quick wins for immediate impact
            - Long-term content strategy recommendations
            """)
        
        # Download tab is now tab5 when AI is enabled
        download_tab = tab5
    else:
        # Download tab is tab4 when AI is not enabled
        download_tab = tab4
    
    with download_tab:
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

OVERVIEW:
- Total URL pairs analyzed: {len(results_df):,}
- High risk pairs: {len(results_df[results_df['Risk_Level'] == 'High'])}
- Medium risk pairs: {len(results_df[results_df['Risk_Level'] == 'Medium'])}
- Average composite score: {results_df['Composite_Score'].mean():.1f}%

TOP CANNIBALIZATION RISKS:
"""
            for idx, row in results_df.head(10).iterrows():
                summary += f"\n{row['URL_1']}\n  vs {row['URL_2']}\n  Score: {row['Composite_Score']}% | Risk: {row['Risk_Level']}\n"
            
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
    
    3. **OpenAI API Key** (Optional)
       - Add in the sidebar for AI-powered insights
       - Get your key at: https://platform.openai.com/api-keys
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
