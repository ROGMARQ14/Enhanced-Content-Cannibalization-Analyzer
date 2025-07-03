import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from datetime import datetime
from urllib.parse import urlparse, unquote, urlunparse
import re

# Try to import OpenAI, but make it optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Enhanced Content Cannibalization Analyzer",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Enhanced Content Cannibalization Analyzer")
st.markdown("""
This advanced tool analyzes content similarity across multiple dimensions with **AI-powered insights**:
- **Title & H1 similarity** (weighted heavily for SEO impact)
- **Meta description similarity** (SERP competition)
- **Composite cannibalization score** (smart weighted average)
- **🤖 AI-Powered Analysis** (detailed strategic recommendations)

✅ **Features**: Automatic URL normalization + GPT-powered strategic insights
""")

# OpenAI Configuration Section
st.sidebar.header("🤖 AI Enhancement Settings")

def check_openai_setup():
    """Check if OpenAI is properly configured"""
    if not OPENAI_AVAILABLE:
        return False, "OpenAI library not installed. Run: pip install openai"
    
    # Check for API key in session state or secrets
    api_key = None
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        api_key = st.session_state.openai_api_key
    elif hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets['OPENAI_API_KEY']
        st.session_state.openai_api_key = api_key
    
    if not api_key:
        return False, "No API key provided"
    
    return True, api_key

# OpenAI API Key Input
if OPENAI_AVAILABLE:
    openai_status, openai_message = check_openai_setup()
    
    if not openai_status:
        st.sidebar.warning("⚠️ AI Analysis Disabled")
        st.sidebar.info(openai_message)
        
        # API Key input
        api_key_input = st.sidebar.text_input(
            "Enter OpenAI API Key (optional)",
            type="password",
            help="Required for AI-powered analysis and detailed reports"
        )
        
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
            st.sidebar.success("✅ API Key saved!")
            openai_status = True
    else:
        st.sidebar.success("✅ AI Analysis Enabled")
        if st.sidebar.button("🔄 Reset API Key"):
            if 'openai_api_key' in st.session_state:
                del st.session_state.openai_api_key
            st.rerun()

    # AI Settings
    if openai_status:
        st.sidebar.subheader("AI Analysis Options")
        enable_ai_analysis = st.sidebar.checkbox("Enable AI Analysis", value=True)
        gpt_model = st.sidebar.selectbox(
            "GPT Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0,
            help="GPT-4 provides better analysis but costs more"
        )
    else:
        enable_ai_analysis = False
        gpt_model = "gpt-3.5-turbo"
else:
    st.sidebar.error("❌ OpenAI library not available")
    st.sidebar.info("To enable AI features, install: `pip install openai`")
    enable_ai_analysis = False
    gpt_model = "gpt-3.5-turbo"

# Helper functions
def clean_text(text):
    """Clean and normalize text for comparison"""
    if pd.isna(text) or text is None:
        return ""
    return str(text).lower().strip()

def normalize_url(url):
    """Normalize URL by removing parameters and fragments"""
    if pd.isna(url) or not url:
        return ""
    
    url = str(url).strip()
    
    # Handle URL encoding issues (like %20)
    try:
        url = unquote(url)
    except:
        pass
    
    try:
        parsed = urlparse(url)
        clean_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            '',  # Remove params
            '',  # Remove query
            ''   # Remove fragment
        ))
        clean_url = clean_url.rstrip('/').strip().lower()
        return clean_url
    except Exception:
        url = url.lower().strip()
        if '#' in url:
            url = url.split('#')[0]
        if '?' in url:
            url = url.split('?')[0]
        url = url.rstrip('/')
        return url

def deduplicate_urls(df, url_column):
    """Remove duplicate URLs after normalization"""
    original_count = len(df)
    df['normalized_url'] = df[url_column].apply(normalize_url)
    
    duplicate_groups = df.groupby('normalized_url').agg({
        url_column: list,
        'normalized_url': 'first'
    }).reset_index(drop=True)
    
    duplicates = duplicate_groups[duplicate_groups[url_column].apply(len) > 1]
    df_cleaned = df.drop_duplicates(subset=['normalized_url'], keep='first')
    
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
    """Calculate cosine similarity using TF-IDF"""
    if not texts1 or not texts2:
        return np.zeros((len(texts1), len(texts2)))

    all_texts = texts1 + texts2
    all_texts = [text if text else "empty" for text in all_texts]

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        matrix1 = tfidf_matrix[:len(texts1)]
        matrix2 = tfidf_matrix[len(texts1):]
        return cosine_similarity(matrix1, matrix2)
    except:
        return np.zeros((len(texts1), len(texts2)))

def call_openai_api(prompt, model="gpt-3.5-turbo", max_tokens=2000):
    """Safely call OpenAI API with error handling"""
    try:
        if 'openai_api_key' not in st.session_state:
            return None, "No API key available"
        
        client = openai.OpenAI(api_key=st.session_state.openai_api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert SEO analyst specializing in content cannibalization and content strategy optimization."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        return response.choices[0].message.content, None
    
    except openai.AuthenticationError:
        return None, "Invalid API key"
    except openai.RateLimitError:
        return None, "Rate limit exceeded. Try again later."
    except openai.APIError as e:
        return None, f"OpenAI API error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def generate_ai_insights(results_df, url_data, removed_count):
    """Generate AI-powered insights from cannibalization data"""
    if not enable_ai_analysis:
        return "AI analysis disabled. Enable in sidebar to get detailed insights."
    
    # Prepare data summary for AI analysis
    high_risk_count = len(results_df[results_df['Risk_Level'] == 'High'])
    medium_risk_count = len(results_df[results_df['Risk_Level'] == 'Medium'])
    avg_score = results_df['Composite_Score'].mean()
    
    # Top cannibalization issues
    top_issues = results_df.head(10)
    
    # Create prompt for AI analysis
    prompt = f"""
As an expert SEO analyst, analyze this content cannibalization audit and provide strategic recommendations.

DATA SUMMARY:
- Total URL pairs analyzed: {len(results_df):,}
- High risk pairs (>80% similarity): {high_risk_count}
- Medium risk pairs (60-80% similarity): {medium_risk_count}
- Average composite similarity score: {avg_score:.1f}%
- URLs with parameters removed: {removed_count}

TOP 5 CANNIBALIZATION ISSUES:
{top_issues[['URL_1', 'URL_2', 'Composite_Score']].head().to_string()}

Please provide a comprehensive analysis with:

## Executive Summary
Brief overview of findings and urgency level

## Key Issues Identified
- Most critical cannibalization problems
- Patterns in high-risk pairs

## Strategic Recommendations
- Immediate actions needed
- Content consolidation vs differentiation strategies
- SEO optimization priorities

## Implementation Plan
- Phase 1: Critical fixes (next 30 days)
- Phase 2: Content strategy optimization (next 90 days)
- Phase 3: Long-term content planning

Focus on actionable insights that will improve search performance and reduce internal competition.
"""

    with st.spinner('🤖 Generating AI insights...'):
        insights, error = call_openai_api(prompt, model=gpt_model, max_tokens=3000)
    
    if error:
        return f"❌ AI Analysis Failed: {error}\n\nPlease check your API key and try again."
    
    return insights or "No insights generated."

def generate_enhanced_markdown_report(results_df, url_data, removed_count, ai_insights=None):
    """Generate comprehensive markdown report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate key metrics
    high_risk_count = len(results_df[results_df['Risk_Level'] == 'High'])
    medium_risk_count = len(results_df[results_df['Risk_Level'] == 'Medium'])
    avg_score = results_df['Composite_Score'].mean()
    max_score = results_df['Composite_Score'].max()
    
    # Top issues
    top_10_issues = results_df.head(10)
    
    # Build comprehensive report
    report_sections = []
    
    # Header
    report_sections.append(f"""# 🎯 Content Cannibalization Analysis Report

**Generated:** {timestamp}  
**Analysis Type:** Enhanced Multi-Dimensional Similarity Analysis  
**Total URLs Processed:** {len(url_data)}  
**URL Pairs Analyzed:** {len(results_df):,}

---

## 📊 Executive Summary

This comprehensive analysis evaluated **{len(url_data):,} URLs** across **{len(results_df):,} URL pairs** to identify potential content cannibalization risks that could impact your SEO performance.

### Key Metrics
- **High Risk Pairs:** {high_risk_count} (>80% similarity)
- **Medium Risk Pairs:** {medium_risk_count} (60-80% similarity) 
- **Average Similarity:** {avg_score:.1f}%
- **Maximum Similarity:** {max_score:.1f}%
- **URLs Cleaned:** {removed_count} duplicate parameters removed

### Risk Assessment
{"🚨 **HIGH RISK** - Immediate action required" if high_risk_count > 10 else "⚠️ **MEDIUM RISK** - Review recommended" if high_risk_count > 5 else "✅ **LOW RISK** - Minimal issues detected"}

---""")

    # Critical Issues Section
    report_sections.append(f"""## 🚨 Critical Issues Detected

### Top 10 Cannibalization Candidates

| Rank | URL 1 | URL 2 | Score | Risk |
|------|-------|-------|-------|------|""")
    
    for idx, (_, row) in enumerate(top_10_issues.iterrows(), 1):
        url1_short = row['URL_1'][:50] + "..." if len(row['URL_1']) > 50 else row['URL_1']
        url2_short = row['URL_2'][:50] + "..." if len(row['URL_2']) > 50 else row['URL_2']
        report_sections.append(f"| {idx} | {url1_short} | {url2_short} | {row['Composite_Score']}% | {row['Risk_Level']} |")
    
    report_sections.append("\n---")

    # AI Insights Section
    if ai_insights:
        report_sections.append(f"""## 🤖 AI-Powered Strategic Analysis

{ai_insights}

---""")

    # Action Plan
    report_sections.append(f"""## 💡 Action Plan & Recommendations

### Immediate Actions (Next 30 Days)
1. **Critical Fixes:** Address {high_risk_count} high-risk pairs
2. **Title Optimization:** Differentiate similar titles
3. **Content Consolidation:** Merge low-performing duplicates
4. **Internal Linking:** Review link structure for competing pages

### Medium-term Strategy (30-90 Days)  
1. **Content Gap Analysis:** Identify missing topics in your content strategy
2. **Content Differentiation:** Create unique value propositions for similar pages
3. **Performance Monitoring:** Track cannibalization metrics monthly

### Long-term Planning (90+ Days)
1. **Content Calendar:** Plan differentiated content to avoid future cannibalization
2. **Topic Clusters:** Build authoritative content hubs around main themes
3. **Competitive Analysis:** Monitor competitor content for strategic opportunities
4. **SEO Performance Tracking:** Implement ongoing cannibalization monitoring

---

## 🔧 Technical Implementation

### URL Normalization Results
- **Original URLs:** {len(url_data) + removed_count}
- **Duplicates Removed:** {removed_count}
- **Clean URLs Analyzed:** {len(url_data)}
- **Processing Success Rate:** {((len(url_data)) / (len(url_data) + removed_count) * 100):.1f}%

### Analysis Methodology
- **Title Weight:** 50% (highest SEO impact)
- **H1 Weight:** 35% (on-page optimization signal)
- **Meta Description:** 15% (SERP presentation overlap)

---

*Report generated by Enhanced Content Cannibalization Analyzer*  
*For questions about this analysis or implementation guidance, please review the methodology section or consult with your SEO team.*""")

    return "\n".join(report_sections)

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
    # Check if we need to run analysis or use cached results
    if 'analysis_results' not in st.session_state or st.button("🔄 Recalculate Analysis", help="Click to recalculate with new data"):
        
        with st.spinner('Loading files...'):
            # Load internal HTML data
            try:
                internal_df = pd.read_csv(internal_file, sep=';', encoding='utf-8')
            except:
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

        # Identify columns
        with st.spinner('Analyzing data structure...'):
            url_column = None
            for col in internal_df.columns:
                if 'address' in col.lower() or 'url' in col.lower():
                    url_column = col
                    break

            title_column = next((col for col in internal_df.columns if 'Title 1' in col or 'title' in col.lower()), None)
            h1_column = next((col for col in internal_df.columns if 'H1-1' in col or 'h1' in col.lower()), None)
            meta_column = next((col for col in internal_df.columns if 'Meta Description' in col or 'meta description' in col.lower()), None)

            st.info(f"Detected columns - URL: {url_column}, Title: {title_column}, H1: {h1_column}")

        # Filter and normalize URLs
        if url_column:
            internal_df = internal_df[internal_df[url_column].str.contains('http', na=False)]
            internal_df = internal_df[~internal_df[url_column].str.contains('redirect', na=False, case=False)]
            
            # URL normalization and deduplication
            with st.spinner('🧹 Normalizing URLs and removing duplicates...'):
                cleaned_df, removed_count, duplicate_report = deduplicate_urls(internal_df, url_column)
            
            # Show deduplication results
            if removed_count > 0:
                st.warning(f"🧹 **URL Cleaning Results**: Removed {removed_count} duplicate URLs with parameters")
                
                with st.expander(f"📋 View {len(duplicate_report)} duplicate groups"):
                    for i, dup in enumerate(duplicate_report[:10]):
                        st.write(f"**Group {i+1}:** `{dup['normalized_url']}`")
                        for orig_url in dup['original_urls']:
                            st.write(f"  - {orig_url}")
                        st.write("---")
                    
                    if len(duplicate_report) > 10:
                        st.info(f"... and {len(duplicate_report) - 10} more duplicate groups")
            else:
                st.success("✅ No URL duplicates found - your data is already clean!")
            
            internal_df = cleaned_df
            st.metric("Valid URLs for analysis (after deduplication)", len(internal_df))

            # Process similarity analysis
            with st.spinner('Calculating multi-dimensional similarity...'):
                results = []
                urls = internal_df['normalized_url'].tolist()
                original_urls = internal_df[url_column].tolist()
                n = len(urls)

                # Prepare data for each URL
                url_data = {}
                for idx, row in internal_df.iterrows():
                    normalized_url = row['normalized_url']
                    original_url = row[url_column]
                    
                    url_data[normalized_url] = {
                        'original_url': original_url,
                        'title': clean_text(row.get(title_column, '')) if title_column else '',
                        'h1': clean_text(row.get(h1_column, '')) if h1_column else '',
                        'meta': clean_text(row.get(meta_column, '')) if meta_column else ''
                    }

                # Calculate similarity matrices
                titles = [url_data[url]['title'] for url in urls]
                h1s = [url_data[url]['h1'] for url in urls]
                metas = [url_data[url]['meta'] for url in urls]

                title_sim_matrix = calculate_text_similarity(titles, titles)
                h1_sim_matrix = calculate_text_similarity(h1s, h1s)
                meta_sim_matrix = calculate_text_similarity(metas, metas)

                # Calculate results for each pair
                progress_bar = st.progress(0)
                total_pairs = (n * (n - 1)) // 2
                pair_count = 0
                
                for i in range(n):
                    for j in range(i + 1, n):
                        url1 = urls[i]
                        url2 = urls[j]

                        if url1 == url2:
                            continue

                        # Get similarities
                        title_sim = title_sim_matrix[i, j] * 100
                        h1_sim = h1_sim_matrix[i, j] * 100
                        meta_sim = meta_sim_matrix[i, j] * 100

                        # Calculate composite score with updated weights (removed keyword/embedding components)
                        weights = {
                            'title': 0.50,  # Increased from 0.35
                            'h1': 0.35,     # Increased from 0.25
                            'meta': 0.15    # Same as before
                        }

                        composite_score = (
                            weights['title'] * title_sim +
                            weights['h1'] * h1_sim +
                            weights['meta'] * meta_sim
                        )

                        results.append({
                            'URL_1': url_data[url1]['original_url'],
                            'URL_2': url_data[url2]['original_url'],
                            'Composite_Score': round(composite_score, 1),
                            'Title_Similarity': round(title_sim, 1),
                            'H1_Similarity': round(h1_sim, 1),
                            'Meta_Similarity': round(meta_sim, 1),
                            'Risk_Level': 'High' if composite_score > 80 else 'Medium' if composite_score > 60 else 'Low'
                        })
                        
                        pair_count += 1
                        if pair_count % 100 == 0:
                            progress_bar.progress(min(pair_count / total_pairs, 1.0))

                progress_bar.progress(1.0)
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Composite_Score', ascending=False)

                # Store in session state to prevent recalculation
                st.session_state.analysis_results = results_df
                st.session_state.url_data = url_data
                st.session_state.removed_count = removed_count

                st.success(f"✅ Analysis complete! Analyzed {len(results_df):,} URL pairs (cached for instant downloads)")

    else:
        # Use cached results
        results_df = st.session_state.analysis_results
        url_data = st.session_state.url_data
        removed_count = st.session_state.removed_count
        
        st.info("📊 Using cached analysis results. Click 'Recalculate Analysis' to refresh with new data.")

    # Summary statistics
    st.markdown("### 📊 Cannibalization Risk Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        high_risk = len(results_df[results_df['Risk_Level'] == 'High'])
        st.metric("High Risk Pairs", high_risk, help=">80% similarity")

    with col2:
        medium_risk = len(results_df[results_df['Risk_Level'] == 'Medium'])
        st.metric("Medium Risk Pairs", medium_risk, help="60-80% similarity")

    with col3:
        avg_composite = results_df['Composite_Score'].mean()
        st.metric("Avg Composite Score", f"{avg_composite:.1f}%")

    with col4:
        high_similarity = len(results_df[results_df['Composite_Score'] > 70])
        st.metric("Pairs >70% Similar", high_similarity)

    # Generate AI insights if enabled
    ai_insights = None
    if enable_ai_analysis and OPENAI_AVAILABLE:
        ai_insights = generate_ai_insights(results_df, url_data, removed_count)

    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 High Risk Pairs", "📊 All Results", "🤖 AI Insights", "📥 Download"])

    with tab1:
        st.markdown("### High Risk Cannibalization Candidates")
        high_risk_df = results_df[results_df['Risk_Level'] == 'High'].head(50)
        
        if len(high_risk_df) > 0:
            display_df = high_risk_df[['URL_1', 'URL_2', 'Composite_Score', 'Title_Similarity',
                                     'H1_Similarity', 'Meta_Similarity']].copy()
            
            for col in ['Composite_Score', 'Title_Similarity', 'H1_Similarity', 'Meta_Similarity']:
                display_df[col] = display_df[col].astype(str) + '%'
            
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.info("No high-risk cannibalization pairs found. Your content is well-differentiated!")

    with tab2:
        st.markdown("### All URL Pairs Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            min_composite = st.slider("Min Composite Score %", 0, 100, 50)
        with col2:
            risk_filter = st.multiselect("Risk Levels", ['High', 'Medium', 'Low'], default=['High', 'Medium'])

        filtered_df = results_df[
            (results_df['Composite_Score'] >= min_composite) &
            (results_df['Risk_Level'].isin(risk_filter))
        ]

        st.metric("Filtered Results", len(filtered_df))
        display_cols = ['URL_1', 'URL_2', 'Composite_Score', 'Title_Similarity',
                       'H1_Similarity', 'Meta_Similarity', 'Risk_Level']
        display_df = filtered_df[display_cols].head(100).copy()
        
        for col in ['Composite_Score', 'Title_Similarity', 'H1_Similarity', 'Meta_Similarity']:
            display_df[col] = display_df[col].astype(str) + '%'
        
        st.dataframe(display_df, use_container_width=True, height=400)

    with tab3:
        st.markdown("### 🤖 AI-Powered Strategic Analysis")
        
        if not OPENAI_AVAILABLE:
            st.error("❌ OpenAI library not installed. Run: `pip install openai`")
        elif not enable_ai_analysis:
            st.info("💡 Enable AI analysis in the sidebar to get detailed strategic insights")
        elif ai_insights:
            st.markdown(ai_insights)
        else:
            st.warning("⚠️ AI analysis failed. Check your API key in the sidebar.")

    with tab4:
        st.markdown("### Download Complete Analysis")
        
        # Generate enhanced markdown report
        enhanced_report = generate_enhanced_markdown_report(
            results_df, url_data, removed_count, ai_insights
        )
        
        # Prepare download data (instant - uses cached results)
        download_df = results_df.copy()
        
        csv_buffer = io.StringIO()
        download_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_cannibalization_analysis_{timestamp}.csv"

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Full Analysis (CSV)",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                help="Contains all URL pairs with similarity scores and risk levels"
            )

        with col2:
            st.download_button(
                label="📄 Download Enhanced Report (Markdown)",
                data=enhanced_report,
                file_name=f"cannibalization_report_{timestamp}.md",
                mime="text/markdown",
                help="Comprehensive markdown report with AI insights and strategic recommendations"
            )

    # Insights section
    st.markdown("### 💡 Key Insights")
    
    very_high_title_sim = len(results_df[results_df['Title_Similarity'] > 90])
    insights = []

    if removed_count > 0:
        insights.append(f"🧹 Removed {removed_count} duplicate URLs with parameters - analysis is now more accurate")

    if very_high_title_sim > 0:
        insights.append(f"⚠️ Found {very_high_title_sim} URL pairs with >90% title similarity - consider differentiating titles")

    if len(results_df[results_df['Risk_Level'] == 'High']) > 10:
        insights.append("🚨 Multiple high-risk cannibalization issues detected - prioritize content consolidation or differentiation")

    if enable_ai_analysis and ai_insights:
        insights.append("🤖 AI analysis completed - see AI Insights tab for detailed strategic recommendations")

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

    2. **GSC Report** (from Google Search Console)
       - Must contain: Queries and Landing Pages
    """)

    with st.expander("📖 How this enhanced analysis works"):
        st.markdown("""
        This tool provides comprehensive content cannibalization analysis:

        **🔍 Multi-Dimensional Analysis:**
        - **Title Similarity (50% weight)** - Most critical for SEO
        - **H1 Similarity (35% weight)** - On-page optimization signal
        - **Meta Description Similarity (15% weight)** - SERP presentation

        **🧹 URL Normalization:**
        - Removes parameters (&, ?, =, #) and URL encoding (%20, etc.)
        - Deduplicates similar URLs before analysis
        - Focuses on genuine content differences

        **🤖 AI-Powered Insights:**
        - Strategic recommendations from GPT analysis
        - Executive summaries and action plans
        - Content optimization priorities
        - Implementation roadmaps

        **📊 Enhanced Reporting:**
        - Professional markdown reports
        - Detailed strategic analysis  
        - Export-ready documentation
        - Cached results for instant downloads
        """)

# Footer
st.markdown("---")
st.markdown("🎯 Enhanced Content Cannibalization Analyzer - Built for SEO Professionals | **NEW**: AI-Powered Insights")
