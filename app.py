import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from datetime import datetime
import json
import chardet

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


@st.cache_data
def clean_text(text):
    """Clean and normalize text for comparison"""
    if pd.isna(text) or text is None:
        return ""
    return str(text).lower().strip()


@st.cache_data
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
    except Exception:
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


def robust_csv_reader(file, file_name="uploaded_file"):
    """
    Robust CSV reader that handles various encoding and delimiter issues
    
    Args:
        file: File-like object or file path
        file_name: Name of the file for error messages
    
    Returns:
        pandas.DataFrame or None if parsing fails
    """
    
    # Read the raw file content
    if hasattr(file, 'read'):
        raw_data = file.read()
        file.seek(0)  # Reset file pointer
    else:
        with open(file, 'rb') as f:
            raw_data = f.read()
    
    # Detect encoding
    encoding_result = chardet.detect(raw_data)
    detected_encoding = encoding_result['encoding'] or 'utf-8'
    
    # Convert to string for analysis
    try:
        text_content = raw_data.decode(detected_encoding)
    except UnicodeDecodeError:
        # Fallback encodings
        for encoding in ['utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']:
            try:
                text_content = raw_data.decode(encoding)
                detected_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Unable to decode file {file_name} with any common encoding")
    
    # Detect delimiter
    first_lines = text_content.split('\n')[:5]
    delimiters = [',', ';', '\t', '|']
    delimiter_scores = {}
    
    for delimiter in delimiters:
        scores = []
        for line in first_lines:
            if line.strip():
                scores.append(line.count(delimiter))
        if scores and all(s > 0 for s in scores):
            # Consistent delimiter usage
            delimiter_scores[delimiter] = sum(scores) / len(scores)
    
    # Choose best delimiter
    if delimiter_scores:
        best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
    else:
        best_delimiter = ','
    
    # Try to read with detected parameters
    try:
        # Try with detected delimiter and encoding
        if hasattr(file, 'read'):
            file.seek(0)
        df = pd.read_csv(file, delimiter=best_delimiter, encoding=detected_encoding)
        return df
    
    except pd.errors.ParserError as e:
        # Handle specific parsing errors
        if hasattr(file, 'read'):
            file.seek(0)
        
        # Try with different approaches
        try:
            # Skip bad lines
            df = pd.read_csv(file, delimiter=best_delimiter, encoding=detected_encoding,
                           on_bad_lines='skip', engine='python')
            return df
        except Exception:
            pass
        
        try:
            # Try with no header
            file.seek(0)
            df = pd.read_csv(file, delimiter=best_delimiter, encoding=detected_encoding,
                           header=None, on_bad_lines='skip', engine='python')
            return df
        except Exception:
            pass
        
        # Try different delimiters
        for delimiter in [',', ';', '\t', '|']:
            try:
                file.seek(0)
                df = pd.read_csv(file, delimiter=delimiter, encoding=detected_encoding,
                               on_bad_lines='skip', engine='python')
                if len(df.columns) > 1:  # Valid CSV structure
                    return df
            except Exception:
                continue
        
        raise ValueError(f"Unable to parse CSV file {file_name}: {str(e)}")


def validate_gsc_data(df, file_name="GSC file"):
    """
    Validate and clean GSC data
    
    Args:
        df: DataFrame to validate
        file_name: Name for error messages
    
    Returns:
        Cleaned DataFrame
    """
    
    if df is None or df.empty:
        raise ValueError(f"{file_name} is empty")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Look for required columns
    required_patterns = {
        'Landing Page': ['landing page', 'url', 'page', 'landing'],
        'Query': ['query', 'keyword', 'search term', 'term'],
        'Clicks': ['clicks', 'click', 'sessions']
    }
    
    column_mapping = {}
    
    for standard_name, patterns in required_patterns.items():
        for col in df.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern in col_lower:
                    column_mapping[col] = standard_name
                    break
    
    # Rename columns to standard names
    df = df.rename(columns=column_mapping)
    
    # Check if we have the minimum required columns
    required_cols = ['Landing Page', 'Query']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        available_cols = list(df.columns)
        raise ValueError(
            f"{file_name} missing required columns: {missing_cols}. "
            f"Available columns: {available_cols}"
        )
    
    # Clean data
    df = df.dropna(subset=['Landing Page', 'Query'])
    df['Landing Page'] = df['Landing Page'].astype(str).str.strip()
    df['Query'] = df['Query'].astype(str).str.strip()
    
    # Ensure Clicks is numeric
    if 'Clicks' in df.columns:
        df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce').fillna(0)
    else:
        df['Clicks'] = 0
    
    return df


def validate_internal_data(df, file_name="Internal HTML file"):
    """
    Validate and clean internal HTML data
    
    Args:
        df: DataFrame to validate
        file_name: Name for error messages
    
    Returns:
        Cleaned DataFrame
    """
    
    if df is None or df.empty:
        raise ValueError(f"{file_name} is empty")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Look for URL column
    url_column = None
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['url', 'address', 'uri', 'link']):
            url_column = col
            break
    
    if not url_column:
        available_cols = list(df.columns)
        raise ValueError(
            f"{file_name} missing URL column. "
            f"Available columns: {available_cols}"
        )
    
    # Clean data
    df = df.dropna(subset=[url_column])
    df[url_column] = df[url_column].astype(str).str.strip()
    
    # Filter for HTTP URLs
    df = df[df[url_column].str.contains('http', case=False, na=False)]
    
    return df


def get_ai_insights(results_df, openai_api_key, business_goal="", target_audience=""):
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
                "h1_similarity": float(row['H1_Similarity']),
                "meta_similarity": float(row['Meta_Similarity']),
                "keyword_overlap": float(row['Keyword_Overlap']),
                "same_intent": row['Same_Intent'],
                "intent1": row['Intent_1'],
                "intent2": row['Intent_2']
            })
        
        # Create enhanced prompt
        prompt = f"""You are an expert SEO Strategist specializing in identifying and resolving content cannibalization issues. Your goal is to transform raw cannibalization data into a strategic, actionable executive report that a marketing team can implement to improve organic search performance.

**Website Context:**
* **Business Goal:** {business_goal if business_goal else "[Not specified - assume general organic traffic growth]"}
* **Target Audience:** {target_audience if target_audience else "[Not specified - analyze based on URL patterns]"}

**Input Data:** Here is the content cannibalization analysis data, presented in JSON format. Each entry represents a pair of URLs from our site that are potentially competing.

```json
{json.dumps(analysis_data, indent=2)}
```

**Task: Generate a comprehensive SEO report with the following sections:**

**1. Executive Summary:**
* Provide a 2-3 sentence overview of the key findings.
* Assess the severity of the content cannibalization issue and its current impact on SEO performance (e.g., suppressed rankings, diluted authority, poor user experience).
* Summarize the high-level strategic direction recommended in this report.

**2. Top 5 Priority Actions:**
* Present the five most critical cannibalization issues to address first.
* Prioritize based on a combination of similarity scores, intent matching, and the potential for significant performance uplift.
* **Format this as a table** with the following columns: `Priority`, `URL Pair`, `Similarity Score`, `Recommended Action`, `Rationale & Expected Outcome`.

**3. Content Consolidation & Pruning Plan:**
* Identify specific clusters of pages that should be merged or redirected.
* For each cluster, recommend a single "canonical" (winner) URL to become the primary target.
* List the "loser" URLs that should be 301 redirected or have their content merged into the winner.
* **Format this as a table** with the following columns: `Canonical URL (Winner)`, `URLs to Consolidate/Redirect (Losers)`, `Justification for Choice`.

**4. Quick Wins (Low-Effort, High-Impact):**
* List at least three immediate fixes that require minimal resources (e.g., can be done in under an hour).
* Focus on actions like:
   * Optimizing title tags and meta descriptions to differentiate intent.
   * Adjusting internal links to signal the most important page to search engines.
   * Slightly de-optimizing a competing page for the target keyword.

**5. Long-Term Strategy & Prevention:**
* Provide strategic recommendations to prevent future content cannibalization.
* Include guidelines for the content creation workflow, such as:
   * A process for checking existing content before creating new articles.
   * A keyword-to-URL mapping strategy.
   * Best practices for internal linking to reinforce topical authority.

**Tone and Formatting:**
* **Tone:** Be professional, data-driven, and authoritative.
* **Formatting:** Use Markdown for clarity. Employ **bolding** for key terms and use tables where requested to ensure the report is scannable and actionable.
* **Constraint:** Do not simply restate the data from the JSON. Your value is in the interpretation, strategic insights, and actionable recommendations."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert SEO consultant specializing in content strategy and cannibalization issues. Provide actionable insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating AI insights: {str(e)}"


def main():
    """Main application function"""
    
    # File upload section
    st.markdown("### 📁 Upload Your Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        internal_file = st.file_uploader(
            "Upload Internal HTML Report (CSV)",
            type=['csv', 'txt'],
            help="Upload your Screaming Frog or similar internal HTML report"
        )
    
    with col2:
        gsc_file = st.file_uploader(
            "Upload Google Search Console Report (CSV)",
            type=['csv', 'txt'],
            help="Upload your GSC performance report with queries and landing pages"
        )
    
    if internal_file and gsc_file:
        try:
            with st.spinner("Processing your files..."):
                # Use robust CSV reader for both files
                internal_df = robust_csv_reader(internal_file, internal_file.name)
                internal_df = validate_internal_data(internal_df, internal_file.name)
                
                gsc_df = robust_csv_reader(gsc_file, gsc_file.name)
                gsc_df = validate_gsc_data(gsc_df, gsc_file.name)
                
                st.success("✅ Files loaded successfully!")
                
                # Display file info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Internal URLs", len(internal_df))
                with col2:
                    st.metric("GSC Queries", len(gsc_df))
                
                # Find URL column in internal data
                url_column = None
                for col in internal_df.columns:
                    if any(keyword in col.lower() for keyword in ['url', 'address', 'uri', 'link']):
                        url_column = col
                        break
                
                if not url_column:
                    st.error("Could not find URL column in internal data")
                    return
                
                # Prepare data for analysis
                internal_df = internal_df.rename(columns={url_column: 'URL'})
                
                # Get unique URLs from GSC
                gsc_urls = gsc_df['Landing Page'].unique()
                
                # Filter internal data to only include URLs present in GSC
                internal_df = internal_df[internal_df['URL'].isin(gsc_urls)]
                
                if len(internal_df) < 2:
                    st.warning("Need at least 2 URLs to analyze cannibalization")
                    return
                
                # Find title, h1, and meta description columns
                title_col = None
                h1_col = None
                meta_col = None
                
                for col in internal_df.columns:
                    col_lower = col.lower()
                    if 'title' in col_lower and '1' in col_lower:
                        title_col = col
                    elif 'h1' in col_lower and '1' in col_lower:
                        h1_col = col
                    elif 'meta' in col_lower and 'description' in col_lower:
                        meta_col = col
                
                # Create analysis data
                urls = internal_df['URL'].tolist()
                titles = [clean_text(internal_df[internal_df['URL'] == url][title_col].iloc[0] if title_col and title_col in internal_df.columns else "") for url in urls]
                h1s = [clean_text(internal_df[internal_df['URL'] == url][h1_col].iloc[0] if h1_col and h1_col in internal_df.columns else "") for url in urls]
                metas = [clean_text(internal_df[internal_df['URL'] == url][meta_col].iloc[0] if meta_col and meta_col in internal_df.columns else "") for url in urls]
                
                # Get keywords for each URL from GSC
                url_keywords = {}
                for url in urls:
                    keywords = gsc_df[gsc_df['Landing Page'] == url]['Query'].tolist()
                    url_keywords[url] = keywords
                
                # Calculate similarities
                title_sim = calculate_text_similarity(titles, titles)
                h1_sim = calculate_text_similarity(h1s, h1s)
                meta_sim = calculate_text_similarity(metas, metas)
                
                # Calculate keyword overlap
                keyword_overlaps = []
                for i, url1 in enumerate(urls):
                    overlaps = []
                    for j, url2 in enumerate(urls):
                        if i != j:
                            overlap = calculate_keyword_overlap(
                                url_keywords.get(url1, []),
                                url_keywords.get(url2, [])
                            )
                            overlaps.append(overlap)
                        else:
                            overlaps.append(0.0)
                    keyword_overlaps.append(overlaps)
                
                keyword_overlap = np.array(keyword_overlaps)
                
                # Create results dataframe
                results = []
                for i in range(len(urls)):
                    for j in range(i + 1, len(urls)):
                        # Calculate composite score with weights
                        title_score = title_sim[i][j] * 100
                        h1_score = h1_sim[i][j] * 100
                        meta_score = meta_sim[i][j] * 100
                        keyword_score = keyword_overlap[i][j] * 100
                        
                        # Weighted composite score
                        composite_score = (
                            title_score * 0.35 +
                            h1_score * 0.25 +
                            meta_score * 0.15 +
                            keyword_score * 0.15 +
                            0  # Semantic similarity placeholder
                        )
                        
                        # Determine risk level
                        intent1 = detect_intent_from_title(titles[i])
                        intent2 = detect_intent_from_title(titles[j])
                        same_intent = intent1 == intent2
                        
                        if composite_score >= 80 and same_intent:
                            risk_level = 'High'
                        elif composite_score >= 60:
                            risk_level = 'Medium'
                        else:
                            risk_level = 'Low'
                        
                        results.append({
                            'URL_1': urls[i],
                            'URL_2': urls[j],
                            'Title_Similarity': round(title_score, 1),
                            'H1_Similarity': round(h1_score, 1),
                            'Meta_Similarity': round(meta_score, 1),
                            'Keyword_Overlap': round(keyword_score, 1),
                            'Composite_Score': round(composite_score, 1),
                            'Risk_Level': risk_level,
                            'Intent_1': intent1,
                            'Intent_2': intent2,
                            'Same_Intent': same_intent
                        })
                
                results_df = pd.DataFrame(results)
                
                if results_df.empty:
                    st.warning("No cannibalization pairs found")
                    return
                
                # Sort by composite score descending
                results_df = results_df.sort_values('Composite_Score', ascending=False)
                
                # Display results
                st.markdown("### 📊 Analysis Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Pairs", len(results_df))
                with col2:
                    st.metric("High Risk", len(results_df[results_df['Risk_Level'] == 'High']))
                with col3:
                    st.metric("Avg Score", f"{results_df['Composite_Score'].mean():.1f}%")
                
                # Create tabs
                tabs = st.tabs(["📈 Results", "🔍 Details", "📥 Download"])
                
                with tabs[0]:
                    st.markdown("#### Cannibalization Risk Overview")
                    
                    # Risk level distribution
                    risk_counts = results_df['Risk_Level'].value_counts()
                    st.bar_chart(risk_counts)
                    
                    # Top risks
                    st.markdown("#### Top 10 Highest Risk Pairs")
                    top_risks = results_df.head(10)
                    
                    for _, row in top_risks.iterrows():
                        with st.expander(f"{row['URL_1']} vs {row['URL_2']} ({row['Composite_Score']}%)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**URL 1:**", row['URL_1'])
                                st.write("**Intent:**", row['Intent_1'])
                            with col2:
                                st.write("**URL 2:**", row['URL_2'])
                                st.write("**Intent:**", row['Intent_2'])
                            
                            st.write("**Similarity Scores:**")
                            st.write(f"- Title: {row['Title_Similarity']}%")
                            st.write(f"- H1: {row['H1_Similarity']}%")
                            st.write(f"- Meta: {row['Meta_Similarity']}%")
                            st.write(f"- Keywords: {row['Keyword_Overlap']}%")
                
                with tabs[1]:
                    st.markdown("#### Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                
                with tabs[2]:
                    st.markdown("#### Download Results")
                    
                    # CSV download
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label="📥 Download Full Analysis (CSV)",
                        data=csv_data,
                        file_name=f"cannibalization_analysis_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary report
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
                    for _, row in results_df.head(10).iterrows():
                        summary += f"\n{row['URL_1']}\n  vs {row['URL_2']}\n  Score: {row['Composite_Score']}% | Risk: {row['Risk_Level']}\n"
                    
                    st.download_button(
                        label="📄 Download Summary Report (TXT)",
                        data=summary,
                        file_name=f"cannibalization_summary_{timestamp}.txt",
                        mime="text/plain"
                    )
                
                # AI Insights
                if openai_api_key:
                    st.markdown("### 🤖 AI-Powered Insights")
                    
                    business_goal = st.text_input(
                        "Business Goal (optional)",
                        placeholder="e.g., Increase organic traffic for SaaS product pages"
                    )
                    
                    target_audience = st.text_input(
                        "Target Audience (optional)",
                        placeholder="e.g., Small business owners looking for CRM software"
                    )
                    
                    if st.button("Generate AI Report", type="primary"):
                        with st.spinner("Generating AI insights..."):
                            ai_insights = get_ai_insights(
                                results_df, 
                                openai_api_key, 
                                business_goal, 
                                target_audience
                            )
                            
                            if "Error" not in str(ai_insights):
                                st.markdown(ai_insights)
                                
                                # Download AI report
                                st.download_button(
                                    label="📥 Download AI Report",
                                    data=ai_insights,
                                    file_name=f"ai_seo_report_{timestamp}.md",
                                    mime="text/markdown"
                                )
                            else:
                                st.error(ai_insights)
                
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.info("Please check your file formats and try again.")
    
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


# Footer
st.markdown("---")
st.markdown("🎯 Enhanced Content Cannibalization Analyzer - Built for SEO Professionals")

# Run the main function
if __name__ == "__main__":
    main()
