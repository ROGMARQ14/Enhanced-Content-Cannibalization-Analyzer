import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from datetime import datetime

# ──────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enhanced Content Cannibalization Analyzer",
    page_icon="🎯",
    layout="wide"
)

# ──────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Return a lowercase, stripped string (empty if None/NaN)."""
    if text is None or pd.isna(text):
        return ""
    return str(text).lower().strip()


def detect_intent_from_title(title) -> str:
    """
    Robust intent detection that never crashes on None / NaN / non-string.
    """
    # Normalise input
    if title is None or pd.isna(title):
        return "unknown"

    title_lower = str(title).strip().lower()
    if not title_lower:                         # empty after cleaning
        return "unknown"

    # Pattern checks
    if '?' in title_lower:
        return "question"
    if any(word in title_lower for word in ['how to', 'guide', 'tutorial', 'step']):
        return "how-to"
    if any(word in title_lower for word in ['best', 'top', 'review', 'vs', 'versus', 'comparison']):
        return "comparison"
    if any(word in title_lower for word in ['what is', 'definition', 'meaning', 'explained']):
        return "informational"
    if any(word in title_lower for word in ['buy', 'price', 'cost', 'cheap', 'deal', 'discount']):
        return "transactional"
    return "general"


def calculate_text_similarity(texts1, texts2):
    """Cosine similarity (TF-IDF) with graceful fallback for empty input."""
    if not texts1 or not texts2:
        return np.zeros((len(texts1), len(texts2)))

    # Replace blank strings with placeholder to avoid zero-row TF-IDF errors
    all_texts = [(t or "empty") for t in texts1 + texts2]

    try:
        vec = TfidfVectorizer(max_features=1_000, stop_words="english")
        tfidf = vec.fit_transform(all_texts)
        return cosine_similarity(tfidf[:len(texts1)], tfidf[len(texts1):])
    except Exception:
        return np.zeros((len(texts1), len(texts2)))


def calculate_keyword_overlap(kws1, kws2):
    """Jaccard similarity of two keyword lists."""
    if not kws1 or not kws2:
        return 0.0
    s1, s2 = set(kws1), set(kws2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


# ──────────────────────────────────────────────────────────
# User interface
# ──────────────────────────────────────────────────────────
st.title("🎯 Enhanced Content Cannibalization Analyzer")
st.markdown(
    """
    This advanced tool compares pages across five dimensions  
    (titles, H1s, meta descriptions, GSC query overlap, semantic embeddings)  
    and flags potential SEO cannibalisation risks.
    """
)

col_left, col_right = st.columns(2)
with col_left:
    internal_file = st.file_uploader(
        "Upload Internal HTML Report (CSV)",
        type=['csv'],
        help="Screaming Frog / Sitebulb export with URL, Title, H1, Meta, …"
    )
with col_right:
    gsc_file = st.file_uploader(
        "Upload GSC Report (CSV)",
        type=['csv'],
        help="Search Console query export with Landing Page column"
    )

# ──────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────
if internal_file and gsc_file:
    # ── Load data ─────────────────────────────────────────
    with st.spinner("Loading files…"):
        internal_df = pd.read_csv(internal_file, sep=';', encoding='utf-8', engine='python')
        if internal_df.shape[1] < 2:                       # wrong separator fallback
            internal_df = pd.read_csv(internal_file, encoding='utf-8')
        gsc_df = pd.read_csv(gsc_file, encoding='utf-8')

    st.success("Files loaded ✔️")

    # ── Identify key columns dynamically ─────────────────
    url_col = next((c for c in internal_df.columns if "address" in c.lower() or "url" in c.lower()), None)
    title_col = next((c for c in internal_df.columns if "title" in c.lower()), None)
    h1_col    = next((c for c in internal_df.columns if "h1"    in c.lower()), None)
    meta_col  = next((c for c in internal_df.columns if "meta description" in c.lower()), None)
    emb_col   = next((c for c in internal_df.columns if "embedding" in c.lower()), None)

    if url_col is None:
        st.error("URL column not found in Internal HTML report.")
        st.stop()

    # ── Basic cleaning ───────────────────────────────────
    internal_df = internal_df[internal_df[url_col].str.contains("http", na=False)]
    st.metric("Valid URLs analysed", len(internal_df))

    # ── Aggregate queries by URL ─────────────────────────
    gsc_df.columns = gsc_df.columns.str.strip()
    url_queries = {}
    for _, r in gsc_df.iterrows():
        url = r.get('Landing Page', r.get('URL', ''))
        if not url:
            continue
        url_queries.setdefault(url, []).append(r.get('Query', ''))

    # ── Prepare per-URL data dict ────────────────────────
    urls = internal_df[url_col].tolist()
    url_data = {}
    for _, row in internal_df.iterrows():
        url = row[url_col]
        url_data[url] = {
            'title'     : clean_text(row.get(title_col, "")),
            'h1'        : clean_text(row.get(h1_col, "")),
            'meta'      : clean_text(row.get(meta_col, "")),
            'queries'   : url_queries.get(url, []),
            'intent'    : detect_intent_from_title(row.get(title_col, "")),
            'embedding' : row.get(emb_col, "")
        }

    # ── Text similarity matrices ─────────────────────────
    titles = [url_data[u]['title'] for u in urls]
    h1s    = [url_data[u]['h1']    for u in urls]
    metas  = [url_data[u]['meta']  for u in urls]

    title_sim = calculate_text_similarity(titles, titles)
    h1_sim    = calculate_text_similarity(h1s, h1s)
    meta_sim  = calculate_text_similarity(metas, metas)

    # ── Optional embedding similarity ────────────────────
    emb_sim = None
    valid_emb_idx, emb_vectors = [], []
    if emb_col:
        for idx, url in enumerate(urls):
            emb_raw = url_data[url]['embedding']
            try:
                vec = np.array([float(x) for x in str(emb_raw).split(',')])
                emb_vectors.append(vec)
                valid_emb_idx.append(idx)
            except Exception:
                pass
        if emb_vectors:
            emb_vectors = np.vstack(emb_vectors)
            emb_sim = cosine_similarity(emb_vectors)

    # ── Pair-wise scoring ────────────────────────────────
    results = []
    n = len(urls)
    weights = {'title': .35, 'h1': .25, 'meta': .15, 'kw': .15, 'emb': .10}

    with st.spinner("Calculating similarities…"):
        for i in range(n):
            for j in range(i + 1, n):
                u1, u2 = urls[i], urls[j]
                kw_overlap = calculate_keyword_overlap(url_data[u1]['queries'],
                                                       url_data[u2]['queries']) * 100
                emb_score = 0
                if emb_sim is not None and i in valid_emb_idx and j in valid_emb_idx:
                    ei, ej = valid_emb_idx.index(i), valid_emb_idx.index(j)
                    emb_score = emb_sim[ei, ej] * 100

                composite = (
                    weights['title'] * title_sim[i, j] * 100 +
                    weights['h1']   * h1_sim[i, j]    * 100 +
                    weights['meta'] * meta_sim[i, j]  * 100 +
                    weights['kw']   * kw_overlap +
                    weights['emb']  * emb_score
                )

                same_intent = url_data[u1]['intent'] == url_data[u2]['intent']
                risk = ("High" if composite > 80 and same_intent else
                        "Medium" if composite > 60 else "Low")

                results.append({
                    "URL_1": u1, "URL_2": u2,
                    "Composite_Score": round(composite, 1),
                    "Title_Similarity": round(title_sim[i, j] * 100, 1),
                    "H1_Similarity":    round(h1_sim[i, j]    * 100, 1),
                    "Meta_Similarity":  round(meta_sim[i, j]  * 100, 1),
                    "Keyword_Overlap":  round(kw_overlap, 1),
                    "Embedding_Similarity": round(emb_score, 1),
                    "Intent_1": url_data[u1]['intent'],
                    "Intent_2": url_data[u2]['intent'],
                    "Same_Intent": same_intent,
                    "Risk_Level": risk
                })

    df_results = pd.DataFrame(results).sort_values("Composite_Score", ascending=False)
    st.success(f"Analysis complete – {len(df_results):,} URL pairs scored")

    # ── Basic dashboard (high-risk overview) ─────────────
    high = df_results[df_results.Risk_Level == "High"]
    st.subheader("🚨 High-risk cannibalisation candidates")
    if high.empty:
        st.info("Great news – no pairs over 80 % with the same intent!")
    else:
        st.dataframe(
            high[['URL_1', 'URL_2', 'Composite_Score',
                  'Title_Similarity', 'H1_Similarity', 'Keyword_Overlap']],
            use_container_width=True, height=400
        )

    # ── Download button ─────────────────────────────────
    csv = df_results.to_csv(index=False).encode()
    st.download_button(
        "Download full results (CSV)",
        data=csv,
        file_name=f"cannibalisation_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv"
    )

else:
    st.info("⬆️ Upload your *Internal HTML* and *GSC* CSV files to begin.")


# ──────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ to keep your content cannibalisation-free.")
