import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from src.models import Review, AnalyzedReview, PulseReport, FeeExplanation
from src.llm_analyzer import ReviewAnalyzer
from src.review_processor import ReviewProcessor
from src.fee_explainer import FeeExplainer
from mcp.client import MCPClient

# Load environment variables
load_dotenv()

# Setup Streamlit Page
st.set_page_config(page_title="Weekly Product Pulse", page_icon="📈", layout="wide")

# Inject Custom CSS for Premium Design
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
    }

    /* Custom Header Styling */
    .pulse-header {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .pulse-subheader {
        font-size: 1.2rem;
        font-weight: 300;
        color: #6c757d;
        margin-bottom: 2rem;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        transition: transform 0.2s ease-in-out;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 3rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    /* Fee Explainer Card */
    .fee-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4ECDC4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .fee-card h4 {
        margin-top: 0;
        color: #2b2b2b;
    }
    .fee-source {
        font-size: 0.9rem;
        color: #6c757d;
        font-style: italic;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# State Machine Initialization
if 'is_analyzed' not in st.session_state:
    st.session_state.is_analyzed = False
    st.session_state.analyzed_reviews = []
    st.session_state.agg_df = pd.DataFrame()
    st.session_state.pulse_report = None
    st.session_state.fee_explanations = []
    st.session_state.published_report_url = None
    st.session_state.published_emails = {}

def run_pipeline(uploaded_file=None):
    """Executes the full LLM pipeline and stores results in session_state."""
    st.session_state.is_analyzed = False
    st.session_state.published_report_url = None
    st.session_state.published_emails = {}
    
    with st.spinner("Initializing LLM Pipeline..."):
        analyzer = ReviewAnalyzer()
        processor = ReviewProcessor("data/reviews.csv")
        
        with open("data/fee_schedule.txt", "r") as f:
            fee_schedule = f.read()
        fee_explainer = FeeExplainer(fee_schedule_text=fee_schedule)

        # 1. Load Data
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("data/reviews.csv")
            
        # Validate CSV format
        required_cols = ["review_id", "date", "rating", "review_text"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Invalid CSV format! Missing required columns: {', '.join(missing_cols)}")
            st.stop()
            
        reviews = []
        for _, row in df.iterrows():
            reviews.append(Review(
                review_id=row['review_id'],
                date=pd.to_datetime(row['date']).date(),
                rating=row['rating'],
                review_text=row['review_text']
            ))

    # 2. Categorize Reviews (Concurrent)
    progress_bar = st.progress(0, text="Categorizing Reviews via Llama 3...")
    analyzed_reviews = []
    
    def analyze_single(r):
        return analyzer.analyze_review(r)
        
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_review = [executor.submit(analyze_single, r) for r in reviews]
        completed = 0
        for future in as_completed(future_to_review):
            res = future.result()
            analyzed_reviews.append(res)
            completed += 1
            progress_bar.progress(completed / len(reviews), text=f"Categorized {completed}/{len(reviews)} reviews...")
            
    st.session_state.analyzed_reviews = analyzed_reviews

    # 3. Aggregate Data
    progress_bar.progress(0.9, text="Aggregating metrics...")
    agg_df = processor.aggregate_by_theme(analyzed_reviews)
    st.session_state.agg_df = agg_df

    # 4. Generate Pulse Report
    progress_bar.progress(0.95, text="Drafting Executive Summary...")
    valid_quotes = [r.key_quote for r in analyzed_reviews if r.key_quote]
    # Sample up to 25 quotes so we don't exceed LLM context window limits
    sampled_quotes = valid_quotes[:25]
    pulse_report = analyzer.generate_pulse_report(agg_df.to_json(), sampled_quotes)
    st.session_state.pulse_report = pulse_report

    # 5. Run Fee Explainer
    progress_bar.progress(0.98, text="Drafting Fee Explanations...")
    pricing_reviews = [r for r in analyzed_reviews if r.theme == "Pricing/Fees"]
    fee_explanations = []
    for r in pricing_reviews:
        explanation = fee_explainer.explain_fee(r.review_text)
        if explanation.is_applicable:
            fee_explanations.append({"review": r.review_text, "explanation": explanation})
    st.session_state.fee_explanations = fee_explanations

    progress_bar.progress(1.0, text="Analysis Complete!")
    time.sleep(0.5)
    progress_bar.empty()
    st.session_state.is_analyzed = True


# --- UI LAYOUT ---

st.markdown('<p class="pulse-header">Weekly Product Pulse 📈</p>', unsafe_allow_html=True)
st.markdown('<p class="pulse-subheader">AI-Powered Customer Feedback Intelligence</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/12187/12187128.png", width=100)
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload Reviews CSV", type=["csv"])
    st.caption("Required columns: `review_id`, `date`, `rating`, `review_text`")
    if st.button("🚀 Generate Weekly Pulse", use_container_width=True, type="primary"):
        run_pipeline(uploaded_file)

# Main Dashboard Content
if not st.session_state.is_analyzed:
    st.info("👈 Click **Generate Weekly Pulse** in the sidebar to start the analysis.")
else:
    # --- SECTION 1: Pulse Report ---
    st.markdown('<p class="section-title">Executive Summary</p>', unsafe_allow_html=True)
    report = st.session_state.pulse_report
    
    st.markdown(f"**{report.executive_summary}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.error("📉 **Top Issues**")
        for issue in report.top_issues:
            st.markdown(f"- {issue}")
    with col2:
        st.success("📈 **Positive Highlights**")
        for highlight in report.positive_highlights:
            st.markdown(f"- {highlight}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.published_report_url:
        st.success(f"✅ **Report Published!** [View in Google Docs]({st.session_state.published_report_url})")
    else:
        if st.button("📄 Publish Report to Google Docs", type="secondary"):
            with st.spinner("Drafting via MCP..."):
                mcp = MCPClient()
                formatted_doc = f"WEEKLY PRODUCT PULSE\n\n" \
                                f"EXECUTIVE SUMMARY:\n{report.executive_summary}\n\n" \
                                f"TOP ISSUES:\n" + "\n".join([f"• {issue}" for issue in report.top_issues]) + "\n\n" \
                                f"POSITIVE HIGHLIGHTS:\n" + "\n".join([f"• {h}" for h in report.positive_highlights])
                url = mcp.draft_google_doc("Weekly Product Pulse", formatted_doc)
                st.session_state.published_report_url = url
                st.rerun()

    # --- SECTION 2: Aggregated Metrics ---
    st.markdown('<p class="section-title">Theme Aggregations</p>', unsafe_allow_html=True)
    df = st.session_state.agg_df
    
    # Render pretty dataframe
    st.dataframe(
        df.style.background_gradient(cmap='Blues', subset=['total_reviews'])
                .background_gradient(cmap='RdYlGn', subset=['average_rating'], vmin=1, vmax=5),
        use_container_width=True,
        hide_index=True
    )

    # --- SECTION 3: Customer Support AI (Fee Explainer) ---
    st.markdown('<p class="section-title">Fee Mitigation (Auto-Drafted)</p>', unsafe_allow_html=True)
    if st.session_state.fee_explanations:
        for idx, item in enumerate(st.session_state.fee_explanations):
            review_text = item["review"]
            expl = item["explanation"]
            
            st.markdown(f"""
            <div class="fee-card">
                <h4>Original Complaint:</h4>
                <p>"{review_text}"</p>
                <h4>AI Drafted Response:</h4>
                <p>{expl.explanation}</p>
                <div class="fee-source">
                    <strong>Source Cited:</strong> "{expl.source_quote}"
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Approval Gate & Idempotency
            if review_text in st.session_state.published_emails:
                st.success(f"✅ **Email Drafted!** [View in Gmail]({st.session_state.published_emails[review_text]})")
            else:
                if st.button("✉️ Draft Email in Gmail", key=f"draft_{idx}"):
                    with st.spinner("Drafting via MCP..."):
                        mcp = MCPClient()
                        url = mcp.draft_gmail("Response to Fee Complaint", expl.explanation)
                        st.session_state.published_emails[review_text] = url
                        st.rerun()
    else:
        st.info("No fee-related complaints detected this week.")

    # --- SECTION 4: Raw Data Explorer ---
    st.markdown('<p class="section-title">Raw Categorized Reviews</p>', unsafe_allow_html=True)
    with st.expander("View all processed reviews"):
        raw_data = []
        for r in st.session_state.analyzed_reviews:
            raw_data.append({
                "Theme": r.theme,
                "Sentiment": r.sentiment,
                "Rating": r.rating,
                "Key Quote": r.key_quote,
                "PII Detected": r.has_pii,
                "Full Text": r.review_text
            })
        st.dataframe(pd.DataFrame(raw_data), use_container_width=True)
