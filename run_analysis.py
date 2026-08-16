import os
from dotenv import load_dotenv
from src.review_processor import ReviewProcessor
from src.llm_analyzer import ReviewAnalyzer

def main():
    # Load environment variables (API keys)
    load_dotenv()
    
    # Initialize processor
    processor = ReviewProcessor("data/reviews.csv")
    
    # 1. Load Reviews
    print("Loading reviews from CSV...")
    reviews = processor.load_and_filter_reviews(weeks_back=24)
    print(f"Loaded {len(reviews)} reviews.")
    
    # Check if API key is present
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY not found in environment. Please set it to test the LLM integration.")
        return
        
    # 2. Analyze Reviews
    print("Sending reviews to LLM for analysis... (This might take a moment)")
    analyzer = ReviewAnalyzer()
    
    # We will only analyze the first 5 to save API quota for this test
    sample_reviews = reviews[:5]
    analyzed_reviews = analyzer.analyze_batch(sample_reviews)
    
    print("\n--- Analyzed Sample ---")
    for r in analyzed_reviews:
        print(f"Review: {r.review_text[:50]}...")
        quote_text = f'"{r.key_quote}"' if r.key_quote else "(No valid quote extracted)"
        pii_warning = "[PII DETECTED]" if r.has_pii else ""
        print(f"-> Theme: {r.theme} | Sentiment: {r.sentiment} {pii_warning}")
        print(f"-> Quote: {quote_text}\n")

        
    # 3. Aggregate Data
    print("Aggregating data using Pandas...")
    agg_df = processor.aggregate_by_theme(analyzed_reviews)
    print("\n--- Aggregated DataFrame ---")
    print(agg_df.to_string())
    
    # 4. Generate Weekly Product Pulse
    print("\nGenerating Weekly Product Pulse... (This might take a moment)")
    # Collect quotes
    valid_quotes = [r.key_quote for r in analyzed_reviews if r.key_quote]
    
    pulse_report = analyzer.generate_pulse_report(agg_df.to_json(), valid_quotes)
    print("\n=============================================")
    print("        WEEKLY PRODUCT PULSE")
    print("=============================================\n")
    print(f"EXECUTIVE SUMMARY:\n{pulse_report.executive_summary}\n")
    print("TOP ISSUES:")
    for issue in pulse_report.top_issues:
        print(f" - {issue}")
    print("\nPOSITIVE HIGHLIGHTS:")
    for highlight in pulse_report.positive_highlights:
        print(f" - {highlight}")
    print("\n=============================================")
    
    # 5. Fee Explainer
    print("\nRunning Fee Explainer on Pricing/Fees complaints...")
    from src.fee_explainer import FeeExplainer
    
    with open("data/fee_schedule.txt", "r") as f:
        fee_schedule = f.read()
        
    fee_explainer = FeeExplainer(fee_schedule_text=fee_schedule)
    
    pricing_reviews = [r for r in analyzed_reviews if r.theme == "Pricing/Fees"]
    if not pricing_reviews:
        print("No Pricing/Fees reviews found in this sample.")
    
    for r in pricing_reviews:
        print(f"\nReview: {r.review_text}")
        explanation = fee_explainer.explain_fee(r.review_text)
        if explanation.is_applicable:
            print(f"Explanation: {explanation.explanation}")
            print(f"Source Cited: \"{explanation.source_quote}\"")
        else:
            print(f"Blocked or N/A: {explanation.explanation}")

if __name__ == "__main__":
    main()
