import os
from groq import Groq
from typing import List, Literal
from pydantic import BaseModel
from src.models import Review, AnalyzedReview

# Define the acceptable themes
THEMES = ["UI/UX", "Pricing/Fees", "Performance/Stability", "Customer Support", "General/Other"]

class ReviewAnalyzer:
    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                    api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass
        if not api_key:
            raise ValueError("API key for Groq is not set.")
        
        self.client = Groq(api_key=api_key)

    def analyze_review(self, review: Review) -> AnalyzedReview:
        """
        Takes a single Review, asks the LLM to classify its theme and sentiment,
        extracts a verbatim quote, checks for PII, and returns an AnalyzedReview.
        """
        prompt = f"""
        Analyze the following user review for a financial application.
        1. Assign it to the most relevant theme from this list: {THEMES}.
        2. Determine the sentiment (Positive, Neutral, Negative).
        3. Extract a short, concise verbatim quote from the text that highlights the main point (max 15 words).
           The quote MUST be an exact substring of the original text.
        4. Set has_pii to true if the review contains ANY Personally Identifiable Information (names, emails, phone numbers).

        Review Text: "{review.review_text}"
        
        Respond ONLY with a valid JSON object matching this schema:
        {{
            "theme": "string from {THEMES}",
            "sentiment": "Positive" or "Neutral" or "Negative",
            "key_quote": "string",
            "has_pii": boolean
        }}
        """
        
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model='llama-3.1-8b-instant',
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                
                import json
                data = json.loads(response.choices[0].message.content)
                
                extracted_quote = data.get("key_quote", "")
                has_pii = data.get("has_pii", False)
                
                # Verbatim Validation: Check if quote is actually in the text
                if extracted_quote and extracted_quote not in review.review_text:
                    extracted_quote = None
                    
                # PII Safety: Discard quote if PII is detected anywhere in the review
                if has_pii:
                    extracted_quote = None
                
                return AnalyzedReview(
                    review_id=review.review_id,
                    date=review.date,
                    rating=review.rating,
                    review_text=review.review_text,
                    theme=data.get("theme", "General/Other"),
                    sentiment=data.get("sentiment", "Neutral"),
                    key_quote=extracted_quote,
                    has_pii=has_pii
                )
            except Exception as e:
                if ("429" in str(e) or "rate_limit" in str(e)) and attempt < 3:
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"Error classifying review {review.review_id}: {e}")
                # Fallback if parsing fails
                return AnalyzedReview(
                    review_id=review.review_id,
                    date=review.date,
                    rating=review.rating,
                    review_text=review.review_text,
                    theme="General/Other",
                    sentiment="Neutral",
                    key_quote=None,
                    has_pii=False
                )

    def analyze_batch_chunk(self, chunk: List[Review]) -> List[AnalyzedReview]:
        """
        Analyzes a chunk of up to 10 reviews in a SINGLE LLM prompt call for 10x speedup.
        """
        if not chunk:
            return []
            
        formatted_reviews = "\n".join([
            f"- Review ID: {r.review_id} | Text: \"{r.review_text}\""
            for r in chunk
        ])
        
        prompt = f"""
        Analyze the following batch of user reviews for a financial application.
        For EACH review:
        1. Assign it to the most relevant theme from this list: {THEMES}.
        2. Determine the sentiment (Positive, Neutral, Negative).
        3. Extract a short verbatim key_quote (max 15 words) that MUST be an exact substring of the review.
        4. Set has_pii to true if it contains names, emails, phone numbers.

        Reviews to analyze:
        {formatted_reviews}

        Respond ONLY with a valid JSON object formatted as:
        {{
            "results": [
                {{
                    "review_id": "exact Review ID",
                    "theme": "string from {THEMES}",
                    "sentiment": "Positive" or "Neutral" or "Negative",
                    "key_quote": "string",
                    "has_pii": boolean
                }}
            ]
        }}
        """
        
        results_map = {}
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model='llama-3.1-8b-instant',
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                import json
                raw_content = response.choices[0].message.content
                data = json.loads(raw_content)
                items = data.get("results", [])
                if isinstance(items, list):
                    for item in items:
                        rid = item.get("review_id")
                        if rid:
                            results_map[rid] = item
                break
            except Exception as e:
                if ("429" in str(e) or "rate_limit" in str(e)) and attempt < 2:
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"Error in batch chunk: {e}")
                break

        analyzed = []
        for r in chunk:
            item = results_map.get(r.review_id, {})
            extracted_quote = item.get("key_quote", "")
            has_pii = item.get("has_pii", False)
            if extracted_quote and extracted_quote not in r.review_text:
                extracted_quote = None
            if has_pii:
                extracted_quote = None
            
            analyzed.append(AnalyzedReview(
                review_id=r.review_id,
                date=r.date,
                rating=r.rating,
                review_text=r.review_text,
                theme=item.get("theme", "General/Other"),
                sentiment=item.get("sentiment", "Neutral"),
                key_quote=extracted_quote,
                has_pii=has_pii
            ))
        return analyzed

    def analyze_batch(self, reviews: List[Review]) -> List[AnalyzedReview]:
        """
        Analyzes a list of reviews using mini-batches of 10.
        """
        analyzed = []
        chunk_size = 10
        for i in range(0, len(reviews), chunk_size):
            chunk = reviews[i:i + chunk_size]
            analyzed.extend(self.analyze_batch_chunk(chunk))
        return analyzed

    def generate_pulse_report(self, aggregated_df_json: str, quotes: List[str]) -> "src.models.PulseReport":
        """
        Takes the aggregated JSON DataFrame and qualitative quotes, 
        and generates an executive PulseReport.
        """
        from src.models import PulseReport
        
        prompt = f"""
        You are a seasoned Product Manager writing a weekly "Product Pulse" executive summary.
        Based on the following aggregated customer review metrics and key quotes, generate a brief, high-level report.

        Aggregated Metrics (JSON):
        {aggregated_df_json}

        Key Quotes:
        {quotes}
        
        Respond ONLY with a valid JSON object matching this schema:
        {{
            "executive_summary": "string",
            "top_issues": ["string"],
            "positive_highlights": ["string"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model='llama-3.1-8b-instant',
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            import json
            data = json.loads(response.choices[0].message.content)
            return PulseReport(
                executive_summary=data.get("executive_summary", "Failed to generate summary."),
                top_issues=data.get("top_issues", []),
                positive_highlights=data.get("positive_highlights", [])
            )
        except Exception as e:
            print(f"Error generating pulse report: {e}")
            return PulseReport(
                executive_summary="Could not generate executive summary due to an error.",
                top_issues=["Error occurred during generation."],
                positive_highlights=[]
            )

