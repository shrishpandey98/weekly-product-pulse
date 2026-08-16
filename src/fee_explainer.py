import os
from groq import Groq
from src.models import FeeExplanation

class FeeExplainer:
    def __init__(self, fee_schedule_text: str, api_key: str = None):
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
        self.fee_schedule_text = fee_schedule_text

    def explain_fee(self, review_text: str) -> FeeExplanation:
        """
        Takes a negative review, checks if it's complaining about a fee in the schedule,
        and drafts a factual explanation with a source quote.
        """
        prompt = f"""
        You are a helpful and polite Customer Support agent. A user has left a negative review 
        complaining about a fee. You must review the provided Fee Schedule to draft a factual response.

        Fee Schedule:
        \"\"\"{self.fee_schedule_text}\"\"\"

        User Review:
        "{review_text}"

        Tasks:
        1. Set `is_applicable` to true if the review mentions a fee listed in the Fee Schedule.
        2. Draft a polite `explanation` addressing the user's concern directly based on the schedule.
        3. Extract the exact `source_quote` from the Fee Schedule that justifies this fee. The quote MUST be a verbatim substring of the Fee Schedule text.
        
        Respond ONLY with a valid JSON object matching this schema:
        {{
            "is_applicable": boolean,
            "explanation": "string",
            "source_quote": "string"
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
                
                is_applicable = data.get("is_applicable", False)
                explanation = data.get("explanation", "")
                source_quote = data.get("source_quote", "")
                
                # Verbatim Validation: Check if quote is actually in the Fee Schedule
                if is_applicable and source_quote and source_quote not in self.fee_schedule_text:
                    # LLM hallucinated the source quote! We cannot trust the explanation.
                    return FeeExplanation(
                        is_applicable=False,
                        explanation="Automated explanation blocked due to source validation failure (hallucination detected).",
                        source_quote=""
                    )
                    
                return FeeExplanation(
                    is_applicable=is_applicable,
                    explanation=explanation,
                    source_quote=source_quote
                )
            except Exception as e:
                if ("429" in str(e) or "rate_limit" in str(e)) and attempt < 3:
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"Error explaining fee: {e}")
                return FeeExplanation(
                    is_applicable=False,
                    explanation="Could not generate explanation due to an error.",
                    source_quote=""
                )

