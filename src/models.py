from datetime import date
from pydantic import BaseModel, Field

class Review(BaseModel):
    review_id: str
    date: date
    rating: int = Field(ge=1, le=5)
    review_text: str

from typing import Optional

class AnalyzedReview(Review):
    theme: str
    sentiment: str
    key_quote: Optional[str] = None
    has_pii: bool = False

from typing import List

class PulseReport(BaseModel):
    executive_summary: str
    top_issues: List[str]
    positive_highlights: List[str]

class FeeExplanation(BaseModel):
    is_applicable: bool
    explanation: str
    source_quote: str
