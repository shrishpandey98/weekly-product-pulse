import pytest
import os
from datetime import date
from src.review_processor import ReviewProcessor

def test_load_and_filter_reviews():
    # Arrange
    csv_path = os.path.join(os.path.dirname(__file__), '../data/reviews.csv')
    processor = ReviewProcessor(csv_path)
    
    # Act
    reviews = processor.load_and_filter_reviews(weeks_back=12)
    
    # Assert
    assert len(reviews) > 0, "Should load at least some reviews"
    assert all(isinstance(r.date, date) for r in reviews), "Dates should be parsed correctly"
    assert all(1 <= r.rating <= 5 for r in reviews), "Ratings should be bounded"
    
    # Check that older reviews are filtered out
    # 12 weeks from the latest date (2026-08-11) is approx May 19, 2026.
    # REV-011 and REV-016 are May 10 and May 01, they should be filtered out.
    review_ids = [r.review_id for r in reviews]
    assert "REV-011" not in review_ids, "Older reviews should be filtered"
    assert "REV-016" not in review_ids, "Older reviews should be filtered"
    assert "REV-001" in review_ids, "Recent reviews should be included"

def test_aggregate_by_theme():
    from src.models import AnalyzedReview
    processor = ReviewProcessor("dummy_path.csv")
    
    analyzed_reviews = [
        AnalyzedReview(review_id="1", date=date(2026, 8, 1), rating=5, review_text="A", theme="UI/UX", sentiment="Positive"),
        AnalyzedReview(review_id="2", date=date(2026, 8, 1), rating=4, review_text="B", theme="UI/UX", sentiment="Positive"),
        AnalyzedReview(review_id="3", date=date(2026, 8, 1), rating=2, review_text="C", theme="Pricing/Fees", sentiment="Negative"),
        AnalyzedReview(review_id="4", date=date(2026, 8, 1), rating=3, review_text="D", theme="Pricing/Fees", sentiment="Neutral"),
    ]
    
    df = processor.aggregate_by_theme(analyzed_reviews)
    
    assert len(df) == 2, "Should aggregate into 2 themes"
    
    ui_row = df[df['theme'] == 'UI/UX'].iloc[0]
    assert ui_row['total_reviews'] == 2
    assert ui_row['average_rating'] == 4.5
    assert ui_row['positive_sentiment'] == 2
    
    pricing_row = df[df['theme'] == 'Pricing/Fees'].iloc[0]
    assert pricing_row['total_reviews'] == 2
    assert pricing_row['average_rating'] == 2.5
    assert pricing_row['negative_sentiment'] == 1
    assert pricing_row['neutral_sentiment'] == 1
