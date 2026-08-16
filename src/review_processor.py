import pandas as pd
from datetime import datetime, timedelta
from typing import List, TYPE_CHECKING
from src.models import Review

if TYPE_CHECKING:
    from src.models import AnalyzedReview

class ReviewProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_and_filter_reviews(self, weeks_back: int = 12) -> List[Review]:
        """
        Loads reviews from the CSV, handles missing data, filters by the last N weeks,
        and returns a list of Review Pydantic models.
        """
        # Load CSV
        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")

        # Basic cleaning
        df = df.dropna(subset=['review_id', 'date', 'rating', 'review_text'])
        df['review_text'] = df['review_text'].str.strip()
        df = df[df['review_text'] != ""]

        # Date parsing
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # 8-12 week filtering (relative to the most recent review to ensure data always exists for demo)
        if len(df) == 0:
            return []
            
        latest_date = df['date'].max()
        cutoff_date = latest_date - timedelta(weeks=weeks_back)
        
        filtered_df = df[df['date'] >= cutoff_date]

        # Map to Pydantic models
        reviews = []
        for _, row in filtered_df.iterrows():
            review = Review(
                review_id=str(row['review_id']),
                date=row['date'].date(),
                rating=int(row['rating']),
                review_text=str(row['review_text'])
            )
            reviews.append(review)

        return reviews

    def aggregate_by_theme(self, analyzed_reviews: List['AnalyzedReview']) -> pd.DataFrame:
        """
        Takes a list of AnalyzedReview and returns a Pandas DataFrame aggregated by theme.
        Calculates:
        - Total reviews per theme
        - Average rating per theme
        - Count of positive/neutral/negative sentiments
        """
        if not analyzed_reviews:
            return pd.DataFrame()
            
        data = []
        for r in analyzed_reviews:
            data.append({
                'theme': r.theme,
                'rating': r.rating,
                'sentiment': r.sentiment
            })
            
        df = pd.DataFrame(data)
        
        # Group by theme
        agg_df = df.groupby('theme').agg(
            total_reviews=('theme', 'count'),
            average_rating=('rating', 'mean'),
            positive_sentiment=('sentiment', lambda x: (x == 'Positive').sum()),
            neutral_sentiment=('sentiment', lambda x: (x == 'Neutral').sum()),
            negative_sentiment=('sentiment', lambda x: (x == 'Negative').sum())
        ).reset_index()
        
        # Round the average rating
        agg_df['average_rating'] = agg_df['average_rating'].round(2)
        
        return agg_df
