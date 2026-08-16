import pytest
from unittest.mock import MagicMock, patch
from datetime import date
from src.models import Review, AnalyzedReview
from src.llm_analyzer import ReviewAnalyzer

@patch('src.llm_analyzer.Groq')
def test_analyze_review_success(mock_client_class):
    with patch('src.llm_analyzer.os.environ.get') as mock_env:
        mock_env.return_value = 'mock_api_key'
        
        # Setup mock client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the generation response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"theme": "UI/UX", "sentiment": "Positive", "key_quote": "faster than before", "has_pii": false}'
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        analyzer = ReviewAnalyzer()
        review = Review(
            review_id="REV-001",
            date=date(2026, 8, 1),
            rating=5,
            review_text="Love the new interface! It is so much faster than before."
        )

        analyzed = analyzer.analyze_review(review)

        assert isinstance(analyzed, AnalyzedReview)
        assert analyzed.theme == "UI/UX"
        assert analyzed.sentiment == "Positive"
        assert analyzed.key_quote == "faster than before"
        assert analyzed.has_pii is False
        assert analyzed.review_text == review.review_text

@patch('src.llm_analyzer.Groq')
def test_analyze_review_verbatim_validation_fails(mock_client_class):
    with patch('src.llm_analyzer.os.environ.get') as mock_env:
        mock_env.return_value = 'mock_api_key'
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # LLM hallucinates a quote not in the original text
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"theme": "UI/UX", "sentiment": "Positive", "key_quote": "way faster than before", "has_pii": false}'
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        analyzer = ReviewAnalyzer()
        review = Review(
            review_id="REV-001",
            date=date(2026, 8, 1),
            rating=5,
            review_text="Love the new interface! It is so much faster than before."
        )

        analyzed = analyzer.analyze_review(review)
        # Should be None because "way faster than before" is not in the text
        assert analyzed.key_quote is None

@patch('src.llm_analyzer.Groq')
def test_analyze_review_pii_detection(mock_client_class):
    with patch('src.llm_analyzer.os.environ.get') as mock_env:
        mock_env.return_value = 'mock_api_key'
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # LLM detects PII
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"theme": "Customer Support", "sentiment": "Positive", "key_quote": "called me", "has_pii": true}'
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        analyzer = ReviewAnalyzer()
        review = Review(
            review_id="REV-002",
            date=date(2026, 8, 1),
            rating=5,
            review_text="John Doe called me back."
        )

        analyzed = analyzer.analyze_review(review)
        assert analyzed.has_pii is True
        # Quote should be dropped because of PII
        assert analyzed.key_quote is None

@patch('src.llm_analyzer.Groq')
def test_analyze_review_fallback_on_error(mock_client_class):
    with patch('src.llm_analyzer.os.environ.get') as mock_env:
        mock_env.return_value = 'mock_api_key'
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the generation response to fail JSON parsing
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Not a json'
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)
        
        analyzer = ReviewAnalyzer()
        review = Review(
            review_id="REV-001",
            date=date(2026, 8, 1),
            rating=5,
            review_text="Some text"
        )

        analyzed = analyzer.analyze_review(review)

        assert analyzed.theme == "General/Other"
        assert analyzed.sentiment == "Neutral"

@patch('src.llm_analyzer.Groq')
def test_generate_pulse_report(mock_client_class):
    with patch('src.llm_analyzer.os.environ.get') as mock_env:
        mock_env.return_value = 'mock_api_key'
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"executive_summary": "Overall positive.", "top_issues": ["Bugs"], "positive_highlights": ["UI"]}'
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)
        
        analyzer = ReviewAnalyzer()
        report = analyzer.generate_pulse_report('{"theme": {"0": "UI/UX"}}', ["Great UI"])
        
        assert report.executive_summary == "Overall positive."
        assert report.top_issues == ["Bugs"]
        assert report.positive_highlights == ["UI"]
