import pytest
from unittest.mock import MagicMock, patch
from src.fee_explainer import FeeExplainer
from src.models import FeeExplanation

FEE_SCHEDULE = """
Early Withdrawal Charge: If you withdraw funds from a Fixed Term Deposit before the maturity date, an early withdrawal charge of $50.00 will be applied to the transaction.
"""

@patch('src.fee_explainer.Groq')
def test_explain_fee_success(mock_client_class):
    with patch('src.fee_explainer.os.environ.get') as mock_env:
        mock_env.return_value = 'mock_api_key'
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"is_applicable": true, "explanation": "I understand your frustration. However, as per our fee schedule...", "source_quote": "an early withdrawal charge of $50.00 will be applied"}'
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        explainer = FeeExplainer(fee_schedule_text=FEE_SCHEDULE)
        result = explainer.explain_fee("I got hit with a $50 early withdrawal charge, this is unfair!")

        assert result.is_applicable is True
        assert result.source_quote == "an early withdrawal charge of $50.00 will be applied"
        assert result.explanation.startswith("I understand")

@patch('src.fee_explainer.Groq')
def test_explain_fee_hallucination_blocked(mock_client_class):
    with patch('src.fee_explainer.os.environ.get') as mock_env:
        mock_env.return_value = 'mock_api_key'
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # LLM hallucinates a source quote
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"is_applicable": true, "explanation": "Sorry about that.", "source_quote": "a fee of $100 will be applied"}'
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        explainer = FeeExplainer(fee_schedule_text=FEE_SCHEDULE)
        result = explainer.explain_fee("I got hit with an early withdrawal charge.")

        # Should be blocked
        assert result.is_applicable is False
        assert "blocked" in result.explanation
