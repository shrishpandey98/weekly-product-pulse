# Weekly Product Pulse 📈

An AI-Powered Customer Feedback Intelligence application that analyzes thousands of customer reviews in seconds and automates the creation of executive summaries and customer support emails using Llama-3.1 via Groq.

## Features
- **Blazing Fast LLM Analysis**: Leverages Groq's Llama-3.1 to categorize reviews, extract sentiment, perform PII validation, and pull verbatim key quotes.
- **Robust Pipeline**: Includes advanced batch processing with ThreadPool concurrency and exponential backoff retry logic to handle rate-limiting automatically.
- **Interactive Metrics Dashboard**: Built with Streamlit, providing a sleek, responsive UI with Pandas background gradients. 
- **Customer Support AI**: Automatically drafts polite, well-reasoned explanations for negative reviews, grounding the response in your company's actual fee schedule.
- **Human-in-the-Loop & Idempotency**: Strict per-session execution states prevent duplicate actions or accidental LLM reprompting.
- **Real Google Workspace Integration**: Seamlessly authenticates with your Google account via OAuth 2.0 to publish the Pulse Report directly to Google Docs, or save AI-drafted responses straight to your Gmail Drafts!

## Installation

1. Clone the repository and navigate into the project directory.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
5. *(Optional)* Add your `credentials.json` OAuth 2.0 Client ID file from the Google Cloud Console to enable Google Docs/Gmail integration.

## Usage

Start the Streamlit application:
```bash
PYTHONPATH=. ./venv/bin/streamlit run app.py
```
Open your browser to `http://localhost:8501`. 
Upload the `large_reviews.csv` file (or use the default) and click **"Generate Weekly Pulse"**.

## Architecture
- `src/models.py`: Pydantic schemas enforcing strict data validation.
- `src/llm_analyzer.py`: The core LLM integration, optimized for batch processing.
- `src/review_processor.py`: Handles pandas DataFrame aggregations.
- `src/fee_explainer.py`: Specialized RAG/prompt-chaining for customer support workflows.
- `mcp/client.py`: The Google SDK abstraction.
- `app.py`: The Streamlit frontend and State Machine coordinator.

## Testing
Run the test suite using `pytest`:
```bash
PYTHONPATH=. ./venv/bin/pytest
```
