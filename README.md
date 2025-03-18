# LLM-Powered Booking Analytics & QA System

This project processes hotel booking data, extracts insights, and enables retrieval-augmented question answering (RAG). The system provides analytics on revenue trends, cancellation rates, geographical distribution, and booking lead times, while also answering natural language queries about the data.

## Project Structure

```
├── data/                  # Data storage directory
│   └── hotel_bookings.csv # Sample dataset
├── notebooks/             # Jupyter notebooks for exploration and analysis
│   └── data_exploration.ipynb
├── src/                   # Source code
│   ├── api/               # API implementation
│   │   ├── __init__.py
│   │   ├── main.py        # FastAPI application
│   │   ├── models.py      # API models
│   │   └── routes.py      # API endpoints
│   ├── analytics/         # Analytics implementation
│   │   ├── __init__.py
│   │   └── reports.py     # Analytics generation
│   ├── data/              # Data processing
│   │   ├── __init__.py
│   │   ├── loader.py      # Data loading utilities
│   │   └── preprocessor.py # Data cleaning and preprocessing
│   ├── rag/               # Retrieval-Augmented Generation
│   │   ├── __init__.py
│   │   ├── embeddings.py  # Vector embeddings generation
│   │   ├── llm.py         # LLM integration
│   │   └── retriever.py   # Vector database retrieval
│   └── utils/             # Utility functions
│       ├── __init__.py
│       └── config.py      # Configuration
├── tests/                 # Test cases
│   ├── __init__.py
│   ├── test_analytics.py
│   ├── test_api.py
│   └── test_rag.py
├── .env.example           # Example environment variables
├── .gitignore             # Git ignore file
├── requirements.txt       # Project dependencies
└── setup.py               # Package setup
```

## Setup Instructions

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Download the dataset and place it in the `data/` directory
6. Run the API: `uvicorn src.api.main:app --reload`

## API Endpoints

- `POST /analytics` - Returns analytics reports
- `POST /ask` - Answers booking-related questions
- `GET /health` - Checks system status (optional)

## Sample Queries

- "Show me total revenue for July 2017."
- "Which locations had the highest booking cancellations?"
- "What is the average price of a hotel booking?"

## Implementation Details

- **Data Processing**: Pandas for data cleaning and preprocessing
- **Analytics**: Pandas, NumPy, Matplotlib, and Seaborn for generating insights
- **Vector Database**: FAISS for storing embeddings
- **LLM**: Sentence-Transformers for embeddings, Hugging Face's Transformers for LLM
- **API**: FastAPI for REST API implementation
- **Testing**: Pytest for unit and integration tests

## Performance Evaluation

- Accuracy of Q&A responses is evaluated using a set of test queries
- API response time is measured and optimized for retrieval speed

## Challenges and Solutions

(To be filled after implementation)
