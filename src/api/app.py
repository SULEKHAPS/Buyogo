import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
import pandas as pd

from ..data.loader import DataLoader
from ..data.preprocessor import DataPreprocessor
from ..analytics.reports import AnalyticsReports
from ..rag.embeddings import EmbeddingGenerator
from ..rag.vector_db import VectorDatabase
from ..rag.qa_system import QASystem

# Define API models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: Optional[List[str]] = None
    analytics_data: Optional[Dict[str, Any]] = None
    report_type: Optional[str] = None

class AnalyticsResponse(BaseModel):
    report_type: str
    data: Dict[str, Any]

# Create FastAPI app
app = FastAPI(
    title="Hotel Booking Analytics & QA API",
    description="API for querying hotel booking data and retrieving analytics reports",
    version="1.0.0"
)

# Global variables to store loaded components
_data_loader = None
_preprocessor = None
_analytics = None
_embedding_generator = None
_vector_db = None
_qa_system = None

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
VECTOR_DB_PATH = DATA_DIR / 'vector_db'

# Dependency to get the QA system
def get_qa_system():
    global _data_loader, _preprocessor, _analytics, _embedding_generator, _vector_db, _qa_system
    
    # Initialize components if not already initialized
    if _qa_system is None:
        # Initialize data loader
        _data_loader = DataLoader()
        
        # Load and preprocess data
        try:
            data = _data_loader.load_hotel_bookings()
            _preprocessor = DataPreprocessor()
            processed_data = _preprocessor.preprocess(data)
            
            # Initialize analytics
            _analytics = AnalyticsReports(processed_data)
            
            # Initialize embedding generator
            _embedding_generator = EmbeddingGenerator()
            
            # Check if vector database exists
            if os.path.exists(str(VECTOR_DB_PATH) + '.index'):
                # Load existing vector database
                _vector_db = VectorDatabase.load(VECTOR_DB_PATH)
            else:
                # Create new vector database
                _vector_db = VectorDatabase()
                
                # Generate embeddings
                embeddings = _embedding_generator.generate_booking_embeddings(processed_data)
                
                # Create data dictionary
                data_dict = {}
                for idx, row in processed_data.iterrows():
                    data_dict[str(idx)] = row.to_dict()
                
                # Add embeddings to vector database
                _vector_db.add_embeddings(embeddings, data_dict)
                
                # Save vector database
                os.makedirs(DATA_DIR, exist_ok=True)
                _vector_db.save(VECTOR_DB_PATH)
            
            # Initialize QA system
            _qa_system = QASystem(_vector_db, _embedding_generator)
            _qa_system.set_analytics(_analytics)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize QA system: {str(e)}")
    
    return _qa_system

@app.get("/")
async def root():
    return {"message": "Welcome to the Hotel Booking Analytics & QA API"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, qa_system: QASystem = Depends(get_qa_system)):
    """Query the system with a natural language question about hotel bookings."""
    try:
        result = qa_system.answer_query(request.query, k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/analytics/{report_type}", response_model=AnalyticsResponse)
async def get_analytics(report_type: str, qa_system: QASystem = Depends(get_qa_system)):
    """Get a specific analytics report."""
    try:
        # Check if report type is valid
        valid_reports = [
            "revenue_trends",
            "cancellation_rate",
            "geographical_distribution",
            "lead_time_distribution"
        ]
        
        if report_type not in valid_reports:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid report type. Valid types are: {', '.join(valid_reports)}"
            )
        
        # Get analytics report
        analytics = qa_system.analytics
        report_method = getattr(analytics, f'analyze_{report_type}')
        report_data = report_method()
        
        return {
            "report_type": report_type,
            "data": report_data
        }
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Report type '{report_type}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics report: {str(e)}")

@app.get("/analytics", response_model=Dict[str, Any])
async def get_all_analytics(qa_system: QASystem = Depends(get_qa_system)):
    """Get all analytics reports."""
    try:
        analytics = qa_system.analytics
        reports = analytics.generate_all_reports()
        return reports
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics reports: {str(e)}")