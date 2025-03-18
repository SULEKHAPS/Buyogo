import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
from transformers import pipeline
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDatabase
from ..analytics.reports import AnalyticsReports

class QASystem:
    """Class for answering natural language queries about hotel booking data."""
    
    def __init__(self, vector_db: VectorDatabase, embedding_generator: EmbeddingGenerator, model_name: str = 'google/flan-t5-base'):
        """Initialize the QASystem.
        
        Args:
            vector_db: Vector database containing booking embeddings
            embedding_generator: Embedding generator for encoding queries
            model_name: Name of the question answering model to use
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.model_name = model_name
        
        # Initialize the question answering pipeline
        self.qa_pipeline = pipeline('text2text-generation', model=model_name, max_length=512)
        
        # Initialize analytics reports
        self.analytics = None
    
    def set_analytics(self, analytics: AnalyticsReports) -> None:
        """Set the analytics reports object.
        
        Args:
            analytics: AnalyticsReports object containing pre-computed analytics
        """
        self.analytics = analytics
    
    def answer_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Answer a natural language query about hotel booking data.
        
        Args:
            query: Natural language query
            k: Number of similar bookings to retrieve
            
        Returns:
            Dictionary containing the answer and supporting information
        """
        # Check if the query is about analytics
        analytics_result = self._check_analytics_query(query)
        if analytics_result:
            return analytics_result
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search for similar bookings
        search_results = self.vector_db.search(query_embedding, k=k)
        
        # Extract context from search results
        context = self._extract_context(search_results)
        
        # Generate answer
        answer = self._generate_answer(query, context)
        
        # Prepare result
        result = {
            'query': query,
            'answer': answer,
            'sources': [result['booking_id'] for result in search_results]
        }
        
        return result
    
    def _check_analytics_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if the query is about analytics and return pre-computed analytics if available.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary containing analytics results if applicable, None otherwise
        """
        if self.analytics is None:
            return None
        
        # Define keywords for each analytics report
        analytics_keywords = {
            'revenue_trends': ['revenue', 'sales', 'income', 'earnings', 'money', 'profit'],
            'cancellation_rate': ['cancel', 'cancellation', 'canceled', 'cancelled'],
            'geographical_distribution': ['country', 'countries', 'geography', 'location', 'origin', 'geographical'],
            'lead_time_distribution': ['lead time', 'booking time', 'advance', 'days before']
        }
        
        # Check if query contains keywords for any analytics report
        query_lower = query.lower()
        for report_name, keywords in analytics_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Get the analytics report
                try:
                    report_method = getattr(self.analytics, f'analyze_{report_name}')
                    report_data = report_method()
                    
                    # Generate a natural language summary of the report
                    summary = self._summarize_analytics(report_name, report_data, query)
                    
                    return {
                        'query': query,
                        'answer': summary,
                        'analytics_data': report_data,
                        'report_type': report_name
                    }
                except (AttributeError, Exception) as e:
                    # If there's an error, continue with regular QA
                    print(f"Error retrieving analytics report: {e}")
                    pass
        
        return None
    
    def _extract_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract context from search results.
        
        Args:
            search_results: List of search results from vector database
            
        Returns:
            String containing context for question answering
        """
        context_parts = []
        
        for result in search_results:
            if 'data' in result:
                # Convert booking data to text
                booking_data = result['data']
                booking_text = "Booking information: \n"
                
                for key, value in booking_data.items():
                    booking_text += f"{key}: {value}\n"
                
                context_parts.append(booking_text)
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate an answer to the query based on the context.
        
        Args:
            query: Natural language query
            context: Context for question answering
            
        Returns:
            Generated answer
        """
        # Prepare input for the QA pipeline
        input_text = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
        
        # Generate answer
        result = self.qa_pipeline(input_text)[0]['generated_text']
        
        return result
    
    def _summarize_analytics(self, report_name: str, report_data: Dict[str, Any], query: str) -> str:
        """Generate a natural language summary of analytics data.
        
        Args:
            report_name: Name of the analytics report
            report_data: Analytics report data
            query: Original query
            
        Returns:
            Natural language summary of the analytics data
        """
        # Prepare input for the summarization pipeline
        input_text = f"Summarize the following {report_name.replace('_', ' ')} data in response to the question: {query}\n\n"
        input_text += json.dumps(report_data, indent=2)
        
        # Generate summary
        summary = self.qa_pipeline(input_text)[0]['generated_text']
        
        return summary