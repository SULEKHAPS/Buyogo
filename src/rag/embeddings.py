import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingGenerator:
    """Class for generating vector embeddings from hotel booking data."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the EmbeddingGenerator with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        # Load the model
        self.model = SentenceTransformer(model_name)
        
    def generate_booking_embeddings(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate embeddings for hotel booking data.
        
        Args:
            data: DataFrame containing the preprocessed hotel bookings data
            
        Returns:
            Dictionary containing booking IDs and their corresponding embeddings
        """
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Ensure we have a unique identifier for each booking
        if 'booking_id' not in df.columns:
            df['booking_id'] = df.index.astype(str)
        
        # Create text representations of each booking
        booking_texts = self._create_booking_texts(df)
        
        # Generate embeddings
        embeddings = self.model.encode(booking_texts, show_progress_bar=True)
        
        # Create a dictionary mapping booking IDs to embeddings
        embedding_dict = {}
        for i, idx in enumerate(df['booking_id']):
            embedding_dict[str(idx)] = embeddings[i]
        
        return embedding_dict
    
    def _create_booking_texts(self, data: pd.DataFrame) -> List[str]:
        """Create text representations of bookings for embedding generation.
        
        Args:
            data: DataFrame containing the preprocessed hotel bookings data
            
        Returns:
            List of text representations for each booking
        """
        booking_texts = []
        
        for _, row in data.iterrows():
            # Create a text representation of the booking
            text_parts = []
            
            # Add hotel information if available
            if 'hotel' in row:
                text_parts.append(f"Hotel: {row['hotel']}")
            
            # Add arrival date information if available
            if 'arrival_date' in row:
                text_parts.append(f"Arrival date: {row['arrival_date']}")
            elif all(col in row.index for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
                text_parts.append(f"Arrival date: {row['arrival_date_year']}-{row['arrival_date_month']}-{row['arrival_date_day_of_month']}")
            
            # Add stay duration information if available
            if 'total_nights' in row:
                text_parts.append(f"Stay duration: {row['total_nights']} nights")
            elif all(col in row.index for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
                total_nights = row['stays_in_weekend_nights'] + row['stays_in_week_nights']
                text_parts.append(f"Stay duration: {total_nights} nights")
            
            # Add guest information if available
            if 'adults' in row:
                text_parts.append(f"Adults: {row['adults']}")
            if 'children' in row:
                text_parts.append(f"Children: {row['children']}")
            if 'babies' in row:
                text_parts.append(f"Babies: {row['babies']}")
            
            # Add country information if available
            if 'country' in row:
                text_parts.append(f"Country: {row['country']}")
            
            # Add price information if available
            if 'adr' in row:
                text_parts.append(f"Average daily rate: {row['adr']}")
            if 'total_price' in row:
                text_parts.append(f"Total price: {row['total_price']}")
            
            # Add cancellation information if available
            if 'is_canceled' in row:
                status = "Canceled" if row['is_canceled'] == 1 else "Not canceled"
                text_parts.append(f"Status: {status}")
            
            # Add lead time information if available
            if 'lead_time' in row:
                text_parts.append(f"Lead time: {row['lead_time']} days")
            
            # Join all parts into a single text representation
            booking_text = " | ".join(text_parts)
            booking_texts.append(booking_text)
        
        return booking_texts
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate an embedding for a query string.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector for the query
        """
        return self.model.encode(query)