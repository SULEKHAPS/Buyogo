import os
import numpy as np
import pandas as pd
import faiss
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from .embeddings import EmbeddingGenerator

class VectorDatabase:
    """Class for storing and retrieving vector embeddings using FAISS."""
    
    def __init__(self, dimension: int = 384, index_type: str = 'L2'):
        """Initialize the VectorDatabase.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index to use ('L2' or 'IP' for inner product)
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == 'L2':
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'IP':
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}. Use 'L2' or 'IP'.")
        
        # Initialize mapping from index to booking ID
        self.id_mapping = {}
        # Initialize reverse mapping from booking ID to index
        self.reverse_mapping = {}
        # Initialize data storage
        self.data = {}
        
    def add_embeddings(self, embedding_dict: Dict[str, np.ndarray], data_dict: Optional[Dict[str, Any]] = None) -> None:
        """Add embeddings to the vector database.
        
        Args:
            embedding_dict: Dictionary mapping booking IDs to embedding vectors
            data_dict: Optional dictionary mapping booking IDs to original data
        """
        # Convert embeddings to numpy array
        ids = list(embedding_dict.keys())
        embeddings = np.array([embedding_dict[id] for id in ids], dtype=np.float32)
        
        # Add embeddings to FAISS index
        start_idx = len(self.id_mapping)
        self.index.add(embeddings)
        
        # Update mappings
        for i, id in enumerate(ids):
            idx = start_idx + i
            self.id_mapping[idx] = id
            self.reverse_mapping[id] = idx
            
            # Store original data if provided
            if data_dict and id in data_dict:
                self.data[id] = data_dict[id]
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings in the database.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of dictionaries containing search results with booking IDs, distances, and data
        """
        # Ensure query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.id_mapping):
                continue  # Skip invalid indices
                
            booking_id = self.id_mapping[idx]
            result = {
                'booking_id': booking_id,
                'distance': float(distances[0][i])
            }
            
            # Add original data if available
            if booking_id in self.data:
                result['data'] = self.data[booking_id]
                
            results.append(result)
            
        return results
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save the vector database to disk.
        
        Args:
            file_path: Path to save the database to
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = file_path.with_suffix('.index')
        faiss.write_index(self.index, str(index_path))
        
        # Save mappings and data
        metadata = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'id_mapping': self.id_mapping,
            'reverse_mapping': self.reverse_mapping,
            'data': self.data
        }
        metadata_path = file_path.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'VectorDatabase':
        """Load a vector database from disk.
        
        Args:
            file_path: Path to load the database from
            
        Returns:
            Loaded VectorDatabase instance
        """
        file_path = Path(file_path)
        
        # Load mappings and data
        metadata_path = file_path.with_suffix('.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(dimension=metadata['dimension'], index_type=metadata['index_type'])
        
        # Load FAISS index
        index_path = file_path.with_suffix('.index')
        instance.index = faiss.read_index(str(index_path))
        
        # Restore mappings and data
        instance.id_mapping = metadata['id_mapping']
        instance.reverse_mapping = metadata['reverse_mapping']
        instance.data = metadata['data']
        
        return instance
    
    def get_booking_data(self, booking_id: str) -> Optional[Dict[str, Any]]:
        """Get the original data for a booking ID.
        
        Args:
            booking_id: Booking ID to get data for
            
        Returns:
            Original data for the booking ID, or None if not found
        """
        return self.data.get(booking_id)
    
    def __len__(self) -> int:
        """Get the number of embeddings in the database.
        
        Returns:
            Number of embeddings
        """
        return self.index.ntotal