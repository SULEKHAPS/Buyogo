import os
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path

class DataLoader:
    """Class for loading hotel booking data from various sources."""
    
    def __init__(self, data_dir: str = None):
        """Initialize the DataLoader with the data directory path.
        
        Args:
            data_dir: Path to the directory containing data files. If None, uses default 'data' directory.
        """
        if data_dir is None:
            # Get the project root directory (assuming this file is in src/data/)
            project_root = Path(__file__).parent.parent.parent
            self.data_dir = project_root / 'data'
        else:
            self.data_dir = Path(data_dir)
            
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load data from a CSV file.
        
        Args:
            filename: Name of the CSV file in the data directory
            **kwargs: Additional arguments to pass to pandas.read_csv
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        return pd.read_csv(file_path, **kwargs)
    
    def load_hotel_bookings(self) -> pd.DataFrame:
        """Load the hotel bookings dataset.
        
        Returns:
            DataFrame containing the hotel bookings data
            
        Raises:
            FileNotFoundError: If the hotel_bookings.csv file doesn't exist
        """
        try:
            # Try to load the hotel bookings dataset
            return self.load_csv('hotel_bookings.csv')
        except FileNotFoundError:
            raise FileNotFoundError(
                "Hotel bookings dataset not found. Please download the dataset and "
                "place it in the 'data' directory as 'hotel_bookings.csv'."
            )
    
    def save_processed_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save processed data to a file.
        
        Args:
            data: DataFrame containing the processed data
            filename: Name of the file to save the data to
        """
        file_path = self.data_dir / filename
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine file format based on extension
        if filename.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif filename.endswith('.parquet'):
            data.to_parquet(file_path, index=False)
        elif filename.endswith('.pickle') or filename.endswith('.pkl'):
            data.to_pickle(file_path)
        else:
            # Default to CSV
            data.to_csv(file_path, index=False)
            
        print(f"Saved processed data to {file_path}")