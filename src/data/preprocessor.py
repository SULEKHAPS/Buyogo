import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

class DataPreprocessor:
    """Class for preprocessing hotel booking data."""
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        pass
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the hotel booking data.
        
        Args:
            data: DataFrame containing the hotel bookings data
            
        Returns:
            DataFrame containing the preprocessed data
        """
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Apply preprocessing steps
        df = self._handle_missing_values(df)
        df = self._fix_data_types(df)
        df = self._create_derived_features(df)
        
        return df
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data.
        
        Args:
            data: DataFrame containing the hotel bookings data
            
        Returns:
            DataFrame with missing values handled
        """
        df = data.copy()
        
        # Fill missing values for children with 0 (assuming no children if not specified)
        if 'children' in df.columns:
            df['children'] = df['children'].fillna(0)
        
        # Fill missing values for country with 'unknown'
        if 'country' in df.columns:
            df['country'] = df['country'].fillna('unknown')
        
        # Fill missing values for agent with 0 (no agent)
        if 'agent' in df.columns:
            df['agent'] = df['agent'].fillna(0)
        
        # Fill missing values for company with 0 (no company)
        if 'company' in df.columns:
            df['company'] = df['company'].fillna(0)
        
        # Drop rows with missing values in critical columns
        critical_columns = ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 
                           'adults', 'hotel', 'is_canceled']
        critical_columns = [col for col in critical_columns if col in df.columns]
        df = df.dropna(subset=critical_columns)
        
        return df
    
    def _fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix data types in the DataFrame.
        
        Args:
            data: DataFrame containing the hotel bookings data
            
        Returns:
            DataFrame with corrected data types
        """
        df = data.copy()
        
        # Convert numeric columns to appropriate types
        numeric_columns = ['adults', 'children', 'babies', 'is_canceled', 'lead_time', 
                          'stays_in_weekend_nights', 'stays_in_week_nights', 'agent', 'company']
        numeric_columns = [col for col in numeric_columns if col in df.columns]
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns if they exist separately
        if all(col in df.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
            # Create a datetime column from separate date components
            month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
            
            # Convert month names to numbers if needed
            if df['arrival_date_month'].dtype == 'object':
                df['arrival_date_month_num'] = df['arrival_date_month'].map(month_map)
            else:
                df['arrival_date_month_num'] = df['arrival_date_month']
            
            # Create a datetime column
            df['arrival_date'] = pd.to_datetime({
                'year': df['arrival_date_year'],
                'month': df['arrival_date_month_num'],
                'day': df['arrival_date_day_of_month']
            })
        
        return df
    
    def _create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing data.
        
        Args:
            data: DataFrame containing the hotel bookings data
            
        Returns:
            DataFrame with additional derived features
        """
        df = data.copy()
        
        # Calculate total nights stayed
        if all(col in df.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        
        # Calculate total guests
        guest_columns = [col for col in ['adults', 'children', 'babies'] if col in df.columns]
        if guest_columns:
            df['total_guests'] = df[guest_columns].sum(axis=1)
        
        # Calculate cancellation rate (1 for canceled, 0 for not canceled)
        if 'is_canceled' in df.columns:
            df['is_canceled'] = df['is_canceled'].astype(int)
        
        # Calculate total price if adr (average daily rate) and total_nights exist
        if all(col in df.columns for col in ['adr', 'total_nights']):
            df['total_price'] = df['adr'] * df['total_nights']
        
        return df
    
    def split_data(self, data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into training and testing sets.
        
        Args:
            data: DataFrame containing the preprocessed data
            test_size: Proportion of the data to include in the test split
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        # Shuffle the data
        shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Calculate the split index
        split_idx = int(len(shuffled_data) * (1 - test_size))
        
        # Split the data
        train_data = shuffled_data.iloc[:split_idx].reset_index(drop=True)
        test_data = shuffled_data.iloc[split_idx:].reset_index(drop=True)
        
        return train_data, test_data