import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json

class AnalyticsReports:
    """Class for generating analytics reports from hotel booking data."""
    
    def __init__(self, data: pd.DataFrame = None):
        """Initialize the AnalyticsReports with data.
        
        Args:
            data: DataFrame containing the preprocessed hotel bookings data
        """
        self.data = data
    
    def set_data(self, data: pd.DataFrame) -> None:
        """Set the data for analysis.
        
        Args:
            data: DataFrame containing the preprocessed hotel bookings data
        """
        self.data = data
    
    def generate_all_reports(self) -> Dict[str, Any]:
        """Generate all analytics reports.
        
        Returns:
            Dictionary containing all analytics reports
        """
        if self.data is None:
            raise ValueError("No data available for analysis. Please set data using set_data() method.")
        
        reports = {
            "revenue_trends": self.analyze_revenue_trends(),
            "cancellation_rate": self.analyze_cancellation_rate(),
            "geographical_distribution": self.analyze_geographical_distribution(),
            "lead_time_distribution": self.analyze_lead_time_distribution()
        }
        
        return reports
    
    def analyze_revenue_trends(self) -> Dict[str, Any]:
        """Analyze revenue trends over time.
        
        Returns:
            Dictionary containing revenue trends analysis
        """
        if self.data is None:
            raise ValueError("No data available for analysis. Please set data using set_data() method.")
        
        # Make a copy of the data to avoid modifying the original
        df = self.data.copy()
        
        # Ensure we have the necessary columns
        required_columns = ['arrival_date', 'total_price']
        if not all(col in df.columns for col in required_columns):
            # Try to create arrival_date if it doesn't exist but components do
            if 'arrival_date' not in df.columns and all(col in df.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
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
            
            # Try to create total_price if it doesn't exist but components do
            if 'total_price' not in df.columns and all(col in df.columns for col in ['adr', 'total_nights']):
                df['total_price'] = df['adr'] * df['total_nights']
            elif 'total_price' not in df.columns and all(col in df.columns for col in ['adr', 'stays_in_weekend_nights', 'stays_in_week_nights']):
                df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
                df['total_price'] = df['adr'] * df['total_nights']
        
        # Check again if we have the necessary columns after attempting to create them
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns for revenue analysis: {missing_cols}")
        
        # Filter out canceled bookings if that information is available
        if 'is_canceled' in df.columns:
            df = df[df['is_canceled'] == 0]
        
        # Group by month and calculate total revenue
        df['year_month'] = df['arrival_date'].dt.to_period('M')
        monthly_revenue = df.groupby('year_month')['total_price'].sum().reset_index()
        monthly_revenue['year_month_str'] = monthly_revenue['year_month'].astype(str)
        
        # Calculate year-over-year growth if we have multiple years
        years = df['arrival_date'].dt.year.unique()
        yoy_growth = None
        if len(years) > 1:
            # Group by year and calculate total revenue per year
            yearly_revenue = df.groupby(df['arrival_date'].dt.year)['total_price'].sum()
            # Calculate year-over-year growth
            yoy_growth = {}
            for i in range(1, len(years)):
                prev_year = years[i-1]
                curr_year = years[i]
                if prev_year in yearly_revenue.index and curr_year in yearly_revenue.index:
                    growth = (yearly_revenue[curr_year] - yearly_revenue[prev_year]) / yearly_revenue[prev_year] * 100
                    yoy_growth[f"{prev_year}-{curr_year}"] = round(growth, 2)
        
        # Prepare the result
        result = {
            "monthly_revenue": {
                "months": monthly_revenue['year_month_str'].tolist(),
                "revenue": monthly_revenue['total_price'].tolist()
            },
            "total_revenue": round(df['total_price'].sum(), 2),
            "average_revenue_per_booking": round(df['total_price'].mean(), 2)
        }
        
        if yoy_growth:
            result["year_over_year_growth"] = yoy_growth
        
        return result
    
    def analyze_cancellation_rate(self) -> Dict[str, Any]:
        """Analyze cancellation rate as percentage of total bookings.
        
        Returns:
            Dictionary containing cancellation rate analysis
        """
        if self.data is None:
            raise ValueError("No data available for analysis. Please set data using set_data() method.")
        
        # Make a copy of the data to avoid modifying the original
        df = self.data.copy()
        
        # Ensure we have the necessary columns
        if 'is_canceled' not in df.columns:
            raise ValueError("Missing required column 'is_canceled' for cancellation analysis")
        
        # Calculate overall cancellation rate
        total_bookings = len(df)
        canceled_bookings = df['is_canceled'].sum()
        cancellation_rate = (canceled_bookings / total_bookings) * 100
        
        # Calculate cancellation rate by hotel type if available
        hotel_cancellation = None
        if 'hotel' in df.columns:
            hotel_cancellation = df.groupby('hotel').agg(
                total_bookings=('is_canceled', 'count'),
                canceled_bookings=('is_canceled', 'sum')
            ).reset_index()
            hotel_cancellation['cancellation_rate'] = (hotel_cancellation['canceled_bookings'] / 
                                                     hotel_cancellation['total_bookings']) * 100
            hotel_cancellation = hotel_cancellation.to_dict('records')
        
        # Calculate cancellation rate by month if arrival_date is available
        monthly_cancellation = None
        if 'arrival_date' in df.columns:
            df['year_month'] = df['arrival_date'].dt.to_period('M')
            monthly_cancellation = df.groupby('year_month').agg(
                total_bookings=('is_canceled', 'count'),
                canceled_bookings=('is_canceled', 'sum')
            ).reset_index()
            monthly_cancellation['cancellation_rate'] = (monthly_cancellation['canceled_bookings'] / 
                                                       monthly_cancellation['total_bookings']) * 100
            monthly_cancellation['year_month_str'] = monthly_cancellation['year_month'].astype(str)
            monthly_cancellation = {
                "months": monthly_cancellation['year_month_str'].tolist(),
                "rates": monthly_cancellation['cancellation_rate'].tolist()
            }
        elif all(col in df.columns for col in ['arrival_date_year', 'arrival_date_month']):
            # Create a year-month column from separate components
            df['year_month'] = df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'].astype(str)
            monthly_cancellation = df.groupby('year_month').agg(
                total_bookings=('is_canceled', 'count'),
                canceled_bookings=('is_canceled', 'sum')
            ).reset_index()
            monthly_cancellation['cancellation_rate'] = (monthly_cancellation['canceled_bookings'] / 
                                                       monthly_cancellation['total_bookings']) * 100
            monthly_cancellation = {
                "months": monthly_cancellation['year_month'].tolist(),
                "rates": monthly_cancellation['cancellation_rate'].tolist()
            }
        
        # Prepare the result
        result = {
            "overall_cancellation_rate": round(cancellation_rate, 2),
            "total_bookings": int(total_bookings),
            "canceled_bookings": int(canceled_bookings)
        }
        
        if hotel_cancellation:
            result["cancellation_by_hotel_type"] = hotel_cancellation
        
        if monthly_cancellation:
            result["monthly_cancellation_rate"] = monthly_cancellation
        
        return result
    
    def analyze_geographical_distribution(self) -> Dict[str, Any]:
        """Analyze geographical distribution of users doing the bookings.
        
        Returns:
            Dictionary containing geographical distribution analysis
        """
        if self.data is None:
            raise ValueError("No data available for analysis. Please set data using set_data() method.")
        
        # Make a copy of the data to avoid modifying the original
        df = self.data.copy()
        
        # Ensure we have the necessary columns
        if 'country' not in df.columns:
            raise ValueError("Missing required column 'country' for geographical analysis")
        
        # Count bookings by country
        country_counts = df['country'].value_counts().reset_index()
        country_counts.columns = ['country', 'booking_count']
        
        # Calculate percentage of total bookings
        total_bookings = len(df)
        country_counts['percentage'] = (country_counts['booking_count'] / total_bookings) * 100
        
        # Get top 10 countries by booking count
        top_countries = country_counts.head(10).to_dict('records')
        
        # Calculate cancellation rate by country if available
        country_cancellation = None
        if 'is_canceled' in df.columns:
            country_cancellation = df.groupby('country').agg(
                total_bookings=('is_canceled', 'count'),
                canceled_bookings=('is_canceled', 'sum')
            ).reset_index()
            country_cancellation['cancellation_rate'] = (country_cancellation['canceled_bookings'] / 
                                                       country_cancellation['total_bookings']) * 100
            # Get top 10 countries by cancellation rate (with at least 10 bookings)
            min_bookings = 10
            top_cancellation_countries = country_cancellation[country_cancellation['total_bookings'] >= min_bookings]
            top_cancellation_countries = top_cancellation_countries.sort_values('cancellation_rate', ascending=False)
            top_cancellation_countries = top_cancellation_countries.head(10).to_dict('records')
        
        # Prepare the result
        result = {
            "total_countries": len(country_counts),
            "top_countries_by_bookings": top_countries
        }
        
        if country_cancellation is not None:
            result["top_countries_by_cancellation_rate"] = top_cancellation_countries
        
        return result
    
    def analyze_lead_time_distribution(self) -> Dict[str, Any]:
        """Analyze booking lead time distribution.
        
        Returns:
            Dictionary containing lead time distribution analysis
        """
        if self.data is None:
            raise ValueError("No data available for analysis. Please set data using set_data() method.")
        
        # Make a copy of the data to avoid modifying the original
        df = self.data.copy()
        
        # Ensure we have the necessary columns
        if 'lead_time' not in df.columns:
            raise ValueError("Missing required column 'lead_time' for lead time analysis")
        
        # Calculate lead time statistics
        lead_time_stats = {
            "mean": round(df['lead_time'].mean(), 2),
            "median": round(df['lead_time'].median(), 2),
            "min": int(df['lead_time'].min()),
            "max": int(df['lead_time'].max()),
            "std": round(df['lead_time'].std(), 2)
        }
        
        # Create lead time bins
        bins = [0, 7, 30, 90, 180, 365, float('inf')]
        labels = ['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days']
        df['lead_time_bin'] = pd.cut(df['lead_time'], bins=bins, labels=labels)
        
        # Count bookings by lead time bin
        lead_time_distribution = df['lead_time_bin'].value_counts().sort_index().reset_index()
        lead_time_distribution.columns = ['lead_time_range', 'booking_count']
        
        # Calculate percentage of total bookings
        total_bookings = len(df)
        lead_time_distribution['percentage'] = (lead_time_distribution['booking_count'] / total_bookings) * 100
        
        # Calculate lead time by hotel type if available
        hotel_lead_time = None
        if 'hotel' in df.columns:
            hotel_lead_time = df.groupby('hotel')['lead_time'].agg(['mean', 'median']).reset_index()
            hotel_lead_time = hotel_lead_time.rename(columns={'mean': 'average_lead_time', 'median': 'median_lead_time'})
            hotel_lead_time['average_lead_time'] = hotel_lead_time['average_lead_time'].round(1)
            hotel_lead_time['median_lead_time'] = hotel_lead_time['median_lead_time'].round(1)
            hotel_lead_time = hotel_lead_time.to_dict('records')
        
        # Prepare the result
        result = {
            "lead_time_stats": lead_time_stats,
            "lead_time_distribution": lead_time_distribution.to_dict('records')
        }
        
        if hotel_lead_time:
            result["lead_time_by_hotel_type"] = hotel_lead_time
        
        return result