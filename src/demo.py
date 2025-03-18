import os
import pandas as pd
import json
from pathlib import Path

# Import project components
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from analytics.reports import AnalyticsReports
from rag.embeddings import EmbeddingGenerator
from rag.vector_db import VectorDatabase
from rag.qa_system import QASystem

def main():
    """Demonstrate the RAG system for hotel booking analytics."""
    print("\n===== Hotel Booking Analytics & QA System Demo =====\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    data_loader = DataLoader()
    try:
        data = data_loader.load_hotel_bookings()
        print(f"Loaded {len(data)} hotel booking records")
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(data)
        print(f"Preprocessed data: {len(processed_data)} records retained")
        print(f"Columns: {', '.join(processed_data.columns)}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the hotel_bookings.csv dataset and place it in the 'data' directory.")
        return
    
    # Step 2: Generate analytics reports
    print("Step 2: Generating analytics reports...")
    analytics = AnalyticsReports(processed_data)
    reports = analytics.generate_all_reports()
    
    # Print a summary of each report
    for report_name, report_data in reports.items():
        print(f"\n- {report_name.replace('_', ' ').title()}:")
        if report_name == 'revenue_trends':
            print(f"  Total Revenue: ${report_data['total_revenue']}")
            print(f"  Average Revenue per Booking: ${report_data['average_revenue_per_booking']}")
            if 'year_over_year_growth' in report_data:
                for period, growth in report_data['year_over_year_growth'].items():
                    print(f"  Growth {period}: {growth}%")
        elif report_name == 'cancellation_rate':
            print(f"  Overall Cancellation Rate: {report_data['overall_cancellation_rate']}%")
            print(f"  Total Bookings: {report_data['total_bookings']}")
            print(f"  Canceled Bookings: {report_data['canceled_bookings']}")
        elif report_name == 'geographical_distribution':
            print(f"  Total Countries: {report_data['total_countries']}")
            print(f"  Top 3 Countries by Bookings:")
            for i, country in enumerate(report_data['top_countries_by_bookings'][:3]):
                print(f"    {i+1}. {country['country']}: {country['booking_count']} bookings ({country['percentage']:.1f}%)")
        elif report_name == 'lead_time_distribution':
            print(f"  Average Lead Time: {report_data['lead_time_stats']['mean']} days")
            print(f"  Median Lead Time: {report_data['lead_time_stats']['median']} days")
            print(f"  Lead Time Range: {report_data['lead_time_stats']['min']} to {report_data['lead_time_stats']['max']} days")
    
    # Step 3: Set up the RAG system
    print("\nStep 3: Setting up the RAG system...")
    
    # Define paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    vector_db_path = data_dir / 'vector_db'
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()
    print("Initialized embedding generator")
    
    # Initialize or load vector database
    if os.path.exists(str(vector_db_path) + '.index'):
        print("Loading existing vector database...")
        vector_db = VectorDatabase.load(vector_db_path)
        print(f"Loaded vector database with {len(vector_db)} embeddings")
    else:
        print("Creating new vector database...")
        vector_db = VectorDatabase()
        
        # Generate embeddings
        print("Generating embeddings for hotel bookings...")
        embeddings = embedding_generator.generate_booking_embeddings(processed_data)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Create data dictionary
        data_dict = {}
        for idx, row in processed_data.iterrows():
            data_dict[str(idx)] = row.to_dict()
        
        # Add embeddings to vector database
        vector_db.add_embeddings(embeddings, data_dict)
        print(f"Added {len(embeddings)} embeddings to vector database")
        
        # Save vector database
        os.makedirs(data_dir, exist_ok=True)
        vector_db.save(vector_db_path)
        print(f"Saved vector database to {vector_db_path}")
    
    # Initialize QA system
    qa_system = QASystem(vector_db, embedding_generator)
    qa_system.set_analytics(analytics)
    print("Initialized QA system")
    
    # Step 4: Demonstrate QA capabilities
    print("\nStep 4: Demonstrating QA capabilities...")
    
    # Example queries
    example_queries = [
        "What is the average revenue per booking?",
        "Which country has the highest number of bookings?",
        "What is the cancellation rate for resort hotels?",
        "What is the average lead time for bookings?",
        "How does the revenue trend look over time?"
    ]
    
    for i, query in enumerate(example_queries):
        print(f"\nQuery {i+1}: {query}")
        try:
            result = qa_system.answer_query(query)
            print(f"Answer: {result['answer']}")
            if 'analytics_data' in result and result['analytics_data']:
                print(f"Report Type: {result['report_type']}")
            elif 'sources' in result and result['sources']:
                print(f"Sources: {', '.join(result['sources'][:3])}" + 
                      (" and more..." if len(result['sources']) > 3 else ""))
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n===== Demo Complete =====\n")
    print("To use the API, run the following command:")
    print("uvicorn src.api.app:app --reload")
    print("Then visit http://localhost:8000/docs in your browser to access the API documentation.")

if __name__ == "__main__":
    main()