"""
Data processing module for complaint data EDA and preprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from typing import Tuple


class ComplaintDataProcessor:
    """Handles EDA and preprocessing of complaint data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the processor.
        
        Args:
            data_path: Path to the raw complaint dataset
        """
        self.data_path = data_path
        self.df = None
        self.filtered_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the complaint dataset."""
        print(f"Loading data from {self.data_path}...")
        
        # Try different file formats
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path, low_memory=False)
        elif self.data_path.endswith('.parquet'):
            self.df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
            
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def perform_eda(self) -> dict:
        """
        Perform exploratory data analysis.
        
        Returns:
            Dictionary containing EDA statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print("\nDataset Shape:", self.df.shape)
        print("\nColumn Names:")
        print(self.df.columns.tolist())
        
        # Product distribution
        if 'Product' in self.df.columns:
            print("\n" + "-"*50)
            print("Product Distribution:")
            print("-"*50)
            product_dist = self.df['Product'].value_counts()
            print(product_dist)
            
            # Visualize product distribution
            plt.figure(figsize=(12, 6))
            product_dist.plot(kind='bar')
            plt.title('Distribution of Complaints by Product')
            plt.xlabel('Product')
            plt.ylabel('Number of Complaints')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('data/processed/product_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\nSaved product distribution plot to data/processed/product_distribution.png")
        
        # Narrative length analysis
        narrative_col = None
        for col in ['Consumer complaint narrative', 'consumer_complaint_narrative', 'narrative']:
            if col in self.df.columns:
                narrative_col = col
                break
        
        if narrative_col:
            print("\n" + "-"*50)
            print("Narrative Length Analysis:")
            print("-"*50)
            
            # Calculate word counts
            self.df['narrative_word_count'] = self.df[narrative_col].astype(str).apply(
                lambda x: len(x.split()) if pd.notna(x) and x != 'nan' else 0
            )
            
            # Statistics
            print(f"Mean word count: {self.df['narrative_word_count'].mean():.2f}")
            print(f"Median word count: {self.df['narrative_word_count'].median():.2f}")
            print(f"Min word count: {self.df['narrative_word_count'].min()}")
            print(f"Max word count: {self.df['narrative_word_count'].max()}")
            print(f"Std word count: {self.df['narrative_word_count'].std():.2f}")
            
            # Count empty narratives
            empty_narratives = self.df[narrative_col].isna() | (self.df[narrative_col].astype(str).str.strip() == '') | (self.df[narrative_col].astype(str) == 'nan')
            print(f"\nComplaints with narratives: {len(self.df) - empty_narratives.sum()}")
            print(f"Complaints without narratives: {empty_narratives.sum()}")
            
            # Visualize narrative length distribution
            plt.figure(figsize=(12, 6))
            plt.hist(self.df['narrative_word_count'], bins=50, edgecolor='black')
            plt.title('Distribution of Narrative Word Counts')
            plt.xlabel('Word Count')
            plt.ylabel('Frequency')
            plt.axvline(self.df['narrative_word_count'].mean(), color='r', linestyle='--', label='Mean')
            plt.axvline(self.df['narrative_word_count'].median(), color='g', linestyle='--', label='Median')
            plt.legend()
            plt.tight_layout()
            plt.savefig('data/processed/narrative_length_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\nSaved narrative length distribution plot to data/processed/narrative_length_distribution.png")
        
        # Summary statistics
        eda_stats = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'narrative_col': narrative_col,
            'empty_narratives': int(empty_narratives.sum()) if narrative_col else None,
            'mean_word_count': float(self.df['narrative_word_count'].mean()) if narrative_col else None,
            'median_word_count': float(self.df['narrative_word_count'].median()) if narrative_col else None,
        }
        
        return eda_stats
    
    def filter_data(self) -> pd.DataFrame:
        """
        Filter data to include only specified products and non-empty narratives.
        
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*50)
        print("FILTERING DATA")
        print("="*50)
        
        # Find narrative column
        narrative_col = None
        for col in ['Consumer complaint narrative', 'consumer_complaint_narrative', 'narrative']:
            if col in self.df.columns:
                narrative_col = col
                break
        
        if not narrative_col:
            raise ValueError("Could not find narrative column in dataset")
        
        # Find product column
        product_col = None
        for col in ['Product', 'product', 'product_category']:
            if col in self.df.columns:
                product_col = col
                break
        
        if not product_col:
            raise ValueError("Could not find product column in dataset")
        
        # Filter products - use partial matching to catch variations
        target_keywords = ['credit card', 'personal loan', 'savings account', 'money transfers']
        # Also check for variations
        all_products = self.df[product_col].unique()
        print(f"\nAvailable products: {all_products}")
        
        # Create case-insensitive filter using partial matching
        product_str_lower = self.df[product_col].astype(str).str.lower()
        product_filter = product_str_lower.str.contains('|'.join(target_keywords), case=False, na=False)
        
        print(f"\nRecords before product filter: {len(self.df)}")
        self.df = self.df[product_filter].copy()
        print(f"Records after product filter: {len(self.df)}")
        
        # Filter empty narratives
        print(f"\nRecords before narrative filter: {len(self.df)}")
        empty_narratives = (
            self.df[narrative_col].isna() | 
            (self.df[narrative_col].astype(str).str.strip() == '') | 
            (self.df[narrative_col].astype(str) == 'nan')
        )
        self.filtered_df = self.df[~empty_narratives].copy()
        print(f"Records after narrative filter: {len(self.filtered_df)}")
        
        return self.filtered_df
    
    def clean_text(self, text: str) -> str:
        """
        Clean text narrative.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == 'nan':
            return ''
        
        text = str(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove common boilerplate
        boilerplate_patterns = [
            r'i am writing to file a complaint',
            r'i am writing to complain',
            r'this is a complaint',
            r'complaint regarding',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Complete preprocessing pipeline: filter and clean.
        
        Returns:
            Preprocessed DataFrame
        """
        if self.filtered_df is None:
            self.filter_data()
        
        print("\n" + "="*50)
        print("CLEANING TEXT NARRATIVES")
        print("="*50)
        
        # Find narrative column
        narrative_col = None
        for col in ['Consumer complaint narrative', 'consumer_complaint_narrative', 'narrative']:
            if col in self.filtered_df.columns:
                narrative_col = col
                break
        
        if not narrative_col:
            raise ValueError("Could not find narrative column")
        
        # Clean narratives
        print(f"Cleaning {len(self.filtered_df)} narratives...")
        self.filtered_df['cleaned_narrative'] = self.filtered_df[narrative_col].apply(self.clean_text)
        
        # Remove any narratives that became empty after cleaning
        self.filtered_df = self.filtered_df[
            self.filtered_df['cleaned_narrative'].str.len() > 0
        ].copy()
        
        print(f"Final cleaned dataset: {len(self.filtered_df)} records")
        
        return self.filtered_df
    
    def save_cleaned_data(self, output_path: str = 'data/processed/filtered_complaints.csv'):
        """Save cleaned and filtered data."""
        if self.filtered_df is None:
            raise ValueError("No filtered data to save. Run preprocess_data() first.")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        self.filtered_df.to_csv(output_path, index=False)
        print(f"\nSaved cleaned data to {output_path}")
        print(f"Total records saved: {len(self.filtered_df)}")


def main():
    """Main function to run EDA and preprocessing."""
    # Initialize processor
    # Note: User needs to download the CFPB dataset and place it in data/raw/
    data_path = 'data/raw/complaints.csv'  # Adjust path as needed
    
    processor = ComplaintDataProcessor(data_path)
    
    try:
        # Load data
        processor.load_data()
        
        # Perform EDA
        eda_stats = processor.perform_eda()
        
        # Filter and preprocess
        processor.filter_data()
        processor.preprocess_data()
        
        # Save cleaned data
        processor.save_cleaned_data()
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE")
        print("="*50)
        print(f"Final dataset size: {len(processor.filtered_df)} records")
        
    except FileNotFoundError:
        print(f"\nError: Could not find data file at {data_path}")
        print("Please download the CFPB complaint dataset and place it in data/raw/")
        print("You can download it from: https://www.consumerfinance.gov/data-research/consumer-complaints/")


if __name__ == '__main__':
    main()

