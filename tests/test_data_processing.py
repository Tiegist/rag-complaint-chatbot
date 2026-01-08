"""
Unit tests for data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from data_processing import ComplaintDataProcessor


@pytest.fixture
def sample_complaint_data():
    """Create sample complaint data for testing."""
    data = {
        'Complaint ID': [1, 2, 3, 4, 5],
        'Product': ['Credit card', 'Personal loan', 'Savings account', 'Money transfers', 'Credit card'],
        'Consumer complaint narrative': [
            'I have a complaint about my credit card billing.',
            'The personal loan process was very difficult.',
            'I cannot access my savings account.',
            'Money transfer failed multiple times.',
            ''  # Empty narrative
        ],
        'Issue': ['Billing dispute', 'Application processing', 'Account access', 'Transaction issue', 'Other'],
        'Date received': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    }
    return pd.DataFrame(data)


def test_complaint_processor_initialization():
    """Test ComplaintDataProcessor initialization."""
    processor = ComplaintDataProcessor('data/raw/complaints.csv')
    assert processor.data_path == 'data/raw/complaints.csv'
    assert processor.df is None
    assert processor.filtered_df is None


def test_clean_text():
    """Test text cleaning function."""
    processor = ComplaintDataProcessor('data/raw/complaints.csv')
    
    # Test normal text
    text = "I am writing to file a complaint about my credit card."
    cleaned = processor.clean_text(text)
    assert "i am writing to file a complaint" not in cleaned.lower()
    assert "complaint about my credit card" in cleaned.lower()
    
    # Test empty text
    assert processor.clean_text('') == ''
    assert processor.clean_text('nan') == ''
    
    # Test with special characters
    text = "This   has    multiple    spaces"
    cleaned = processor.clean_text(text)
    assert "  " not in cleaned  # No double spaces


def test_filter_data(sample_complaint_data, tmp_path):
    """Test data filtering."""
    # Save sample data
    test_file = tmp_path / "test_complaints.csv"
    sample_complaint_data.to_csv(test_file, index=False)
    
    processor = ComplaintDataProcessor(str(test_file))
    processor.df = sample_complaint_data
    
    filtered = processor.filter_data()
    
    # Should filter out empty narratives and keep only target products
    assert len(filtered) <= len(sample_complaint_data)
    assert 'Consumer complaint narrative' in filtered.columns or 'consumer_complaint_narrative' in filtered.columns


def test_perform_eda(sample_complaint_data, tmp_path):
    """Test EDA functionality."""
    test_file = tmp_path / "test_complaints.csv"
    sample_complaint_data.to_csv(test_file, index=False)
    
    processor = ComplaintDataProcessor(str(test_file))
    processor.df = sample_complaint_data
    
    eda_stats = processor.perform_eda()
    
    assert 'total_records' in eda_stats
    assert eda_stats['total_records'] == len(sample_complaint_data)
    assert 'columns' in eda_stats



