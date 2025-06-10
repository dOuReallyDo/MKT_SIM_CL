"""
Tests for the data collection module.
"""

import os
import sys
import pytest
import pandas as pd
from datetime import datetime

# Add project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data.collector import DataCollector

# Directory containing test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
# Ensure the test data directory exists within the tests directory
if not os.path.exists(TEST_DATA_DIR):
    os.makedirs(TEST_DATA_DIR)
    # Optional: Create a dummy CSV if needed for initial tests
    dummy_data = {'Date': pd.to_datetime(['2023-01-01', '2023-01-02']), 'Close': [100, 101]}
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv(os.path.join(TEST_DATA_DIR, 'DUMMY.csv'), index=False)


@pytest.fixture(scope="module")
def data_collector():
    """Fixture to create a DataCollector instance."""
    # Point the collector to use the test data directory
    # We might need to adjust DataCollector or use mocking later
    # For now, assume it can work with local files if they exist
    return DataCollector(data_dir=TEST_DATA_DIR) # Assuming DataCollector can take a data_dir argument

# --- Test Initialization ---

def test_data_collector_initialization(data_collector):
    """Test if DataCollector initializes correctly."""
    assert data_collector is not None
    assert data_collector.data_dir == TEST_DATA_DIR

# --- Test Loading Local Data ---

def test_load_local_data(data_collector):
    """Test loading data from an existing local CSV file."""
    symbol = 'AAPL' # Assuming AAPL.csv exists in tests/test_data
    start_date = '2023-01-01' # Use dates covered by sample data if known, otherwise arbitrary
    end_date = '2023-12-31'   # Use dates covered by sample data if known, otherwise arbitrary
    
    # Ensure the test file exists
    test_file_path = os.path.join(TEST_DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(test_file_path):
         pytest.skip(f"Test data file not found: {test_file_path}")

    # Use get_stock_data with force_download=False to load from cache/local
    df = data_collector.get_stock_data(symbol, start_date, end_date, force_download=False)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    # Check for standard columns (case-insensitive check already done by collector)
    assert 'Open' in df.columns
    assert 'High' in df.columns
    assert 'Low' in df.columns
    assert 'Close' in df.columns
    assert 'Volume' in df.columns

# --- Test Data Availability ---

def test_data_availability_exists(data_collector):
    """Test is_data_available for an existing local file."""
    symbol = 'AAPL' # Assuming AAPL.csv exists
    start_date = '2023-01-01' 
    end_date = '2023-12-31'   
    
    test_file_path = os.path.join(TEST_DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(test_file_path):
         pytest.skip(f"Test data file not found: {test_file_path}")
         
    assert data_collector.is_data_available(symbol, start_date, end_date) is True

def test_data_availability_not_exists(data_collector):
    """Test is_data_available for a non-existent local file."""
    symbol = 'NONEXISTENT'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    assert data_collector.is_data_available(symbol, start_date, end_date) is False

# --- Placeholder for Future Tests ---
# TODO: Add tests for data cleaning (requires specific input data)
# TODO: Add tests for downloading (might require mocking yfinance)
# TODO: Add tests for verify_data_integrity
