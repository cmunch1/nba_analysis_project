"""
Tests for the data_validator module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from nba_app.data_validator import DataValidator
from ml_framework.core.error_handling.error_handler import DataValidationError


@pytest.fixture
def mock_config():
    """Create a mock config for data validator."""
    config = Mock()
    config.game_id_column = "GAME_ID"
    config.processed_schema = ["GAME_ID", "TEAM_ID", "PTS", "AST", "REB"]
    return config


@pytest.fixture
def mock_data_access():
    """Create a mock data access object."""
    return Mock()


@pytest.fixture
def mock_app_file_handler():
    """Create a mock app file handler."""
    return Mock()


@pytest.fixture
def validator(mock_config, mock_data_access, mock_app_logger, mock_app_file_handler, mock_error_handler):
    """Create a DataValidator instance."""
    return DataValidator(mock_config, mock_data_access, mock_app_logger, mock_app_file_handler, mock_error_handler)


def test_initialization(mock_config, mock_data_access, mock_app_logger, mock_app_file_handler, mock_error_handler):
    """Test that DataValidator initializes correctly."""
    validator = DataValidator(mock_config, mock_data_access, mock_app_logger, mock_app_file_handler, mock_error_handler)

    assert validator.config == mock_config
    assert validator.data_access == mock_data_access
    assert validator.app_logger == mock_app_logger
    assert validator.app_file_handler == mock_app_file_handler
    assert validator.error_handler == mock_error_handler


def test_validate_dataframe_success(validator):
    """Test successful dataframe validation."""
    df = pd.DataFrame({
        'GAME_ID': [1, 2, 3],
        'TEAM_ID': [101, 102, 103],
        'PTS': [100, 110, 95]
    })

    result = validator._validate_dataframe(df, "test_file.csv")

    assert result is True


def test_validate_dataframe_with_duplicates(validator):
    """Test that validation fails with duplicate records."""
    df = pd.DataFrame({
        'GAME_ID': [1, 1, 2],
        'TEAM_ID': [101, 101, 102],
        'PTS': [100, 100, 110]
    })

    result = validator._validate_dataframe(df, "test_file.csv")

    assert result is False


def test_validate_dataframe_with_nulls(validator):
    """Test that validation fails with null values."""
    df = pd.DataFrame({
        'GAME_ID': [1, 2, None],
        'TEAM_ID': [101, 102, 103],
        'PTS': [100, 110, 95]
    })

    result = validator._validate_dataframe(df, "test_file.csv")

    assert result is False


def test_validate_schema_success(validator):
    """Test successful schema validation."""
    df = pd.DataFrame({
        'GAME_ID': [1, 2, 3],
        'TEAM_ID': [101, 102, 103],
        'PTS': [100, 110, 95],
        'AST': [20, 25, 22],
        'REB': [40, 45, 38]
    })

    result = validator._validate_schema(df, "test_file.csv", validator.config.processed_schema)

    assert result is True


def test_validate_schema_incorrect_columns(validator):
    """Test that schema validation fails with incorrect columns."""
    df = pd.DataFrame({
        'GAME_ID': [1, 2, 3],
        'TEAM_ID': [101, 102, 103],
        'WRONG_COLUMN': [100, 110, 95]
    })

    result = validator._validate_schema(df, "test_file.csv", validator.config.processed_schema)

    assert result is False


def test_validate_scraped_dataframes_success(validator):
    """Test successful validation of scraped dataframes."""
    df1 = pd.DataFrame({
        'GAME_ID': [1, 2, 3],
        'TEAM_ID': [101, 102, 103],
        'PTS': [100, 110, 95]
    })
    df2 = pd.DataFrame({
        'GAME_ID': [1, 2, 3],
        'TEAM_ID': [101, 102, 103],
        'AST': [20, 25, 22]
    })

    result = validator.validate_scraped_dataframes([df1, df2], ["file1.csv", "file2.csv"])

    assert result is True


def test_validate_scraped_dataframes_different_game_ids(validator):
    """Test that validation fails when game IDs don't match across dataframes."""
    df1 = pd.DataFrame({
        'GAME_ID': [1, 2, 3],
        'PTS': [100, 110, 95]
    })
    df2 = pd.DataFrame({
        'GAME_ID': [1, 2, 4],  # Different game ID
        'AST': [20, 25, 22]
    })

    with pytest.raises(DataValidationError):
        validator.validate_scraped_dataframes([df1, df2], ["file1.csv", "file2.csv"])


def test_validate_scraped_dataframes_different_row_counts(validator):
    """Test that validation fails when row counts don't match."""
    df1 = pd.DataFrame({
        'GAME_ID': [1, 2, 3],
        'PTS': [100, 110, 95]
    })
    df2 = pd.DataFrame({
        'GAME_ID': [1, 2],  # Missing row
        'AST': [20, 25]
    })

    with pytest.raises(DataValidationError):
        validator.validate_scraped_dataframes([df1, df2], ["file1.csv", "file2.csv"])


def test_validate_scraped_dataframes_empty_first_dataframe(validator):
    """Test that validation skips when first dataframe is empty."""
    df1 = pd.DataFrame()
    df2 = pd.DataFrame({'GAME_ID': [1, 2], 'AST': [20, 25]})

    result = validator.validate_scraped_dataframes([df1, df2], ["file1.csv", "file2.csv"])

    assert result is True


def test_validate_processed_dataframe_success(validator):
    """Test successful validation of processed dataframe."""
    df = pd.DataFrame({
        'GAME_ID': [1, 2, 3],
        'TEAM_ID': [101, 102, 103],
        'PTS': [100, 110, 95],
        'AST': [20, 25, 22],
        'REB': [40, 45, 38]
    })

    result = validator.validate_processed_dataframe(df, "processed_file.csv")

    assert result is True


def test_validate_processed_dataframe_schema_mismatch(validator):
    """Test that processed dataframe validation fails with schema mismatch."""
    df = pd.DataFrame({
        'GAME_ID': [1, 2, 3],
        'WRONG_COLUMN': [101, 102, 103]
    })

    with pytest.raises(DataValidationError):
        validator.validate_processed_dataframe(df, "processed_file.csv")


def test_validate_processed_dataframe_with_duplicates(validator):
    """Test that processed dataframe validation fails with duplicates."""
    df = pd.DataFrame({
        'GAME_ID': [1, 1, 2],
        'TEAM_ID': [101, 101, 102],
        'PTS': [100, 100, 110],
        'AST': [20, 20, 25],
        'REB': [40, 40, 45]
    })

    with pytest.raises(DataValidationError):
        validator.validate_processed_dataframe(df, "processed_file.csv")


def test_log_performance_decorator(validator):
    """Test that the log_performance decorator works correctly."""
    # The decorator should not throw errors
    @DataValidator.log_performance
    def test_function(self):
        return "test"

    result = test_function(validator)
    assert result == "test"
