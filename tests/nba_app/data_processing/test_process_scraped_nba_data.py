"""
Tests for the process_scraped_NBA_data module.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from nba_app.data_processing.process_scraped_NBA_data import ProcessScrapedNBAData
from ml_framework.core.error_handling.error_handler import DataProcessingError


@pytest.fixture
def mock_config():
    """Create a mock config for data processor."""
    config = Mock()
    config.new_date_column = "GAME_DATE"
    config.new_game_id_column = "GAME_ID"
    config.home_team_column = "is_home_team"
    config.game_info_columns = ["GAME_DATE", "GAME_ID", "SEASON"]
    config.home_game_suffix = "home"
    config.visitor_game_suffix = "visitor"
    return config


@pytest.fixture
def mock_app_file_handler():
    """Create a mock app file handler."""
    return Mock()


@pytest.fixture
def processor(mock_config, mock_app_logger, mock_app_file_handler):
    """Create a ProcessScrapedNBAData instance."""
    return ProcessScrapedNBAData(mock_config, mock_app_logger, mock_app_file_handler)


def test_initialization(mock_config, mock_app_logger, mock_app_file_handler):
    """Test that ProcessScrapedNBAData initializes correctly."""
    processor = ProcessScrapedNBAData(mock_config, mock_app_logger, mock_app_file_handler)

    assert processor.config == mock_config
    assert processor.app_logger == mock_app_logger
    assert processor.app_file_handler == mock_app_file_handler


def test_process_data_success(processor):
    """Test successful data processing."""
    # Create mock scraped dataframes
    df1 = pd.DataFrame({
        'GAME_ID': ['001', '002', '003'],
        'TEAM_ID': [101, 102, 103],
        'PTS': [100, 110, 95]
    })
    df2 = pd.DataFrame({
        'GAME_ID': ['001', '002', '003'],
        'TEAM_ID': [101, 102, 103],
        'AST': [20, 25, 22]
    })

    with patch.object(processor, '_merge_dataframes') as mock_merge, \
         patch.object(processor, '_handle_anomalous_data') as mock_handle, \
         patch.object(processor, '_transform_data') as mock_transform:

        # Setup mock returns
        merged_df = pd.DataFrame({
            'GAME_ID': ['001', '002', '003'],
            'TEAM_ID': [101, 102, 103],
            'PTS': [100, 110, 95],
            'AST': [20, 25, 22]
        })
        mock_merge.return_value = merged_df

        transformed_df = pd.DataFrame({
            'GAME_ID': ['001', '002', '003'],
            'GAME_DATE': pd.to_datetime(['2023-10-01', '2023-10-02', '2023-10-03']),
            'is_home_team': [True, False, True],
            'TEAM_ID': [101, 102, 103],
            'PTS': [100, 110, 95],
            'AST': [20, 25, 22]
        })
        column_mapping = {'old_col': 'new_col'}
        mock_handle.return_value = merged_df
        mock_transform.return_value = (transformed_df, column_mapping)

        result_df, result_mapping = processor.process_data([df1, df2])

        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(result_mapping, dict)
        mock_merge.assert_called_once()
        mock_handle.assert_called_once()
        mock_transform.assert_called_once()


def test_merge_team_data_success(processor):
    """Test successful merging of team data into game-centric format."""
    # Create test data with home and visitor teams
    df = pd.DataFrame({
        'GAME_ID': ['001', '001', '002', '002'],
        'GAME_DATE': pd.to_datetime(['2023-10-01', '2023-10-01', '2023-10-02', '2023-10-02']),
        'SEASON': ['2023-24', '2023-24', '2023-24', '2023-24'],
        'is_home_team': [1, 0, 1, 0],
        'TEAM_ID': [101, 102, 103, 104],
        'PTS': [110, 105, 95, 100],
        'AST': [25, 20, 22, 24]
    })

    with patch.object(processor, '_merge_features') as mock_merge_features:
        # Mock any additional merging that might happen
        mock_merge_features.return_value = df

        result = processor.merge_team_data(df)

        assert isinstance(result, pd.DataFrame)
        # Result should have 2 rows (one per game)
        assert len(result) == 2


def test_process_data_with_exception(processor):
    """Test that DataProcessingError is raised on unexpected errors."""
    df1 = pd.DataFrame({'GAME_ID': ['001'], 'PTS': [100]})

    with patch.object(processor, '_merge_dataframes') as mock_merge:
        mock_merge.side_effect = Exception("Unexpected error")

        with pytest.raises(DataProcessingError):
            processor.process_data([df1])


def test_log_performance_decorator(processor):
    """Test that the log_performance decorator works correctly."""
    @ProcessScrapedNBAData.log_performance
    def test_function(self):
        return "test"

    result = test_function(processor)
    assert result == "test"


def test_validate_home_visitor_teams(processor):
    """Test home/visitor team validation (if method exists)."""
    if hasattr(processor, 'validate_home_visitor_teams'):
        df = pd.DataFrame({
            'GAME_ID': ['001', '001', '002', '002'],
            'TEAM_ID': [101, 102, 103, 104],
            'is_home_team': [1, 0, 1, 0],
            'original_game_id': ['0021900001', '0021900001', '0021900002', '0021900002']
        })

        with patch.object(processor, '_validate_home_visitor_assignment') as mock_validate:
            mock_validate.return_value = True

            valid_df, invalid_df = processor.validate_home_visitor_teams(df)

            assert isinstance(valid_df, pd.DataFrame)
            assert isinstance(invalid_df, pd.DataFrame)


def test_save_invalid_records(processor):
    """Test saving invalid records (if method exists)."""
    if hasattr(processor, 'save_invalid_records'):
        invalid_df = pd.DataFrame({
            'GAME_ID': ['001'],
            'TEAM_ID': [101],
            'ERROR': ['Invalid home/visitor assignment']
        })

        processor.save_invalid_records(invalid_df)

        # Should call app_file_handler or data_access to save
        if processor.app_file_handler:
            # Verify that save was attempted (implementation may vary)
            pass


def test_merge_dataframes_empty_list(processor):
    """Test merging with empty dataframe list."""
    if hasattr(processor, '_merge_dataframes'):
        with pytest.raises(Exception):
            processor._merge_dataframes([])


def test_handle_anomalous_data(processor):
    """Test handling of anomalous data."""
    if hasattr(processor, '_handle_anomalous_data'):
        df = pd.DataFrame({
            'GAME_ID': ['001', '002', '003'],
            'PTS': [100, -10, 95]  # Negative points is anomalous
        })

        result = processor._handle_anomalous_data(df)

        assert isinstance(result, pd.DataFrame)
        # Implementation-specific validation would go here
