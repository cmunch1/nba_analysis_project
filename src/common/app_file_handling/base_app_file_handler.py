from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, List, ContextManager
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt

class BaseAppFileHandler(ABC):
    @abstractmethod
    def read_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Read YAML configuration file"""
        pass
        
    @abstractmethod
    def write_yaml(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Write YAML configuration file"""
        pass
        
    @abstractmethod
    def read_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Read JSON file"""
        pass
        
    @abstractmethod
    def write_json(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Write JSON file"""
        pass
        
    @abstractmethod
    def read_csv(self, path: Union[str, Path]) -> pd.DataFrame:
        """Read CSV file"""
        pass
        
    @abstractmethod
    def write_csv(self, df: pd.DataFrame, path: Union[str, Path]) -> None:
        """Write CSV file"""
        pass
        
    @abstractmethod
    def ensure_directory(self, path: Union[str, Path]) -> None:
        """Ensure directory exists"""
        pass

    @abstractmethod
    def join_paths(self, *paths: Union[str, Path]) -> Path:
        """Join path components into a single path"""
        pass

    @abstractmethod
    def resolve_project_root_path(self, path: str) -> str:
        """Resolves paths that contain ${PROJECT_ROOT} to absolute paths"""
        pass

    @abstractmethod
    def load_yaml_files_in_directory(self, directory: Path) -> Dict[str, Any]:
        """Load and merge all YAML files in a directory"""
        pass

    @abstractmethod
    def create_temp_directory(self) -> ContextManager[str]:
        """
        Create a temporary directory and return a context manager.
        The directory will be automatically cleaned up when the context exits.
        
        Returns:
            Context manager that yields the path to the temporary directory
        """
        pass

    @abstractmethod
    def save_figure(self, fig: plt.Figure, path: Union[str, Path]) -> None:
        """
        Save a matplotlib figure to the specified path.
        
        Args:
            fig: Matplotlib figure to save
            path: Path where the figure should be saved
        """
        pass

