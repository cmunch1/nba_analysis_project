from pathlib import Path
from typing import Dict, Any, Union, List, ContextManager
import yaml
import json
import pandas as pd
from src.common.core.app_file_handling.base_app_file_handler import BaseAppFileHandler
import tempfile
from contextlib import contextmanager
import matplotlib.pyplot as plt

class LocalAppFileHandler(BaseAppFileHandler):
    def read_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def write_yaml(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(data, f, indent=2)
            
    def read_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)
            
    def write_json(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def read_csv(self, path: Union[str, Path]) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return pd.read_csv(path)
            
    def write_csv(self, df: pd.DataFrame, path: Union[str, Path]) -> None:
        path = Path(path)
        df.to_csv(path, index=False)
        
    def ensure_directory(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    
    def join_paths(self, *paths: Union[str, Path]) -> Path:
        return Path(*paths)
    
    def resolve_project_root_path(self, path: str) -> str:
        """Resolves paths that contain ${PROJECT_ROOT} to absolute paths"""
        if "${PROJECT_ROOT}" in path:
            project_root = Path(__file__).parent.parent.parent
            return path.replace("${PROJECT_ROOT}", str(project_root))
        return path

    def load_yaml_files_in_directory(self, directory: Path, required_files: List[str] = None) -> Dict[str, Any]:
        """
        Load and merge all YAML files in a directory
        Args:
            directory: Directory containing YAML files
            required_files: List of filenames that must exist
        Raises:
            FileNotFoundError: If any required files are missing
        """
        config_dict = {}
        
        # Check for required files first
        if required_files:
            missing_files = []
            for required_file in required_files:
                if not (directory / required_file).exists():
                    missing_files.append(required_file)
            if missing_files:
                raise FileNotFoundError(f"Required config files not found: {', '.join(missing_files)}")
        
        # Load all YAML files
        for config_file in directory.glob('*.yaml'):
            config_dict.update(self.read_yaml(config_file))
        return config_dict
    
    @contextmanager
    def create_temp_directory(self) -> ContextManager[str]:
        """
        Create a temporary directory using tempfile.TemporaryDirectory.
        
        Returns:
            Context manager that yields the path to the temporary directory
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def save_figure(self, fig: plt.Figure, path: Union[str, Path]) -> None:
        """
        Save a matplotlib figure to the specified path.
        
        Args:
            fig: Matplotlib figure to save
            path: Path where the figure should be saved
        """
        path = Path(path)
        fig.savefig(path, bbox_inches='tight', dpi=300)
    
    