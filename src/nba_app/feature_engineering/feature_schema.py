"""
Feature schema management for NBA features.

This module provides classes and utilities for defining, exporting, and managing
feature schemas. The schema captures feature types (numeric, categorical, binary, etc.)
and metadata about features produced by feature engineering.

Design Principle:
- Feature schemas are produced by nba_app.feature_engineering (domain layer)
- Feature schemas are consumed by ml_framework.preprocessing (model layer)
- This ensures the preprocessor knows which columns to scale, encode, etc.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Set
import pandas as pd
import json
from pathlib import Path
import logging


@dataclass
class FeatureSchema:
    """
    Schema describing features and their types.

    This schema is produced by feature engineering and consumed by preprocessing
    to ensure correct handling of different feature types.

    Attributes:
        numeric_features: Continuous numeric features (floats/ints that aren't binary)
        categorical_features: Categorical features (strings, low-cardinality)
        binary_features: Binary features (0/1, True/False)
        ordinal_features: Ordinal categorical features (manually specified order)
        datetime_features: DateTime features (dates, timestamps)
        target_column: Name of the target/label column
        game_id_column: Name of the game ID column (for tracking)
        team_id_column: Optional team ID column
        date_column: Optional date column for temporal splits
        feature_descriptions: Optional metadata describing each feature
        feature_groups: Optional grouping of related features
    """
    numeric_features: List[str]
    categorical_features: List[str]
    binary_features: List[str]
    ordinal_features: List[str]
    datetime_features: List[str]
    target_column: str
    game_id_column: str
    team_id_column: Optional[str] = None
    date_column: Optional[str] = None
    feature_descriptions: Dict[str, str] = field(default_factory=dict)
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """
        Save schema to JSON file.

        Args:
            path: Path to save the JSON schema
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> 'FeatureSchema':
        """
        Load schema from JSON file.

        Args:
            path: Path to the JSON schema file

        Returns:
            FeatureSchema instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target_column: str,
        game_id_column: str,
        team_id_column: Optional[str] = None,
        date_column: Optional[str] = None,
        categorical_threshold: int = 10,
        exclude_columns: Optional[List[str]] = None
    ) -> 'FeatureSchema':
        """
        Infer schema from DataFrame.

        Args:
            df: Input DataFrame with features
            target_column: Name of target column
            game_id_column: Name of game ID column
            team_id_column: Optional name of team ID column
            date_column: Optional name of date column
            categorical_threshold: Max unique values for auto-detecting categoricals
            exclude_columns: Columns to exclude from feature lists

        Returns:
            Inferred FeatureSchema instance
        """
        exclude_columns = exclude_columns or []
        exclude_set = set(exclude_columns + [target_column, game_id_column])
        if team_id_column:
            exclude_set.add(team_id_column)
        if date_column:
            exclude_set.add(date_column)

        # Get initial type-based feature lists
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_features = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Identify binary features (0/1 or True/False)
        binary_features = []
        for col in numeric_features:
            if df[col].nunique() == 2:
                unique_vals = set(df[col].dropna().unique())
                if unique_vals.issubset({0, 1, True, False}):
                    binary_features.append(col)

        # Check for low-cardinality numerics that should be categorical
        potential_categoricals = []
        for col in numeric_features:
            if col not in binary_features:
                if df[col].nunique() <= categorical_threshold:
                    potential_categoricals.append(col)

        # Remove binary features from numeric
        numeric_features = [f for f in numeric_features if f not in binary_features]

        # Remove potential categoricals from numeric and add to categorical
        numeric_features = [f for f in numeric_features if f not in potential_categoricals]
        categorical_features.extend(potential_categoricals)

        # Remove excluded columns from all lists
        numeric_features = [f for f in numeric_features if f not in exclude_set]
        categorical_features = [f for f in categorical_features if f not in exclude_set]
        binary_features = [f for f in binary_features if f not in exclude_set]
        datetime_features = [f for f in datetime_features if f not in exclude_set]

        return cls(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            binary_features=binary_features,
            ordinal_features=[],  # Must be specified manually
            datetime_features=datetime_features,
            target_column=target_column,
            game_id_column=game_id_column,
            team_id_column=team_id_column,
            date_column=date_column,
            feature_descriptions={},
            feature_groups={}
        )

    def get_all_features(self) -> List[str]:
        """
        Get all feature columns (excluding target and IDs).

        Returns:
            List of all feature column names
        """
        return (
            self.numeric_features +
            self.categorical_features +
            self.binary_features +
            self.ordinal_features +
            self.datetime_features
        )

    def get_modeling_features(self) -> List[str]:
        """
        Get features suitable for modeling (excludes datetime features).

        Returns:
            List of feature column names for modeling
        """
        return (
            self.numeric_features +
            self.categorical_features +
            self.binary_features +
            self.ordinal_features
        )

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate that a DataFrame conforms to this schema.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results:
            - 'missing_columns': Expected columns not in DataFrame
            - 'extra_columns': Columns in DataFrame not in schema
            - 'type_mismatches': Columns with unexpected types
        """
        validation_results = {
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': []
        }

        expected_columns = set(self.get_all_features())
        expected_columns.add(self.target_column)
        expected_columns.add(self.game_id_column)
        if self.team_id_column:
            expected_columns.add(self.team_id_column)
        if self.date_column:
            expected_columns.add(self.date_column)

        actual_columns = set(df.columns)

        # Check for missing and extra columns
        validation_results['missing_columns'] = list(expected_columns - actual_columns)
        validation_results['extra_columns'] = list(actual_columns - expected_columns)

        # Check types for present columns
        for col in expected_columns.intersection(actual_columns):
            actual_dtype = df[col].dtype

            # Check numeric features
            if col in self.numeric_features or col in self.binary_features:
                if not pd.api.types.is_numeric_dtype(actual_dtype):
                    validation_results['type_mismatches'].append(
                        f"{col}: expected numeric, got {actual_dtype}"
                    )

            # Check categorical features
            elif col in self.categorical_features or col in self.ordinal_features:
                if not (pd.api.types.is_object_dtype(actual_dtype) or
                       pd.api.types.is_categorical_dtype(actual_dtype)):
                    validation_results['type_mismatches'].append(
                        f"{col}: expected categorical/object, got {actual_dtype}"
                    )

            # Check datetime features
            elif col in self.datetime_features:
                if not pd.api.types.is_datetime64_any_dtype(actual_dtype):
                    validation_results['type_mismatches'].append(
                        f"{col}: expected datetime, got {actual_dtype}"
                    )

        return validation_results

    def summary(self) -> str:
        """
        Generate a human-readable summary of the schema.

        Returns:
            String summary of the schema
        """
        lines = [
            "Feature Schema Summary",
            "=" * 60,
            f"Target Column: {self.target_column}",
            f"Game ID Column: {self.game_id_column}",
            f"Team ID Column: {self.team_id_column or 'None'}",
            f"Date Column: {self.date_column or 'None'}",
            "",
            f"Numeric Features: {len(self.numeric_features)}",
            f"Categorical Features: {len(self.categorical_features)}",
            f"Binary Features: {len(self.binary_features)}",
            f"Ordinal Features: {len(self.ordinal_features)}",
            f"DateTime Features: {len(self.datetime_features)}",
            "",
            f"Total Features: {len(self.get_all_features())}",
            f"Modeling Features: {len(self.get_modeling_features())}",
        ]

        if self.feature_groups:
            lines.extend([
                "",
                "Feature Groups:",
                "-" * 60
            ])
            for group_name, features in self.feature_groups.items():
                lines.append(f"  {group_name}: {len(features)} features")

        return "\n".join(lines)
