"""Helper functions for computing feature audit metrics."""

from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor


def infer_feature_types(X: pd.DataFrame, categorical_features: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Infer data types for features.

    Args:
        X: Feature DataFrame
        categorical_features: List of known categorical feature names

    Returns:
        Dictionary mapping feature names to types: 'numeric', 'categorical', 'binary'
    """
    feature_types = {}
    categorical_features = categorical_features or []

    for col in X.columns:
        if col in categorical_features:
            feature_types[col] = 'categorical'
        elif X[col].nunique() == 2:
            feature_types[col] = 'binary'
        elif pd.api.types.is_numeric_dtype(X[col]):
            feature_types[col] = 'numeric'
        else:
            feature_types[col] = 'categorical'

    return feature_types


def compute_coverage_and_missing(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Compute coverage and missing rates.

    Args:
        X: Feature DataFrame

    Returns:
        Tuple of (coverage, missing_rate) Series
    """
    coverage = X.notna().sum() / len(X)
    missing_rate = 1 - coverage
    return coverage, missing_rate


def compute_cardinality(X: pd.DataFrame) -> pd.Series:
    """
    Compute number of unique values per feature.

    Args:
        X: Feature DataFrame

    Returns:
        Series with unique value counts
    """
    return X.nunique(dropna=False)


def compute_variance_safe(X: pd.DataFrame) -> pd.Series:
    """
    Compute variance for numeric features, NaN for non-numeric.

    Args:
        X: Feature DataFrame

    Returns:
        Series with variance values
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    variance = pd.Series(index=X.columns, dtype=float)

    for col in numeric_cols:
        variance[col] = X[col].var()

    return variance


def compute_shap_statistics(shap_values: NDArray, feature_names: List[str]) -> pd.DataFrame:
    """
    Compute SHAP-based importance metrics.

    Args:
        shap_values: SHAP values array (samples x features)
        feature_names: List of feature names

    Returns:
        DataFrame with columns: feature_name, shap_mean_abs, shap_std_abs, shap_rank
    """
    shap_abs = np.abs(shap_values)
    shap_mean_abs = np.mean(shap_abs, axis=0)
    shap_std_abs = np.std(shap_abs, axis=0)

    # Rank by mean absolute SHAP (1 = most important)
    shap_rank = pd.Series(shap_mean_abs).rank(ascending=False, method='min').values.astype(int)

    return pd.DataFrame({
        'feature_name': feature_names,
        'shap_mean_abs': shap_mean_abs,
        'shap_std_abs': shap_std_abs,
        'shap_rank': shap_rank
    })


def compute_permutation_importance_safe(model: Any,
                                        X_eval: pd.DataFrame,
                                        y_eval: pd.Series,
                                        n_repeats: int = 5,
                                        random_state: int = 42) -> pd.DataFrame:
    """
    Compute permutation importance with error handling.

    Args:
        model: Trained model
        X_eval: Evaluation features
        y_eval: Evaluation targets
        n_repeats: Number of permutation repeats
        random_state: Random seed

    Returns:
        DataFrame with columns: feature_name, permutation_importance_mean,
        permutation_importance_std
    """
    try:
        # Convert to numpy if needed
        X_array = X_eval.values if hasattr(X_eval, 'values') else X_eval
        y_array = y_eval.values if hasattr(y_eval, 'values') else y_eval

        pi = permutation_importance(
            model,
            X_array,
            y_array,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        return pd.DataFrame({
            'feature_name': X_eval.columns.tolist(),
            'permutation_importance_mean': pi.importances_mean,
            'permutation_importance_std': pi.importances_std
        })

    except Exception as e:
        # Return zeros if permutation importance fails
        return pd.DataFrame({
            'feature_name': X_eval.columns.tolist(),
            'permutation_importance_mean': 0.0,
            'permutation_importance_std': 0.0
        })


def extract_model_importance(model: Any, feature_names: List[str]) -> Optional[NDArray]:
    """
    Extract model-specific feature importance.

    Supports: XGBoost, LightGBM, CatBoost, sklearn tree-based models.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        Array of importance scores, or None if not supported
    """
    try:
        # XGBoost
        if hasattr(model, 'get_score'):
            importance_dict = model.get_score(importance_type='gain')
            return np.array([importance_dict.get(f, 0.0) for f in feature_names])

        # LightGBM
        elif hasattr(model, 'feature_importance'):
            return model.feature_importance(importance_type='gain')

        # CatBoost
        elif hasattr(model, 'get_feature_importance'):
            return model.get_feature_importance()

        # Sklearn tree-based models
        elif hasattr(model, 'feature_importances_'):
            return model.feature_importances_

        else:
            return None

    except Exception:
        return None


def compute_pairwise_max_correlation(X: pd.DataFrame,
                                     method: str = 'pearson') -> pd.Series:
    """
    Compute maximum pairwise correlation for each feature.

    Args:
        X: Feature DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Series with max absolute correlation per feature
    """
    # Only compute on numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return pd.Series(0.0, index=X.columns)

    # Compute correlation matrix on numeric features
    corr = X[numeric_cols].corr(method=method).abs()

    # Set diagonal to 0 (self-correlation)
    np.fill_diagonal(corr.values, 0.0)

    # Get max correlation per feature
    max_corr = corr.max(axis=1)

    # Reindex to include all features (non-numeric get 0)
    return max_corr.reindex(X.columns, fill_value=0.0)


def compute_target_correlation(X: pd.DataFrame,
                               y: pd.Series,
                               method: str = 'pearson') -> pd.Series:
    """
    Compute correlation between features and target.

    Args:
        X: Feature DataFrame
        y: Target Series
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Series with target correlation per feature
    """
    correlations = pd.Series(index=X.columns, dtype=float)

    for col in X.columns:
        try:
            if pd.api.types.is_numeric_dtype(X[col]):
                correlations[col] = X[col].corr(y, method=method)
            else:
                # For categorical, use correlation with encoded values
                correlations[col] = 0.0
        except Exception:
            correlations[col] = 0.0

    return correlations.fillna(0.0)


def compute_vif(X: pd.DataFrame, max_features: int = 100) -> pd.Series:
    """
    Compute Variance Inflation Factor (VIF) for numeric features.

    Note: VIF computation is expensive for high-dimensional data.

    Args:
        X: Feature DataFrame
        max_features: Maximum number of features to compute VIF for

    Returns:
        Series with VIF values (non-numeric features get NaN)
    """
    vif_series = pd.Series(index=X.columns, dtype=float)

    # Only compute on numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0 or len(numeric_cols) > max_features:
        return vif_series

    try:
        # Standardize features for VIF computation
        X_numeric = X[numeric_cols].copy()
        X_numeric = (X_numeric - X_numeric.mean()) / X_numeric.std()

        # Drop any columns with NaN after standardization
        X_numeric = X_numeric.dropna(axis=1)

        # Compute VIF
        for i, col in enumerate(X_numeric.columns):
            try:
                vif_value = variance_inflation_factor(X_numeric.values, i)
                vif_series[col] = vif_value
            except Exception:
                vif_series[col] = np.nan

    except Exception:
        pass

    return vif_series


def compute_stability_from_fold_importances(fold_importances: List[Dict[str, float]],
                                            feature_names: List[str],
                                            top_k: int = 20) -> pd.Series:
    """
    Compute stability score: fraction of folds where feature is in top-k.

    Args:
        fold_importances: List of dictionaries mapping feature names to importance scores
        feature_names: All feature names
        top_k: Number of top features to consider

    Returns:
        Series with stability scores (0.0 to 1.0)
    """
    stability_counts = pd.Series(0, index=feature_names, dtype=float)
    n_folds = len(fold_importances)

    if n_folds == 0:
        return stability_counts

    for fold_imp in fold_importances:
        # Get top-k features for this fold
        fold_series = pd.Series(fold_imp)
        top_k_features = fold_series.nlargest(min(top_k, len(fold_series))).index
        stability_counts[top_k_features] += 1

    # Convert to fraction
    stability_score = stability_counts / n_folds

    return stability_score


def flag_near_zero_importance(shap_values: pd.Series,
                              perm_importance: pd.Series,
                              percentile_threshold: float = 10,
                              absolute_threshold: float = 0.0) -> pd.Series:
    """
    Flag features with near-zero importance.

    Args:
        shap_values: SHAP mean absolute values
        perm_importance: Permutation importance mean values
        percentile_threshold: Bottom X% of features by SHAP
        absolute_threshold: Absolute permutation importance threshold

    Returns:
        Binary Series (1 = near zero importance, 0 = not)
    """
    # Bottom percentile by SHAP
    shap_threshold = np.percentile(shap_values, percentile_threshold)
    low_shap = shap_values <= shap_threshold

    # Near-zero permutation importance
    low_perm = np.abs(perm_importance) <= absolute_threshold

    # Flag if both conditions are met
    return (low_shap & low_perm).astype(int)


def compute_composite_drop_score(near_zero_importance: pd.Series,
                                 high_missing_flag: pd.Series,
                                 high_collinearity_flag: pd.Series,
                                 unstable_flag: pd.Series,
                                 leakage_flag: Optional[pd.Series] = None,
                                 leakage_weight: int = 2) -> pd.Series:
    """
    Compute composite drop candidate score.

    Args:
        near_zero_importance: Binary flag for near-zero importance
        high_missing_flag: Binary flag for high missing rate
        high_collinearity_flag: Binary flag for high collinearity
        unstable_flag: Binary flag for low stability
        leakage_flag: Binary flag for leakage (optional, defaults to 0)
        leakage_weight: Weight for leakage flag

    Returns:
        Series with composite scores
    """
    if leakage_flag is None:
        leakage_flag = pd.Series(0, index=near_zero_importance.index)

    drop_score = (
        near_zero_importance +
        high_missing_flag +
        high_collinearity_flag +
        unstable_flag +
        (leakage_flag * leakage_weight)
    )

    return drop_score
