from typing import Dict, Tuple
import logging
import numpy as np
from ml_framework.framework.data_classes import ModelTrainingResults
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler

class TrainerUtils:
    """Utility class providing common functionality for trainers."""
    
    def __init__(self, app_logger: BaseAppLogger, error_handler: BaseErrorHandler):
        self.app_logger = app_logger
        self.error_handler = error_handler

    def _convert_metric_scores(self, 
                             train_score: float, 
                             val_score: float, 
                             metric_name: str) -> Tuple[float, float]:
        """Convert metric scores to a consistent format (higher is better)."""
        try:
            lower_is_better = ['logloss', 'binary_logloss', 'multi_logloss', 'rmse', 'mae']
            
            if any(metric in metric_name.lower() for metric in lower_is_better):
                return -train_score, -val_score
            
            return train_score, val_score
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error converting metric scores",
                original_error=str(e),
                metric_name=metric_name
            )

    def _process_learning_curve_data(self, 
                                   evals_result: Dict, 
                                   results: ModelTrainingResults) -> None:
        """Process and store learning curve data."""
        try:
            # Get first metric from evaluation results
            eval_metric = next(iter(evals_result['train'].keys()))
            results.learning_curve_data.metric_name = eval_metric

            # Process each iteration's scores
            for i, (train_score, val_score) in enumerate(zip(
                evals_result['train'][eval_metric],
                evals_result['eval'][eval_metric]
            )):
                train_score_converted, val_score_converted = self._convert_metric_scores(
                    train_score,
                    val_score,
                    eval_metric
                )
                
                results.learning_curve_data.add_iteration(
                    train_score=train_score_converted,
                    val_score=val_score_converted,
                    iteration=i
                )
                
            self.app_logger.structured_log(
                logging.INFO,
                "Processed learning curve data",
                metric_name=eval_metric,
                iterations=i+1
            )
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error processing learning curve data",
                original_error=str(e)
            )