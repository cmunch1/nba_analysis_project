import contextlib
import logging

import numpy as np
import pytest

from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.postprocessing.conformal_predictor import ConformalPredictor


class DummyLogger(BaseAppLogger):
    """Minimal logger for testing."""

    def __init__(self):
        # BaseAppLogger expects a config argument, but tests do not require it.
        pass

    def setup(self, log_file: str) -> logging.Logger:
        return logging.getLogger(__name__)

    def structured_log(self, level: int, message: str, **kwargs) -> None:
        # No-op logger for deterministic tests
        return None

    def log_performance(self, func):
        return func

    def log_context(self, **kwargs):
        return contextlib.nullcontext()


class DummyErrorHandler:
    """Stub error handler factory used to satisfy interface requirements."""

    def create_error_handler(self, *_, **__):
        raise AssertionError("Error handler should not be invoked during successful tests")


@pytest.fixture
def conformal_components():
    logger = DummyLogger()
    error_handler = DummyErrorHandler()
    return logger, error_handler


def test_conformal_predictor_quantiles(conformal_components):
    logger, error_handler = conformal_components
    predictor = ConformalPredictor(
        app_logger=logger,
        error_handler=error_handler,
        alpha_prediction_set=0.1,
        alpha_probability_interval=0.2,
        class_labels=['away', 'home']
    )

    y_pred = np.array([0.9, 0.8, 0.2, 0.1])
    y_true = np.array([1, 1, 0, 0])

    predictor.fit(y_pred=y_pred, y_true=y_true)
    params = predictor.get_params()

    assert pytest.approx(params['quantiles']['prediction_set']) == 0.2
    assert pytest.approx(params['quantiles']['probability_interval']) == 0.2


def test_conformal_predictor_outputs(conformal_components):
    logger, error_handler = conformal_components
    predictor = ConformalPredictor(
        app_logger=logger,
        error_handler=error_handler,
        alpha_prediction_set=0.1,
        alpha_probability_interval=0.2,
        class_labels=['away', 'home']
    )

    y_pred = np.array([0.9, 0.8, 0.2, 0.1])
    y_true = np.array([1, 1, 0, 0])
    predictor.fit(y_pred=y_pred, y_true=y_true)

    new_probs = np.array([0.9, 0.3])
    prediction_sets = predictor.predict_set(new_probs)
    assert prediction_sets[0] == ['home']
    assert set(prediction_sets[1]) == {'away', 'home'}

    intervals = predictor.predict_interval(new_probs)
    assert intervals.shape == (2, 2)
    assert pytest.approx(intervals[0, 0]) == 0.7
    assert pytest.approx(intervals[0, 1]) == 1.0
