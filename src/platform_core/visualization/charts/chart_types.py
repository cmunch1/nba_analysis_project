from enum import Enum, auto

class ChartType(Enum):
    """Enum defining available chart types."""
    FEATURE = auto()
    METRICS = auto()
    LEARNING_CURVE = auto()
    SHAP = auto()
    MODEL_INTERPRETATION = auto()  # Added new type 