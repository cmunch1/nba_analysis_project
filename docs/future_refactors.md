# Future Refactoring Opportunities

This document tracks design improvements and refactoring opportunities identified during development.

---

## Core Framework - Application Logger

### Issue: Leaky Traceback Handling

**Current State:**
Every module that uses the logger must:
1. Import `traceback` module
2. Manually call `traceback.format_exc()`
3. Pass it as a kwarg to `structured_log()`

**Example of current pattern:**
```python
import traceback  # Required in every file

try:
    # ... code
except Exception as e:
    self.app_logger.structured_log(
        logging.ERROR,
        "Operation failed",
        error=str(e),
        traceback=traceback.format_exc()  # Manual traceback capture
    )
```

**Problem:**
- Violates DRY principle - traceback capture repeated in every error handler
- Forces consumers to know implementation details
- Easy to forget, leading to inconsistent error logging

**Proposed Solution:**
Add automatic traceback capture to the logger:

```python
# In base_app_logger.py and app_logger.py
def structured_log(self, level: int, message: str, include_traceback: bool = False, **kwargs) -> None:
    """
    Log a structured message with additional context

    Args:
        level: Logging level (e.g., logging.INFO)
        message: Log message
        include_traceback: If True, automatically capture and include current traceback
        **kwargs: Additional context to include in log
    """
    if include_traceback:
        import traceback
        kwargs['traceback'] = traceback.format_exc()

    # ... rest of logging implementation
```

**Updated usage pattern:**
```python
# No traceback import needed

try:
    # ... code
except Exception as e:
    self.app_logger.structured_log(
        logging.ERROR,
        "Operation failed",
        include_traceback=True,  # Automatic traceback capture
        error=str(e)
    )
```

**Benefits:**
- Single responsibility - logger handles all logging concerns
- Consistent traceback formatting
- Less boilerplate in consumer code
- Traceback import encapsulated within logger

---

## Core Framework - Logging Level Constants

### Issue: Tight Coupling to Python's Logging Module

**Current State:**
All modules must import Python's `logging` module to access level constants:

```python
import logging

self.app_logger.structured_log(logging.INFO, "Message")
self.app_logger.structured_log(logging.ERROR, "Error")
```

**Problem:**
- Leaky abstraction - consumers need to know about Python's logging module
- Tightly couples all code to standard library implementation
- The `BaseAppLogger` abstraction is incomplete

**Proposed Solution (Optional):**
Expose logging levels through the logger itself:

```python
# In base_app_logger.py
class BaseAppLogger(ABC):
    # Expose logging levels as class attributes
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    # ... rest of class
```

**Updated usage pattern:**
```python
# Could use either:
self.app_logger.structured_log(self.app_logger.INFO, "Message")

# Or if we want to be more opinionated:
# Don't import logging at all, use logger constants
```

**Counterargument:**
The current approach using `logging.INFO` etc. is:
- Standard Python practice
- Well understood by all Python developers
- Not worth changing unless we want to completely abstract away Python's logging

**Recommendation:**
Keep the current approach with `logging` constants - this is actually good design for a Python-specific framework. The constants are part of Python's standard interface.

**Status:** LOW PRIORITY - Current design is acceptable

---

## Summary

### High Priority
- **Traceback Handling**: Encapsulate within logger to reduce boilerplate and improve consistency

### Low Priority
- **Logging Constants**: Current design is acceptable, no change needed

---

**Last Updated:** 2025-10-08


## Logging Improvements - Contextual Information
### Issue: Lack of Contextual Information in Logs
**Current State:**
Logs have changed and need to figure out why and fix this:
{"timestamp": "2024-12-16T17:26:53.079140+00:00", "level": "INFO", "name": "src.data_access.data_access", "message": "{\"message\": \"Data saved successfully\", \"file_path\": \"data\\\\processed\\\\games_boxscores.csv\"}"}
{"timestamp": "2024-12-16T17:26:53.080141+00:00", "level": "INFO", "name": "src.logging.logging_utils", "message": "Function performance", "function_name": "_save_dataframe_csv", "execution_time": "0.95", "unit": "seconds"}
{"timestamp": "2024-12-16T17:26:53.081140+00:00", "level": "INFO", "name": "src.data_access.data_access", "message": "{\"message\": \"Dataframes saved successfully\"}"}
{"timestamp": "2024-12-16T17:26:53.081140+00:00", "level": "INFO", "name": "src.logging.logging_utils", "message": "Function performance", "function_name": "save_dataframes", "execution_time": "0.95", "unit": "seconds"}
{"timestamp": "2024-12-16T17:26:53.082141+00:00", "level": "INFO", "name": "__main__", "message": "{\"message\": \"Data processing completed successfully\"}"}
{"timestamp": "2024-12-16T17:26:53.085140+00:00", "level": "INFO", "name": "src.logging.logging_utils", "message": "Function performance", "function_name": "main", "execution_time": "2.41", "unit": "seconds"}
2025-10-05 10:08:46,691 - ml_framework.core.app_logging.app_logger - INFO - Loading scraped data | Context: {}
2025-10-05 10:08:46,691 - ml_framework.core.app_logging.app_logger - INFO - Getting load directory | Context: {'cumulative': True, 'file_name': 'None'}
2025-10-05 10:08:46,692 - ml_framework.core.app_logging.app_logger - INFO - Loading dataframes | Context: {}


