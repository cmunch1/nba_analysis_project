# ML Framework Core Module Usage Guide for AI Code Assistants

## Project Context

**Environment:**
- VSCode in Windows with files on WSL partition
- Package management: `uv` (not pip)
- Configuration: `pyproject.toml` (not requirements.txt)
- Project structure: Recently reorganized into `src/nba_app/` and `src/ml_framework/`

**Core Framework Location:** `src/ml_framework/core/`

This guide explains the mandatory patterns for using core framework modules. **DO NOT** bypass these patterns by directly reading config files, implementing custom logging, or creating ad-hoc error handling.

---

## Core Principles

### 1. Dependency Injection (DI) Container Pattern

**All modules MUST use dependency injection** - never instantiate core services directly in your code.

#### Example: Correct Pattern

```python
# In di_container.py
from dependency_injector import containers, providers
from ml_framework.core.common_di_container import CommonDIContainer

class MyModuleDIContainer(containers.DeclarativeContainer):
    # Import common container
    common = providers.Container(CommonDIContainer)

    # Use common container's components (DO NOT recreate them)
    config = common.config
    app_logger = common.app_logger
    app_file_handler = common.app_file_handler
    error_handler = common.error_handler_factory
    data_access = common.data_access

    # Your module-specific components
    my_component = providers.Factory(
        MyComponent,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler
    )
```

#### Example: Incorrect Pattern (Never Do This)

```python
# ❌ WRONG - Direct instantiation
from ml_framework.core.config_management.config_manager import ConfigManager
config = ConfigManager()  # NEVER DO THIS

# ❌ WRONG - Direct file reading
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)  # NEVER DO THIS

# ❌ WRONG - Manual logger setup
import logging
logger = logging.getLogger(__name__)  # NEVER DO THIS
```

---

## Core Module Components

### 2. Configuration Management

**Module:** `src/ml_framework/core/config_management/`

#### Key Classes:
- `BaseConfigManager` - Abstract base class
- `ConfigManager` - Concrete implementation
- Injected via DI container

#### Rules:

1. **NEVER read config files directly** - ConfigManager handles all YAML loading
2. **NEVER use hardcoded paths** - Use config properties
3. **Access nested configs via dot notation**
4. **ConfigManager automatically loads and merges all YAML files** in the config directory

#### Usage Pattern:

```python
class MyClass:
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler

    def do_something(self):
        # Access config properties via dot notation
        log_path = self.config.log_path

        # Access nested configs
        model_config = self.config.models.xgboost

        # Access config with defaults
        batch_size = getattr(self.config, 'batch_size', 32)
```

#### Common Config Access Patterns:

```python
# Simple property access
training_file = self.config.training_data_file

# Nested config access
n_splits = self.config.cross_validation_config.n_splits

# Model-specific config
model_params = self.config.models.xgboost.hyperparameters

# Check if property exists
if hasattr(self.config, 'experiment_name'):
    exp_name = self.config.experiment_name
```

#### Config Helper Methods Pattern:

```python
def get_model_config(self, model_name: str) -> Any:
    """Get model-specific configuration."""
    try:
        return getattr(self.config.models, model_name)
    except AttributeError:
        raise self.error_handler.create_error_handler(
            'configuration',
            f"Configuration not found for model: {model_name}"
        )

def get_model_config_value(self, model_name: str, key: str, default: Any) -> Any:
    """Get a configuration value with fallback to default."""
    try:
        model_config = self.get_model_config(model_name)
        return getattr(model_config, key, default)
    except Exception:
        return default
```

---

### 3. Application Logging

**Module:** `src/ml_framework/core/app_logging/`

#### Key Classes:
- `BaseAppLogger` - Abstract base class
- `AppLogger` - Concrete implementation
- Injected via DI container

#### Rules:

1. **NEVER use Python's logging module directly**
2. **NEVER create custom loggers**
3. **Use structured_log() for all logging**
4. **Use log_performance decorator for performance-critical methods**

#### Usage Pattern:

```python
import logging

class MyClass:
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler

        # Log initialization
        self.app_logger.structured_log(
            logging.INFO,
            "MyClass initialized",
            config_type=type(config).__name__
        )

    def process_data(self, df):
        """Regular method with logging."""
        self.app_logger.structured_log(
            logging.INFO,
            "Starting data processing",
            input_shape=df.shape,
            columns=list(df.columns)
        )

        try:
            # Process data...
            result = df.copy()

            self.app_logger.structured_log(
                logging.INFO,
                "Data processing completed",
                output_shape=result.shape
            )
            return result

        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                "Data processing failed",
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise
```

#### Performance Logging Decorator:

**REQUIRED for performance-critical methods** like data loading, model training, preprocessing, etc.

```python
class MyClass:
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging."""
        def wrapper(*args, **kwargs):
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def expensive_operation(self, data):
        """This method's performance will be automatically logged."""
        # The decorator logs: execution time, memory usage, etc.
        result = self._do_processing(data)
        return result
```

#### Logging Levels:

```python
# Information about normal operation
self.app_logger.structured_log(logging.INFO, "Operation started", param=value)

# Warning about potential issues
self.app_logger.structured_log(logging.WARNING, "Low memory detected", available_mb=1024)

# Error conditions
self.app_logger.structured_log(logging.ERROR, "Operation failed", error=str(e))

# Debug information (only in debug mode)
self.app_logger.structured_log(logging.DEBUG, "Intermediate result", data=debug_data)
```

---

### 4. Error Handling

**Module:** `src/ml_framework/core/error_handling/`

#### Key Classes:
- `BaseErrorHandler` - Abstract base class
- `ErrorHandlerFactory` - Creates typed error handlers
- Injected via DI container

#### Available Error Types:

```python
# Core errors
'configuration'        # ConfigurationError
'webdriver'           # WebDriverError

# Scraping errors
'scraping'            # ScrapingError
'page_load'           # PageLoadError
'element_not_found'   # ElementNotFoundError
'data_extraction'     # DataExtractionError
'dynamic_content_load' # DynamicContentLoadError

# Data errors
'data_processing'     # DataProcessingError
'data_validation'     # DataValidationError
'data_storage'        # DataStorageError

# ML/Analytics errors
'feature_engineering' # FeatureEngineeringError
'feature_selection'   # FeatureSelectionError
'model_testing'       # ModelTestingError
'preprocessing'       # PreprocessingError
'optimization'        # OptimizationError

# Visualization errors
'chart_creation'      # ChartCreationError

# Experiment logging errors
'experiment_logger'   # ExperimentLoggerError
```

#### Usage Pattern:

```python
class MyClass:
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler

    def process_model(self, model_name: str, data):
        """Example method with proper error handling."""
        try:
            # Attempt operation
            result = self._train_model(model_name, data)
            return result

        except Exception as e:
            # Use error handler to create typed exception
            raise self.error_handler.create_error_handler(
                'model_testing',
                f"Error processing model {model_name}",
                error_message=str(e),
                traceback=traceback.format_exc(),
                model_name=model_name,
                input_shape=data.shape
            )

    def load_config_value(self, key: str):
        """Example with configuration error."""
        try:
            return getattr(self.config, key)
        except AttributeError as e:
            raise self.error_handler.create_error_handler(
                'configuration',
                f"Required configuration key not found: {key}",
                original_error=str(e),
                available_keys=list(vars(self.config).keys())
            )
```

#### Error Handling in main.py:

```python
def main() -> None:
    """Main function with proper error handling."""
    app_logger = None
    error_handler = None

    try:
        # Initialize container
        container = MyModuleDIContainer()

        # Get dependencies
        config = container.config()
        app_logger = container.app_logger()
        error_handler = container.error_handler()

        # Setup logger
        log_file = config.log_path / "my_module.log"
        app_logger.setup(log_file)

        # Your main logic here
        result = do_work(config, app_logger, error_handler)

    except Exception as e:
        if error_handler and app_logger:
            # Use error handler if available
            raise error_handler.create_error_handler(
                'data_processing',
                "Main process failed",
                error_message=str(e),
                traceback=traceback.format_exc()
            )
        elif app_logger:
            # If only logger was initialized
            app_logger.structured_log(
                logging.ERROR,
                "Main process failed",
                error=str(e),
                traceback=traceback.format_exc()
            )
        else:
            # Fallback to basic logging
            print(f"ERROR: Main process failed: {str(e)}")
            print(traceback.format_exc())
        sys.exit(1)
```

---

### 5. File Handling

**Module:** `src/ml_framework/core/app_file_handling/`

#### Key Classes:
- `BaseAppFileHandler` - Abstract base class
- `LocalAppFileHandler` - Concrete implementation for local files
- Injected via DI container

#### Rules:

1. **NEVER use open(), yaml.load(), json.load() directly**
2. **Use app_file_handler methods for all file operations**
3. **Use join_paths() for path construction**
4. **Use ensure_directory() before writing files**

#### Usage Pattern:

```python
class MyClass:
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 app_file_handler: BaseAppFileHandler,
                 error_handler: BaseErrorHandler):
        self.config = config
        self.app_logger = app_logger
        self.app_file_handler = app_file_handler
        self.error_handler = error_handler

    def load_data(self):
        """Load data using file handler."""
        # Read CSV
        df = self.app_file_handler.read_csv(self.config.data_file)

        # Read YAML
        settings = self.app_file_handler.read_yaml(self.config.settings_file)

        # Read JSON
        metadata = self.app_file_handler.read_json(self.config.metadata_file)

        return df, settings, metadata

    def save_results(self, df, metadata):
        """Save results using file handler."""
        # Construct output path
        output_path = self.app_file_handler.join_paths(
            self.config.output_dir,
            "results"
        )

        # Ensure directory exists
        self.app_file_handler.ensure_directory(output_path)

        # Save files
        csv_file = self.app_file_handler.join_paths(output_path, "results.csv")
        self.app_file_handler.write_csv(df, csv_file)

        json_file = self.app_file_handler.join_paths(output_path, "metadata.json")
        self.app_file_handler.write_json(metadata, json_file)
```

---

## Complete Module Template

Here's a complete template for creating a new module that properly uses the core framework:

### Directory Structure:

```
src/ml_framework/my_new_module/
├── __init__.py
├── di_container.py          # Dependency injection setup
├── main.py                  # Entry point
├── base_my_component.py     # Abstract base class
└── my_component.py          # Concrete implementation
```

### 1. Base Class (`base_my_component.py`):

```python
from abc import ABC, abstractmethod
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.core.app_file_handling.base_app_file_handler import BaseAppFileHandler

class BaseMyComponent(ABC):
    @abstractmethod
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 app_file_handler: BaseAppFileHandler,
                 error_handler: BaseErrorHandler):
        """Initialize with required dependencies."""
        pass

    @property
    @abstractmethod
    def log_performance(self):
        """Get the performance logging decorator from app_logger."""
        pass

    @abstractmethod
    def process(self, data):
        """Main processing method."""
        pass
```

### 2. Concrete Implementation (`my_component.py`):

```python
import logging
import traceback
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.core.app_file_handling.base_app_file_handler import BaseAppFileHandler
from .base_my_component import BaseMyComponent

class MyComponent(BaseMyComponent):
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 app_file_handler: BaseAppFileHandler,
                 error_handler: BaseErrorHandler):
        self.config = config
        self.app_logger = app_logger
        self.app_file_handler = app_file_handler
        self.error_handler = error_handler

        self.app_logger.structured_log(
            logging.INFO,
            "MyComponent initialized",
            config_type=type(config).__name__
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging."""
        def wrapper(*args, **kwargs):
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def process(self, data):
        """Process data with proper logging and error handling."""
        self.app_logger.structured_log(
            logging.INFO,
            "Starting data processing",
            input_shape=data.shape
        )

        try:
            # Your processing logic here
            result = self._do_processing(data)

            self.app_logger.structured_log(
                logging.INFO,
                "Processing completed",
                output_shape=result.shape
            )

            return result

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Data processing failed",
                error_message=str(e),
                traceback=traceback.format_exc(),
                input_shape=data.shape
            )

    def _do_processing(self, data):
        """Internal processing method."""
        # Implementation here
        return data
```

### 3. DI Container (`di_container.py`):

```python
from dependency_injector import containers, providers
from ml_framework.core.common_di_container import CommonDIContainer
from .my_component import MyComponent

class MyModuleDIContainer(containers.DeclarativeContainer):
    # Import common container
    common = providers.Container(CommonDIContainer)

    # Use common container's components
    config = common.config
    app_logger = common.app_logger
    app_file_handler = common.app_file_handler
    error_handler = common.error_handler_factory
    data_access = common.data_access

    # Module-specific components
    my_component = providers.Factory(
        MyComponent,
        config=config,
        app_logger=app_logger,
        app_file_handler=app_file_handler,
        error_handler=error_handler
    )
```

### 4. Main Entry Point (`main.py`):

```python
import sys
import logging
import traceback
from .di_container import MyModuleDIContainer

def main() -> None:
    """Main function with proper error handling."""
    app_logger = None
    error_handler = None

    try:
        # Initialize container
        container = MyModuleDIContainer()

        # Get core dependencies
        config = container.config()
        app_file_handler = container.app_file_handler()
        app_logger = container.app_logger()

        # Setup logger
        log_file = app_file_handler.join_paths(
            config.log_path,
            config.my_module_log_file
        )
        app_logger.setup(log_file)

        error_handler = container.error_handler()
        data_access = container.data_access()

        # Get module-specific dependencies
        my_component = container.my_component()

        app_logger.structured_log(
            logging.INFO,
            "Starting my module"
        )

        # Load data
        data = data_access.load_dataframe(config.input_data_file)

        # Process
        result = my_component.process(data)

        # Save results
        data_access.save_dataframes([result], ["output.csv"])

        app_logger.structured_log(
            logging.INFO,
            "My module completed successfully"
        )

    except Exception as e:
        if error_handler and app_logger:
            raise error_handler.create_error_handler(
                'data_processing',
                "My module failed",
                error_message=str(e),
                traceback=traceback.format_exc()
            )
        elif app_logger:
            app_logger.structured_log(
                logging.ERROR,
                "My module failed",
                error=str(e),
                traceback=traceback.format_exc()
            )
        else:
            print(f"ERROR: My module failed: {str(e)}")
            print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Common Mistakes to Avoid

### ❌ DON'T: Direct Config File Reading

```python
# WRONG
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

### ✅ DO: Use ConfigManager

```python
# CORRECT
def __init__(self, config: BaseConfigManager, ...):
    self.config = config
    value = self.config.some_setting
```

---

### ❌ DON'T: Direct Logger Creation

```python
# WRONG
import logging
logger = logging.getLogger(__name__)
logger.info("Processing data")
```

### ✅ DO: Use AppLogger

```python
# CORRECT
def __init__(self, app_logger: BaseAppLogger, ...):
    self.app_logger = app_logger
    self.app_logger.structured_log(logging.INFO, "Processing data", data_size=1000)
```

---

### ❌ DON'T: Generic Exceptions

```python
# WRONG
raise Exception("Something went wrong")
```

### ✅ DO: Use ErrorHandler

```python
# CORRECT
raise self.error_handler.create_error_handler(
    'data_processing',
    "Data processing failed",
    error_message=str(e),
    context={"batch_id": batch_id}
)
```

---

### ❌ DON'T: Direct File Operations

```python
# WRONG
with open('data.csv') as f:
    data = pd.read_csv(f)
```

### ✅ DO: Use AppFileHandler

```python
# CORRECT
data = self.app_file_handler.read_csv(self.config.data_file)
```

---

### ❌ DON'T: Missing Performance Logging

```python
# WRONG
def expensive_operation(self, data):
    # Long-running operation without logging
    return result
```

### ✅ DO: Use log_performance Decorator

```python
# CORRECT
@log_performance
def expensive_operation(self, data):
    # Performance automatically logged
    return result
```

---

## Reference Implementation

See `src/ml_framework/model_testing/` for a complete, production-ready example:

- [di_container.py](../../../src/ml_framework/model_testing/di_container.py) - DI setup
- [main.py](../../../src/ml_framework/model_testing/main.py) - Entry point with error handling
- [base_model_testing.py](../../../src/ml_framework/model_testing/base_model_testing.py) - Abstract base class
- [model_tester.py](../../../src/ml_framework/model_testing/model_tester.py) - Concrete implementation

---

## Checklist for New Modules

When creating new modules, verify:

- [ ] Uses `CommonDIContainer` for core dependencies
- [ ] All dependencies injected via constructor
- [ ] Abstract base class defined with type hints
- [ ] Concrete implementation extends base class
- [ ] `@staticmethod` decorator for `log_performance`
- [ ] `@log_performance` decorator on performance-critical methods
- [ ] All logging uses `app_logger.structured_log()`
- [ ] All errors use `error_handler.create_error_handler()`
- [ ] All config access via `self.config` (no direct file reads)
- [ ] All file operations via `app_file_handler`
- [ ] `main.py` has try/except with proper fallback logging
- [ ] No direct imports of `logging`, `yaml`, `json.load`, `open()`

---

---

## ML Framework Structure

The `src/ml_framework/` directory is organized into functional layers:

```
src/ml_framework/
├── core/                    # Core infrastructure (config, logging, errors, file handling)
├── framework/               # Reusable framework components (data access, validation)
├── model_testing/           # Model training and evaluation
├── preprocessing/           # Data preprocessing and feature engineering
├── uncertainty/             # Uncertainty quantification
└── visualization/           # Chart generation and orchestration
```

### Framework Layer Components

The `framework/` directory contains reusable components that are injected via the DI container.

#### Data Access (`framework/data_access/`)

**Available in CommonDIContainer as:** `data_access`

Provides abstracted data loading/saving operations.

**Base Class:** `BaseDataAccess`
**Concrete Implementation:** `CSVDataAccess` (default in CommonDIContainer)

```python
class MyClass:
    def __init__(self,
                 config: BaseConfigManager,
                 data_access: BaseDataAccess,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        self.config = config
        self.data_access = data_access
        self.app_logger = app_logger
        self.error_handler = error_handler

    def load_training_data(self):
        """Load training data using data access layer."""
        # Load single DataFrame
        df = self.data_access.load_dataframe(self.config.training_data_file)

        # Load multiple DataFrames from a directory
        dfs, file_names = self.data_access.load_scraped_data(cumulative=True)

        return df

    def save_results(self, df):
        """Save results using data access layer."""
        # Save single DataFrame
        self.data_access.save_dataframes(
            dataframes=[df],
            file_names=["output.csv"]
        )

        # Save to cumulative directory
        self.data_access.save_dataframes(
            dataframes=[df],
            file_names=["cumulative_output.csv"],
            cumulative=True
        )
```

**Key Methods:**
- `load_dataframe(file_name)` - Load single CSV file
- `load_scraped_data(cumulative=False)` - Load multiple CSVs from directory
- `save_dataframes(dataframes, file_names, cumulative=False)` - Save DataFrames

**Note:** DataAccess automatically uses the configured directories from config (e.g., `processed_data_directory`, `scraped_data_directory`).

#### Data Validation (`framework/base_data_validator.py`)

**Available in CommonDIContainer as:** `data_validator`

Validates dataframes against expected schemas and business rules.

**Base Class:** `BaseDataValidator`

```python
class MyClass:
    def __init__(self,
                 config: BaseConfigManager,
                 data_access: BaseDataAccess,
                 data_validator: BaseDataValidator,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        self.config = config
        self.data_access = data_access
        self.data_validator = data_validator
        self.app_logger = app_logger
        self.error_handler = error_handler

    def process_data(self):
        """Load and validate data."""
        # Load data
        df = self.data_access.load_dataframe(self.config.training_data_file)

        # Validate processed dataframe
        is_valid = self.data_validator.validate_processed_dataframe(
            df,
            self.config.training_data_file
        )

        if not is_valid:
            raise self.error_handler.create_error_handler(
                'data_validation',
                "Training data validation failed"
            )

        return df
```

**Key Methods:**
- `validate_scraped_dataframes(dfs, file_names)` - Validate raw scraped data
- `validate_processed_dataframe(df, file_name)` - Validate processed data

---

### Other ML Framework Modules

#### Preprocessing (`preprocessing/`)

**Base Class:** `BasePreprocessor`

Handles data preprocessing and feature engineering.

```python
class MyPreprocessor(BasePreprocessor):
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 app_file_handler: BaseAppFileHandler,
                 error_handler: BaseErrorHandler):
        self.config = config
        self.app_logger = app_logger
        self.app_file_handler = app_file_handler
        self.error_handler = error_handler

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging."""
        def wrapper(*args, **kwargs):
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def fit_transform(self, X, y=None, model_name=None, preprocessing_results=None):
        """Fit preprocessor and transform data."""
        # Your preprocessing logic
        X_transformed = self._apply_transformations(X)

        # Store preprocessing artifacts
        if preprocessing_results is None:
            preprocessing_results = PreprocessingResults()

        preprocessing_results.feature_names = list(X_transformed.columns)

        return X_transformed, preprocessing_results

    @log_performance
    def transform(self, X):
        """Transform new data using fitted preprocessor."""
        return self._apply_transformations(X)
```

**Usage in DI Container:**

```python
from ml_framework.preprocessing.preprocessor import Preprocessor

class MyModuleDIContainer(containers.DeclarativeContainer):
    common = providers.Container(CommonDIContainer)

    preprocessor = providers.Singleton(
        Preprocessor,
        config=common.config,
        app_logger=common.app_logger,
        app_file_handler=common.app_file_handler,
        error_handler=common.error_handler_factory
    )
```

#### Visualization (`visualization/`)

**Base Class:** `BaseChartOrchestrator`

Handles chart generation and saving.

```python
from ml_framework.visualization.orchestration.chart_orchestrator import ChartOrchestrator

class MyModuleDIContainer(containers.DeclarativeContainer):
    common = providers.Container(CommonDIContainer)

    chart_orchestrator = providers.Singleton(
        ChartOrchestrator,
        config=common.config,
        app_logger=common.app_logger,
        error_handler=common.error_handler_factory,
        app_file_handler=common.app_file_handler
    )
```

**Usage:**

```python
def create_charts(self, results):
    """Generate and save charts."""
    # Create charts from results
    charts = self.chart_orchestrator.create_model_evaluation_charts(results)

    # Save charts to directory
    output_dir = "charts/experiment_1"
    self.chart_orchestrator.save_charts(charts, output_dir)
```

---

## App File Handler vs Core File Handler

There are TWO file handling modules in the framework:

### 1. **Core File Handler** (`core/app_file_handling/`)

**Purpose:** Basic file I/O operations (YAML, JSON, CSV, path handling)

**When to use:** For simple file reading/writing operations

```python
# Read config file
config_data = self.app_file_handler.read_yaml("config.yaml")

# Write JSON
self.app_file_handler.write_json(data, "output.json")

# Path operations
full_path = self.app_file_handler.join_paths(base_dir, "subdir", "file.csv")
self.app_file_handler.ensure_directory(output_dir)
```

### 2. **Framework Data Access** (`framework/data_access/`)

**Purpose:** Application-specific data loading/saving with config integration

**When to use:** For loading/saving application data (training data, predictions, etc.)

```python
# Load training data (uses config to determine directory)
df = self.data_access.load_dataframe(self.config.training_data_file)

# Save predictions (uses config to determine output directory)
self.data_access.save_dataframes([df], ["predictions.csv"])
```

**Rule of Thumb:**
- Use `app_file_handler` for config files, artifacts, and non-data files
- Use `data_access` for loading/saving DataFrames and application data

---

## Questions?

If you're uncertain about implementing a pattern, refer to:
1. This document
2. `src/ml_framework/model_testing/` (reference implementation)
3. `src/ml_framework/core/common_di_container.py` (available core components)

**Remember:** The framework enforces consistency, testability, and maintainability. Following these patterns is not optional.
