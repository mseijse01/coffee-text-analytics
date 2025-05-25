# 📚 Coffee Text Analytics - API Documentation

*Generated on 2025-05-26 00:07:08*

## 📋 Table of Contents

- [config.cli](#configcli)
- [config.environments](#configenvironments)
- [config.settings](#configsettings)
- [config.validation](#configvalidation)
- [data.preprocessing](#datapreprocessing)
- [exceptions](#exceptions)
- [features.base](#featuresbase)
- [features.bert_extractor](#featuresbert_extractor)
- [features.feature_manager](#featuresfeature_manager)
- [features.sentiment_extractor](#featuressentiment_extractor)
- [features.tfidf_extractor](#featurestfidf_extractor)
- [features.topic_extractor](#featurestopic_extractor)
- [models.base](#modelsbase)
- [models.evaluator](#modelsevaluator)
- [models.mnir](#modelsmnir)
- [models.regressors](#modelsregressors)
- [utils.cache](#utilscache)
- [utils.cleaning](#utilscleaning)
- [utils.data_quality](#utilsdata_quality)
- [utils.doc_generator](#utilsdoc_generator)
- [utils.polars_utils](#utilspolars_utils)
- [utils.utils](#utilsutils)
- [visualization.plots](#visualizationplots)
- [visualization.visualize](#visualizationvisualize)

---

## 📦 config.cli

**File:** `/Users/seijas/Code/coffee-text-analytics/src/config/cli.py`

Configuration Management CLI

This module provides a command-line interface for managing configuration settings,
validating configurations, and switching between environments.

### 🔧 Functions

#### `check_dependencies_cli()`

Check and report on dependencies.

---

#### `compare_environments(env1: str, env2: str)`

Compare two environment configurations.

**Parameters:**

- `env1` (<class 'str'>)
- `env2` (<class 'str'>)

---

#### `create_config_parser()`

Create argument parser for configuration CLI.

---

#### `export_configuration(filename: str)`

Export configuration to JSON file.

**Parameters:**

- `filename` (<class 'str'>)

---

#### `get_config_value(key: str)`

Get a specific configuration value.

**Parameters:**

- `key` (<class 'str'>)

---

#### `list_environments()`

List all available environment configurations.

---

#### `main()`

Main CLI function.

---

#### `set_config_value(key: str, value: str)`

Set a specific configuration value.

**Parameters:**

- `key` (<class 'str'>)
- `value` (<class 'str'>)

---

#### `show_configuration_summary()`

Show detailed configuration summary.

---

#### `show_environment_config(environment: str)`

Show configuration for specific environment.

**Parameters:**

- `environment` (<class 'str'>)

---

#### `validate_configuration()`

Validate current configuration and print results.

---


## 📦 config.environments

**File:** `/Users/seijas/Code/coffee-text-analytics/src/config/environments.py`

Environment-specific Configuration Presets

This module provides predefined configuration presets for different environments
(development, production, testing) that can be used to override default settings.

### 🔢 Constants

#### `CICD_CONFIG`
- **Type:** `dict`
- **Value:** `{'data': {'min_rating': 75.0, 'max_missing_percentage': 80.0}, 'features': {'n_topics': 3, 'tfidf_max_features': 500, 'bert_batch_size': 2, 'bert_max_length': 64}, 'models': {'cv_folds': 2, 'test_size': 0.4, 'models_to_train': ['linear'], 'random_forest_params': {'n_estimators': 5, 'max_depth': 3, 'random_state': 42}}, 'logging': {'level': 'ERROR', 'console_handler': True, 'file_handler': False}, 'visualization': {'figure_width': 400, 'figure_height': 300, 'export_dpi': 72}}`

#### `DEMO_CONFIG`
- **Type:** `dict`
- **Value:** `{'data': {'min_rating': 80.0, 'max_missing_percentage': 50.0}, 'features': {'n_topics': 8, 'tfidf_max_features': 3000, 'bert_batch_size': 8}, 'models': {'cv_folds': 3, 'test_size': 0.2, 'random_forest_params': {'n_estimators': 50, 'max_depth': 8, 'random_state': 42}, 'xgboost_params': {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42}, 'models_to_train': ['linear', 'random_forest']}, 'logging': {'level': 'DEBUG', 'console_handler': True, 'file_handler': True, 'log_file': 'research_coffee_analytics.log'}, 'visualization': {'figure_width': 1000, 'figure_height': 700, 'export_dpi': 200}}`

#### `DEVELOPMENT_CONFIG`
- **Type:** `dict`
- **Value:** `{'data': {'min_rating': 80.0, 'max_missing_percentage': 50.0}, 'features': {'n_topics': 8, 'tfidf_max_features': 3000, 'bert_batch_size': 8}, 'models': {'cv_folds': 3, 'test_size': 0.2, 'random_forest_params': {'n_estimators': 50, 'max_depth': 8, 'random_state': 42}, 'xgboost_params': {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42}, 'models_to_train': ['linear', 'random_forest']}, 'logging': {'level': 'DEBUG', 'console_handler': True, 'file_handler': True, 'log_file': 'research_coffee_analytics.log'}, 'visualization': {'figure_width': 1000, 'figure_height': 700, 'export_dpi': 200}}`

#### `PRODUCTION_CONFIG`
- **Type:** `dict`
- **Value:** `{'data': {'min_rating': 85.0, 'max_missing_percentage': 30.0}, 'features': {'n_topics': 15, 'tfidf_max_features': 10000, 'bert_batch_size': 16}, 'models': {'cv_folds': 10, 'test_size': 0.15, 'random_forest_params': {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 3, 'min_samples_leaf': 1, 'random_state': 42}, 'xgboost_params': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42}, 'mnir_params': {'alpha': 0.05, 'max_iter': 2000, 'random_state': 42}}, 'logging': {'level': 'WARNING', 'console_handler': False, 'file_handler': True, 'log_file': 'production_coffee_analytics.log'}, 'visualization': {'figure_width': 1200, 'figure_height': 800, 'export_dpi': 300, 'export_width': 1600, 'export_height': 1200}}`

#### `RESEARCH_CONFIG`
- **Type:** `dict`
- **Value:** `{'data': {'min_rating': 80.0, 'max_missing_percentage': 50.0}, 'features': {'n_topics': 8, 'tfidf_max_features': 3000, 'bert_batch_size': 8}, 'models': {'cv_folds': 3, 'test_size': 0.2, 'random_forest_params': {'n_estimators': 50, 'max_depth': 8, 'random_state': 42}, 'xgboost_params': {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42}, 'models_to_train': ['linear', 'random_forest']}, 'logging': {'level': 'DEBUG', 'console_handler': True, 'file_handler': True, 'log_file': 'research_coffee_analytics.log'}, 'visualization': {'figure_width': 1000, 'figure_height': 700, 'export_dpi': 200}}`

#### `TESTING_CONFIG`
- **Type:** `dict`
- **Value:** `{'data': {'min_rating': 70.0, 'max_missing_percentage': 70.0}, 'features': {'n_topics': 5, 'tfidf_max_features': 1000, 'bert_batch_size': 4, 'bert_max_length': 128}, 'models': {'cv_folds': 3, 'test_size': 0.3, 'random_forest_params': {'n_estimators': 10, 'max_depth': 5, 'random_state': 42}, 'xgboost_params': {'n_estimators': 10, 'max_depth': 3, 'learning_rate': 0.3, 'random_state': 42}, 'mnir_params': {'alpha': 0.1, 'max_iter': 100, 'random_state': 42}}, 'logging': {'level': 'DEBUG', 'console_handler': True, 'file_handler': False, 'log_file': 'test_coffee_analytics.log'}, 'visualization': {'figure_width': 600, 'figure_height': 400, 'export_dpi': 100}}`

### 🔧 Functions

#### `apply_environment_config(config: config.settings.Config, environment: str) -> config.settings.Config`

Apply environment-specific configuration overrides.

**Parameters:**

- `config` (<class 'config.settings.Config'>)
  - Base configuration instance
- `environment` (<class 'str'>)
  - Environment name

**Returns:**

- Type: `<class 'config.settings.Config'>`
- Config: Updated configuration instance

---

#### `create_custom_environment(base_environment: str = 'development', overrides: Dict[str, Any] = None) -> Dict[str, Any]`

Create a custom environment configuration based on an existing one.

**Parameters:**

- `base_environment` (<class 'str'>) = development
  - Base environment to start from
- `overrides` (typing.Dict[str, typing.Any]) = None
  - Dictionary of configuration overrides

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Custom environment configuration dictionary

---

#### `get_environment_config(environment: str) -> Dict[str, Any]`

Get the configuration dictionary for a specific environment.

**Parameters:**

- `environment` (<class 'str'>)
  - Environment name

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary with environment-specific configuration

---

#### `list_available_environments() -> list`

Get list of available environment configurations.

**Returns:**

- Type: `<class 'list'>`

---


## 📦 config.settings

**File:** `/Users/seijas/Code/coffee-text-analytics/src/config/settings.py`

Coffee Text Analytics - Configuration Management

This module provides comprehensive configuration management for the coffee text analytics project.
It supports environment-specific settings, model parameters, data paths, and visualization settings.

Features:
- Environment-specific configurations (development, production, testing)
- Centralized path management
- Model hyperparameters
- Feature extraction parameters
- Visualization settings
- Logging configuration

### 🔢 Constants

#### `PATHS`
- **Type:** `dict`
- **Value:** `{'root': PosixPath('/Users/seijas/Code/coffee-text-analytics'), 'data': PosixPath('/Users/seijas/Code/coffee-text-analytics/data'), 'raw': PosixPath('/Users/seijas/Code/coffee-text-analytics/data/raw'), 'processed': PosixPath('/Users/seijas/Code/coffee-text-analytics/data/processed'), 'models': PosixPath('/Users/seijas/Code/coffee-text-analytics/models'), 'output': PosixPath('/Users/seijas/Code/coffee-text-analytics/output'), 'figures': PosixPath('/Users/seijas/Code/coffee-text-analytics/output/figures')}`

### 🏗️ Classes

#### 🏗️ `Config`

Main configuration class that combines all configuration components.

**Methods:**

#### `__init__(self, environment: str = None)`

Initialize configuration based on environment.

**Parameters:**

- `self`
- `environment` (<class 'str'>) = None
  - Environment name (development, production, testing)

---

#### `get_model_params(self, model_name: str) -> Dict[str, Any]`

Get hyperparameters for a specific model.

**Parameters:**

- `self`
- `model_name` (<class 'str'>)
  - Name of the model

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary of hyperparameters

---

#### `to_dict(self) -> Dict[str, Any]`

Convert configuration to dictionary for serialization.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, typing.Any]`

---

#### 🏗️ `DataConfig`

Configuration for data processing.

**Methods:**

#### `__init__(self, min_rating: float = 80.0, max_missing_percentage: float = 50.0, min_text_length: int = 10, max_text_length: int = 1000, standardize_countries: bool = True, standardize_prices: bool = True, target_currency: str = 'USD', target_unit: str = 'kg') -> None`

Initialize self.  See help(type(self)) for accurate signature.

**Parameters:**

- `self`
- `min_rating` (<class 'float'>) = 80.0
- `max_missing_percentage` (<class 'float'>) = 50.0
- `min_text_length` (<class 'int'>) = 10
- `max_text_length` (<class 'int'>) = 1000
- `standardize_countries` (<class 'bool'>) = True
- `standardize_prices` (<class 'bool'>) = True
- `target_currency` (<class 'str'>) = USD
- `target_unit` (<class 'str'>) = kg

**Returns:**

- Type: `None`

---

#### 🏗️ `FeatureConfig`

Configuration for feature extraction.

**Methods:**

#### `__init__(self, tfidf_max_features: int = 5000, tfidf_ngram_range: tuple = (1, 3), tfidf_min_df: int = 2, tfidf_max_df: float = 0.95, n_topics: int = 10, lda_random_state: int = 42, nmf_random_state: int = 42, bert_model_name: str = 'distilbert-base-uncased', bert_max_length: int = 512, bert_batch_size: int = 16, glove_dimensions: int = 300, glove_model_name: str = 'glove-wiki-gigaword-300', sentiment_model_name: str = 'distilbert-base-uncased-finetuned-sst-2-english', remove_stopwords: bool = True, lemmatize: bool = True, min_word_length: int = 2, max_word_length: int = 20) -> None`

Initialize self.  See help(type(self)) for accurate signature.

**Parameters:**

- `self`
- `tfidf_max_features` (<class 'int'>) = 5000
- `tfidf_ngram_range` (<class 'tuple'>) = (1, 3)
- `tfidf_min_df` (<class 'int'>) = 2
- `tfidf_max_df` (<class 'float'>) = 0.95
- `n_topics` (<class 'int'>) = 10
- `lda_random_state` (<class 'int'>) = 42
- `nmf_random_state` (<class 'int'>) = 42
- `bert_model_name` (<class 'str'>) = distilbert-base-uncased
- `bert_max_length` (<class 'int'>) = 512
- `bert_batch_size` (<class 'int'>) = 16
- `glove_dimensions` (<class 'int'>) = 300
- `glove_model_name` (<class 'str'>) = glove-wiki-gigaword-300
- `sentiment_model_name` (<class 'str'>) = distilbert-base-uncased-finetuned-sst-2-english
- `remove_stopwords` (<class 'bool'>) = True
- `lemmatize` (<class 'bool'>) = True
- `min_word_length` (<class 'int'>) = 2
- `max_word_length` (<class 'int'>) = 20

**Returns:**

- Type: `None`

---

#### 🏗️ `LoggingConfig`

Configuration for logging.

**Methods:**

#### `__init__(self, level: str = 'INFO', format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', file_handler: bool = True, console_handler: bool = True, log_file: str = 'coffee_analytics.log') -> None`

Initialize self.  See help(type(self)) for accurate signature.

**Parameters:**

- `self`
- `level` (<class 'str'>) = INFO
- `format` (<class 'str'>) = %(asctime)s - %(name)s - %(levelname)s - %(message)s
- `file_handler` (<class 'bool'>) = True
- `console_handler` (<class 'bool'>) = True
- `log_file` (<class 'str'>) = coffee_analytics.log

**Returns:**

- Type: `None`

---

#### 🏗️ `ModelConfig`

Configuration for machine learning models.

**Methods:**

#### `__init__(self, target_column: str = 'rating', text_columns: List[str] = <factory>, sensory_columns: List[str] = <factory>, models_to_train: List[str] = <factory>, random_forest_params: Dict[str, Any] = <factory>, xgboost_params: Dict[str, Any] = <factory>, linear_params: Dict[str, Any] = <factory>, mnir_params: Dict[str, Any] = <factory>, cv_folds: int = 5, test_size: float = 0.2, random_state: int = 42) -> None`

Initialize self.  See help(type(self)) for accurate signature.

**Parameters:**

- `self`
- `target_column` (<class 'str'>) = rating
- `text_columns` (typing.List[str]) = <factory>
- `sensory_columns` (typing.List[str]) = <factory>
- `models_to_train` (typing.List[str]) = <factory>
- `random_forest_params` (typing.Dict[str, typing.Any]) = <factory>
- `xgboost_params` (typing.Dict[str, typing.Any]) = <factory>
- `linear_params` (typing.Dict[str, typing.Any]) = <factory>
- `mnir_params` (typing.Dict[str, typing.Any]) = <factory>
- `cv_folds` (<class 'int'>) = 5
- `test_size` (<class 'float'>) = 0.2
- `random_state` (<class 'int'>) = 42

**Returns:**

- Type: `None`

---

#### 🏗️ `PathConfig`

Configuration for project paths.

**Methods:**

#### `__init__(self, root: pathlib.Path = <factory>, raw_data_file: str = 'coffee_clean.csv', processed_data_file: str = 'coffee_processed.csv', features_data_file: str = 'coffee_features.csv') -> None`

Initialize self.  See help(type(self)) for accurate signature.

**Parameters:**

- `self`
- `root` (<class 'pathlib.Path'>) = <factory>
- `raw_data_file` (<class 'str'>) = coffee_clean.csv
- `processed_data_file` (<class 'str'>) = coffee_processed.csv
- `features_data_file` (<class 'str'>) = coffee_features.csv

**Returns:**

- Type: `None`

---

#### `create_directories(self) -> None`

Create all project directories if they don't exist.

**Parameters:**

- `self`

**Returns:**

- Type: `None`

---

#### `get_features_data_path(self) -> pathlib.Path`

Get the full path to the features data file.

**Parameters:**

- `self`

**Returns:**

- Type: `<class 'pathlib.Path'>`

---

#### `get_processed_data_path(self) -> pathlib.Path`

Get the full path to the processed data file.

**Parameters:**

- `self`

**Returns:**

- Type: `<class 'pathlib.Path'>`

---

#### `get_raw_data_path(self) -> pathlib.Path`

Get the full path to the raw data file.

**Parameters:**

- `self`

**Returns:**

- Type: `<class 'pathlib.Path'>`

---

#### 🏗️ `VisualizationConfig`

Configuration for visualizations and plots.

**Methods:**

#### `__init__(self, figure_width: int = 800, figure_height: int = 600, template: str = 'plotly_white', color_palette: List[str] = <factory>, font_family: str = 'Arial', font_size: int = 12, title_font_size: int = 16, export_format: str = 'png', export_dpi: int = 300, export_width: int = 1200, export_height: int = 800) -> None`

Initialize self.  See help(type(self)) for accurate signature.

**Parameters:**

- `self`
- `figure_width` (<class 'int'>) = 800
- `figure_height` (<class 'int'>) = 600
- `template` (<class 'str'>) = plotly_white
- `color_palette` (typing.List[str]) = <factory>
- `font_family` (<class 'str'>) = Arial
- `font_size` (<class 'int'>) = 12
- `title_font_size` (<class 'int'>) = 16
- `export_format` (<class 'str'>) = png
- `export_dpi` (<class 'int'>) = 300
- `export_width` (<class 'int'>) = 1200
- `export_height` (<class 'int'>) = 800

**Returns:**

- Type: `None`

---


## 📦 config.validation

**File:** `/Users/seijas/Code/coffee-text-analytics/src/config/validation.py`

Configuration Validation Utilities

This module provides validation functions for configuration settings to ensure
they are consistent, valid, and compatible with the project requirements.

### 🔧 Functions

#### `check_dependencies() -> Tuple[bool, List[str]]`

Check if required dependencies are available.

**Returns:**

- Type: `typing.Tuple[bool, typing.List[str]]`
- Tuple of (all_available, missing_packages)

---

#### `get_config_summary(config: config.settings.Config) -> Dict[str, Any]`

Get a summary of the current configuration.

**Parameters:**

- `config` (<class 'config.settings.Config'>)
  - Configuration instance

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary with configuration summary

---

#### `print_config_summary(config: config.settings.Config)`

Print a formatted configuration summary.

**Parameters:**

- `config` (<class 'config.settings.Config'>)

---

#### `validate_config(config: config.settings.Config, raise_on_error: bool = False) -> bool`

Validate configuration and optionally raise on errors.

**Parameters:**

- `config` (<class 'config.settings.Config'>)
  - Configuration instance to validate
- `raise_on_error` (<class 'bool'>) = False
  - Whether to raise exception on validation errors

**Returns:**

- Type: `bool: True if configuration is valid`

---

### 🏗️ Classes

#### 🏗️ `ConfigValidationError`

**Inherits from:** `Exception`

Custom exception for configuration validation errors.

#### 🏗️ `ConfigValidator`

Validates configuration settings and provides recommendations.

**Methods:**

#### `__init__(self, config: config.settings.Config)`

Initialize validator with configuration instance.

**Parameters:**

- `self`
- `config` (<class 'config.settings.Config'>)
  - Configuration instance to validate

---

#### `validate_all(self) -> Tuple[bool, List[str], List[str]]`

Validate all configuration components.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Tuple[bool, typing.List[str], typing.List[str]]`
- Tuple of (is_valid, warnings, errors)

---


## 📦 data.loader

**File:** `Unknown`



⚠️ **Import Error:**
```
Failed to import module: attempted relative import beyond top-level package
```


## 📦 data.preprocessing

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/preprocessing.py`

Text preprocessing utilities for coffee review data.

### 🔢 Constants

#### `NLTK_DATA_DOWNLOADED`
- **Type:** `bool`
- **Value:** `False`

### 🔧 Functions

#### `clean_text(text: str, remove_punctuation: bool = True) -> str`

Clean text by removing HTML tags, URLs, and optionally punctuation.

**Parameters:**

- `text` (<class 'str'>)
- `remove_punctuation` (<class 'bool'>) = True

**Returns:**

- Type: `str: Cleaned text`

**Examples:**

```python
>>> clean_text("Great coffee! Very smooth.")
"Great coffee Very smooth"
>>> clean_text("Great coffee! Very smooth.", remove_punctuation=False)
"Great coffee! Very smooth."
```

---

#### `ensure_nltk_data()`

Ensure necessary NLTK data is downloaded.

---

#### `extract_country_info(location: str) -> str`

Extract country name from location string.

Prioritizes known coffee-producing countries that appear at string start.

**Parameters:**

- `location` (<class 'str'>)

**Returns:**

- Type: `str: Extracted country name (e.g., "Ethiopia")`

**Examples:**

```python
>>> extract_country_info("Ethiopia Yirgacheffe")
"Ethiopia"
>>> extract_country_info("Colombia Huila")
"Colombia"
>>> extract_country_info("Jamaica Blue Mountain")
"Jamaica"
```

---

#### `lemmatize_text(tokens)`

Lemmatize tokens to their base form.

**Parameters:**

- `tokens`

**Returns:**

- list: Lemmatized tokens

---

#### `load_csv_for_preprocessing(file_path: str) -> pandas.core.frame.DataFrame`

Load CSV data for text preprocessing operations.

This function loads CSV data specifically for text preprocessing operations.
Uses pandas for easier text manipulation and sklearn compatibility.

**Parameters:**

- `file_path` (<class 'str'>)

**Returns:**

- Type: `<class 'pandas.core.frame.DataFrame'>`
- pd.DataFrame: Data optimized for text preprocessing operations

Note:
Returns pandas DataFrame (not Polars) for easier text processing.
Use convert_pandas_to_polars() if you need Polars format afterward.

---

#### `merge_text_columns(df, columns, output_col='merged_text')`

Merge multiple text columns into one combined text column.

**Parameters:**

- `df`
- `columns`
- `output_col` = merged_text

**Returns:**

- pd.DataFrame: DataFrame with merged text column

---

#### `preprocess_text(text, remove_stop=True)`

Apply full preprocessing pipeline to text.

**Parameters:**

- `text`
- `remove_stop` = True

**Returns:**

- str: Preprocessed text

---

#### `process_raw_data(input_file, output_file, text_columns=None)`

Process raw coffee review data and save processed version.

**Parameters:**

- `input_file`
- `output_file`
- `text_columns` = None

---

#### `remove_stopwords(tokens, keep_stopwords=False)`

Remove common stopwords from token list.

**Parameters:**

- `tokens`
- `keep_stopwords` = False

**Returns:**

- list: Filtered tokens

---

#### `standardize_prices(df, price_col='price')`

Standardize coffee prices to USD per kilogram.

**Parameters:**

- `df`
- `price_col` = price

**Returns:**

- pd.DataFrame: DataFrame with standardized prices

---

#### `tokenize_text(text)`

Tokenize text into individual words.

**Parameters:**

- `text`

**Returns:**

- list: List of tokens

---


## 📦 exceptions

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../exceptions.py`

Coffee Text Analytics - Centralized Exception Handling

This module provides a comprehensive exception hierarchy for the coffee text analytics project.
All custom exceptions inherit from CoffeeAnalyticsError for consistent error handling.

### 🔧 Functions

#### `handle_exception(exception: Exception, context: Optional[Dict[str, Any]] = None, reraise_as: Optional[type] = None, message: Optional[str] = None) -> None`

Utility function to handle exceptions consistently.

**Parameters:**

- `exception` (<class 'Exception'>)
  - The original exception
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context for debugging
- `reraise_as` (typing.Optional[type]) = None
  - Optional exception class to reraise as
- `message` (typing.Optional[str]) = None
  - Optional custom message

**Returns:**

- Type: `None`

---

#### `require_dependency(module_name: str, import_name: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> None`

Require that a dependency is available.

**Parameters:**

- `module_name` (<class 'str'>)
  - Name of the module to import
- `import_name` (typing.Optional[str]) = None
  - Optional specific import name
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context for debugging

**Returns:**

- Type: `None`

---

#### `validate_directory_exists(dir_path: str, context: Optional[Dict[str, Any]] = None) -> None`

Validate that a directory exists.

**Parameters:**

- `dir_path` (<class 'str'>)
  - Path to the directory
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context for debugging

**Returns:**

- Type: `None`

---

#### `validate_file_exists(file_path: str, context: Optional[Dict[str, Any]] = None) -> None`

Validate that a file exists.

**Parameters:**

- `file_path` (<class 'str'>)
  - Path to the file
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context for debugging

**Returns:**

- Type: `None`

---

#### `validate_not_empty(value: Any, name: str, context: Optional[Dict[str, Any]] = None) -> None`

Validate that a value is not empty.

**Parameters:**

- `value` (typing.Any)
  - Value to validate
- `name` (<class 'str'>)
  - Name of the value for error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context for debugging

**Returns:**

- Type: `None`

---

#### `validate_not_none(value: Any, name: str, context: Optional[Dict[str, Any]] = None) -> None`

Validate that a value is not None.

**Parameters:**

- `value` (typing.Any)
  - Value to validate
- `name` (<class 'str'>)
  - Name of the value for error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context for debugging

**Returns:**

- Type: `None`

---

### 🏗️ Classes

#### 🏗️ `BertExtractionError`

**Inherits from:** `FeatureExtractionError`

Raised when BERT extraction fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `CoffeeAnalyticsError`

**Inherits from:** `Exception`

Base exception for all coffee analytics project errors.

Provides consistent error handling with optional context and logging.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ConfigError`

**Inherits from:** `CoffeeAnalyticsError`

Base exception for configuration errors.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ConfigLoadError`

**Inherits from:** `ConfigError`

Raised when configuration loading fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ConfigValidationError`

**Inherits from:** `ConfigError`

Raised when configuration validation fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `DataError`

**Inherits from:** `CoffeeAnalyticsError`

Base exception for data-related errors.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `DataLoadingError`

**Inherits from:** `DataError`

Raised when data loading fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `DataPreprocessingError`

**Inherits from:** `DataError`

Raised when data preprocessing fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `DataQualityError`

**Inherits from:** `DataError`

Raised when data quality issues are detected.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `DataValidationError`

**Inherits from:** `DataError`

Raised when data validation fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `DependencyError`

**Inherits from:** `CoffeeAnalyticsError`

Base exception for dependency-related errors.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `EnvironmentConfigError`

**Inherits from:** `ConfigError`

Raised when environment-specific configuration fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ExtractorConfigError`

**Inherits from:** `FeatureExtractionError`

Raised when there's an issue with extractor configuration.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ExtractorNotFittedError`

**Inherits from:** `FeatureExtractionError`

Raised when trying to use an unfitted extractor.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `FeatureExtractionError`

**Inherits from:** `CoffeeAnalyticsError`

Base exception for feature extraction errors.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `FileError`

**Inherits from:** `CoffeeAnalyticsError`

Base exception for file operation errors.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `FileLoadError`

**Inherits from:** `FileError`

Raised when file loading fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `FileNotFoundError`

**Inherits from:** `FileError`

Raised when a required file is not found.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `FilePermissionError`

**Inherits from:** `FileError`

Raised when file permission issues occur.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `FileSaveError`

**Inherits from:** `FileError`

Raised when file saving fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `GloveExtractionError`

**Inherits from:** `FeatureExtractionError`

Raised when GloVe embedding extraction fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `IncompatibleDependencyError`

**Inherits from:** `DependencyError`

Raised when dependency versions are incompatible.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `MNIRError`

**Inherits from:** `ModelError`

Raised when MNIR-specific operations fail.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `MissingDependencyError`

**Inherits from:** `DependencyError`

Raised when a required dependency is missing.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ModelConfigError`

**Inherits from:** `ModelError`

Raised when there's an issue with model configuration.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ModelError`

**Inherits from:** `CoffeeAnalyticsError`

Base exception for model-related errors.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ModelEvaluationError`

**Inherits from:** `ModelError`

Raised when model evaluation fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ModelLoadError`

**Inherits from:** `ModelError`

Raised when model loading fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ModelNotFittedError`

**Inherits from:** `ModelError`

Raised when trying to use an unfitted model.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ModelSaveError`

**Inherits from:** `ModelError`

Raised when model saving fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `ModelTrainingError`

**Inherits from:** `ModelError`

Raised when model training fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `PlotCreationError`

**Inherits from:** `VisualizationError`

Raised when plot creation fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `PlotSaveError`

**Inherits from:** `VisualizationError`

Raised when plot saving fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `SentimentExtractionError`

**Inherits from:** `FeatureExtractionError`

Raised when sentiment analysis fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `TfidfExtractionError`

**Inherits from:** `FeatureExtractionError`

Raised when TF-IDF extraction fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `TopicExtractionError`

**Inherits from:** `FeatureExtractionError`

Raised when topic modeling fails.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---

#### 🏗️ `VisualizationError`

**Inherits from:** `CoffeeAnalyticsError`

Base exception for visualization errors.

**Methods:**

#### `__init__(self, message: str, context: Optional[Dict[str, Any]] = None, log_level: int = 40)`

Initialize the exception with message and optional context.

**Parameters:**

- `self`
- `message` (<class 'str'>)
  - Error message
- `context` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional context dictionary for debugging
- `log_level` (<class 'int'>) = 40
  - Logging level for this error

---


## 📦 features.base

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/base.py`

Base classes for feature extraction components.

This module provides abstract base classes that define the interface for all
feature extractors in the coffee text analytics project.

### 🏗️ Classes

#### 🏗️ `BaseExtractor`

**Inherits from:** `ABC`

Abstract base class for all feature extractors.

Defines the common interface and behavior for feature extraction components.
All extractors must implement fit() and extract_features() methods.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize the extractor with configuration.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional configuration dictionary

---

#### `extract_features(self, texts: List[str]) -> polars.dataframe.frame.DataFrame`

Extract features from texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of texts to process

**Returns:**

- Type: `Polars DataFrame with extracted features`

---

#### `fit(self, texts: List[str]) -> 'BaseExtractor'`

Fit the extractor to training texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of training texts

**Returns:**

- Type: `Self for method chaining`

---

#### `get_feature_names(self) -> List[str]`

Get the names of extracted features.

**Parameters:**

- `self`

**Returns:**

- Type: `List of feature names`

---

#### 🏗️ `BaseSparseExtractor`

**Inherits from:** `BaseExtractor`

Base class for extractors that produce sparse features.

Examples: TF-IDF, bag-of-words

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize the extractor with configuration.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional configuration dictionary

---

#### `extract_features(self, texts: List[str]) -> polars.dataframe.frame.DataFrame`

Extract features from texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of texts to process

**Returns:**

- Type: `Polars DataFrame with extracted features`

---

#### `fit(self, texts: List[str]) -> 'BaseExtractor'`

Fit the extractor to training texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of training texts

**Returns:**

- Type: `Self for method chaining`

---

#### `get_feature_names(self) -> List[str]`

Get the names of extracted features.

**Parameters:**

- `self`

**Returns:**

- Type: `List of feature names`

---

#### 🏗️ `BaseTopicExtractor`

**Inherits from:** `BaseExtractor`

Base class for topic modeling extractors.

Examples: LDA, NMF topic modeling

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize the extractor with configuration.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional configuration dictionary

---

#### `extract_features(self, texts: List[str]) -> polars.dataframe.frame.DataFrame`

Extract features from texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of texts to process

**Returns:**

- Type: `Polars DataFrame with extracted features`

---

#### `fit(self, texts: List[str]) -> 'BaseExtractor'`

Fit the extractor to training texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of training texts

**Returns:**

- Type: `Self for method chaining`

---

#### `get_feature_names(self) -> List[str]`

Get the names of extracted features.

**Parameters:**

- `self`

**Returns:**

- Type: `List of feature names`

---

#### 🏗️ `BaseVectorExtractor`

**Inherits from:** `BaseExtractor`

Base class for extractors that produce dense vector features.

Examples: BERT embeddings, GloVe embeddings

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize the extractor with configuration.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional configuration dictionary

---

#### `extract_features(self, texts: List[str]) -> polars.dataframe.frame.DataFrame`

Extract features from texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of texts to process

**Returns:**

- Type: `Polars DataFrame with extracted features`

---

#### `fit(self, texts: List[str]) -> 'BaseExtractor'`

Fit the extractor to training texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of training texts

**Returns:**

- Type: `Self for method chaining`

---

#### `get_feature_names(self) -> List[str]`

Get the names of extracted features.

**Parameters:**

- `self`

**Returns:**

- Type: `List of feature names`

---


## 📦 features.bert_extractor

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/bert_extractor.py`

BERT embeddings extractor for coffee review text analysis.

This module implements BERT embeddings extraction following the thesis methodology:
- DistilBERT model for 768-dimensional embeddings
- Efficient batch processing
- Polars DataFrame output for modern data processing

### 🔢 Constants

#### `TRANSFORMERS_AVAILABLE`
- **Type:** `bool`
- **Value:** `False`

### 🏗️ Classes

#### 🏗️ `BertExtractor`

**Inherits from:** `BaseVectorExtractor`

BERT embeddings extractor following thesis methodology.

From thesis: "BERT embeddings using DistilBERT (768-dimensional vectors)"

This extractor produces dense semantic representations using DistilBERT
and outputs results as Polars DataFrames for efficient processing.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize BERT extractor.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with parameters:

---

#### `compute_similarity(self, text1: str, text2: str) -> float`

Compute cosine similarity between two texts using BERT embeddings.

**Parameters:**

- `self`
- `text1` (<class 'str'>)
  - First text
- `text2` (<class 'str'>)
  - Second text

**Returns:**

- Type: `<class 'float'>`
- Cosine similarity score

---

#### `encode_single_text(self, text: str) -> numpy.ndarray`

Encode a single text to BERT embedding.

**Parameters:**

- `self`
- `text` (<class 'str'>)
  - Text to encode

**Returns:**

- Type: `<class 'numpy.ndarray'>`
- BERT embedding array

---

#### `extract_features(self, texts: List[str]) -> polars.dataframe.frame.DataFrame`

Extract BERT embeddings from texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of texts to process

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- Polars DataFrame with BERT embedding features (768 columns)

---

#### `fit(self, texts: List[str]) -> 'BertExtractor'`

Fit the BERT extractor (no training needed for pre-trained models).

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of training texts (not used for BERT)

**Returns:**

- Type: `BertExtractor`
- Self for method chaining

---

#### `get_feature_count(self) -> int`

Get the number of BERT features.

**Parameters:**

- `self`

**Returns:**

- Type: `<class 'int'>`

---

#### `get_feature_names(self) -> List[str]`

Get BERT feature names.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.List[str]`

---

#### `get_model_info(self) -> Dict[str, Any]`

Get information about the BERT model.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary with model information

---

#### `get_vector_dimension(self) -> int`

Get the BERT embedding dimension.

**Parameters:**

- `self`

**Returns:**

- Type: `<class 'int'>`

---


## 📦 features.feature_manager

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/feature_manager.py`

Unified feature extraction manager for coffee review text analysis.

This module orchestrates all feature extractors and provides a single interface
for extracting comprehensive features following the thesis methodology.

### 🔢 Constants

#### `GENSIM_AVAILABLE`
- **Type:** `bool`
- **Value:** `True`

### 🏗️ Classes

#### 🏗️ `CoffeeFeatureManager`

Unified feature extraction manager for coffee reviews.

This manager orchestrates all feature extractors and provides a single
interface for comprehensive feature extraction following thesis methodology.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize feature manager.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with extractor settings

---

#### `extract_all_features(self, df: polars.dataframe.frame.DataFrame, text_columns: List[str] = ['desc_1', 'desc_2', 'desc_3']) -> polars.dataframe.frame.DataFrame`

Extract features from multiple text columns and combine with original data.

**Parameters:**

- `self`
- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - Input DataFrame with text columns
- `text_columns` (typing.List[str]) = ['desc_1', 'desc_2', 'desc_3']
  - List of text column names to process

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- DataFrame with original data plus extracted features

---

#### `extract_features(self, texts: List[str]) -> polars.dataframe.frame.DataFrame`

Extract all features from texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of texts to process

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- Polars DataFrame with all extracted features

---

#### `fit(self, texts: List[str]) -> 'CoffeeFeatureManager'`

Fit all extractors to the training texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of training texts

**Returns:**

- Type: `CoffeeFeatureManager`
- Self for method chaining

---

#### `get_extractor_info(self) -> Dict[str, Dict[str, Any]]`

Get information about all extractors.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, typing.Dict[str, typing.Any]]`
- Dictionary with extractor information

---

#### `get_feature_counts(self) -> Dict[str, int]`

Get feature counts from all extractors.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, int]`
- Dictionary mapping extractor names to their feature counts

---

#### `get_feature_names(self) -> Dict[str, List[str]]`

Get feature names from all extractors.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, typing.List[str]]`
- Dictionary mapping extractor names to their feature names

---

#### `get_total_feature_count(self) -> int`

Get total number of features across all extractors.

**Parameters:**

- `self`

**Returns:**

- Type: `<class 'int'>`

---

#### `load_extractors(self, models_dir: str = 'models') -> 'CoffeeFeatureManager'`

Load previously fitted extractors.

**Parameters:**

- `self`
- `models_dir` (<class 'str'>) = models
  - Directory containing saved models

**Returns:**

- Type: `CoffeeFeatureManager`
- Self for method chaining

---

#### `print_summary(self) -> None`

Print a summary of the feature manager.

**Parameters:**

- `self`

**Returns:**

- Type: `None`

---

#### `save_extractors(self, models_dir: str = 'models') -> None`

Save all fitted extractors.

**Parameters:**

- `self`
- `models_dir` (<class 'str'>) = models
  - Directory to save models

**Returns:**

- Type: `None`

---

#### 🏗️ `GloVeExtractor`

**Inherits from:** `BaseExtractor`

GloVe embeddings extractor for word-level semantics.

From thesis: "GloVe embeddings (300-dimensional) using pre-trained vectors"

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize GloVe extractor.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None

---

#### `extract_features(self, texts: List[str]) -> polars.dataframe.frame.DataFrame`

Extract GloVe embeddings from texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`

---

#### `fit(self, texts: List[str]) -> 'GloVeExtractor'`

Fit GloVe extractor (pre-trained model).

**Parameters:**

- `self`
- `texts` (typing.List[str])

**Returns:**

- Type: `GloVeExtractor`

---

#### `get_feature_count(self) -> int`

Get number of GloVe features.

**Parameters:**

- `self`

**Returns:**

- Type: `<class 'int'>`

---

#### `get_feature_names(self) -> List[str]`

Get GloVe feature names.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.List[str]`

---


## 📦 features.sentiment_extractor

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/sentiment_extractor.py`

Sentiment analysis extractor for coffee review text analysis.

This module implements sentiment analysis following the thesis methodology:
- DistilBERT-based sentiment classification
- Positive/negative probability scores
- Polars DataFrame output for efficient processing

### 🔢 Constants

#### `TRANSFORMERS_AVAILABLE`
- **Type:** `bool`
- **Value:** `True`

### 🏗️ Classes

#### 🏗️ `SentimentExtractor`

**Inherits from:** `BaseExtractor`

Sentiment analysis extractor following thesis methodology.

From thesis: "Sentiment scores (positive/negative probabilities)"

This extractor analyzes sentiment in coffee reviews using DistilBERT
and outputs results as Polars DataFrames for efficient processing.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize sentiment extractor.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with parameters:

---

#### `analyze_single_text(self, text: str) -> Dict[str, float]`

Analyze sentiment for a single text.

**Parameters:**

- `self`
- `text` (<class 'str'>)
  - Text to analyze

**Returns:**

- Type: `typing.Dict[str, float]`
- Dictionary with positive and negative sentiment scores

---

#### `extract_features(self, texts: List[str]) -> polars.dataframe.frame.DataFrame`

Extract sentiment features from texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of texts to process

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- Polars DataFrame with sentiment features (positive/negative probabilities)

---

#### `fit(self, texts: List[str]) -> 'SentimentExtractor'`

Fit the sentiment extractor (no training needed for pre-trained models).

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of training texts (not used for sentiment analysis)

**Returns:**

- Type: `SentimentExtractor`
- Self for method chaining

---

#### `get_dominant_sentiment(self, text: str) -> str`

Get the dominant sentiment (positive/negative) for a text.

**Parameters:**

- `self`
- `text` (<class 'str'>)
  - Text to analyze

**Returns:**

- Type: `<class 'str'>`
- 'positive' or 'negative'

---

#### `get_feature_count(self) -> int`

Get the number of sentiment features.

**Parameters:**

- `self`

**Returns:**

- Type: `<class 'int'>`

---

#### `get_feature_names(self) -> List[str]`

Get sentiment feature names.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.List[str]`

---

#### `get_model_info(self) -> Dict[str, Any]`

Get information about the sentiment model.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary with model information

---

#### `get_sentiment_confidence(self, text: str) -> float`

Get the confidence score for the dominant sentiment.

**Parameters:**

- `self`
- `text` (<class 'str'>)
  - Text to analyze

**Returns:**

- Type: `<class 'float'>`
- Confidence score (0-1)

---

#### `get_sentiment_statistics(self, texts: List[str]) -> Dict[str, Any]`

Get sentiment statistics for a collection of texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of texts to analyze

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary with sentiment statistics

---


## 📦 features.tfidf_extractor

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/tfidf_extractor.py`

TF-IDF feature extractor for coffee review text analysis.

This module provides TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction
following the thesis methodology with robust error handling.

### 🏗️ Classes

#### 🏗️ `TfidfExtractor`

**Inherits from:** `BaseSparseExtractor`

TF-IDF feature extractor with robust error handling.

Extracts TF-IDF features from text documents using scikit-learn's TfidfVectorizer
with comprehensive error handling and validation.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize TF-IDF extractor with configuration.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with TF-IDF parameters

---

#### `extract_features(self, texts: List[str]) -> polars.dataframe.frame.DataFrame`

Extract TF-IDF features from texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of texts to process

**Returns:**

- Type: `Polars DataFrame with TF-IDF features`

---

#### `fit(self, texts: List[str]) -> 'TfidfExtractor'`

Fit the TF-IDF vectorizer to training texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of training texts

**Returns:**

- Type: `Self for method chaining`

---

#### `get_feature_names(self) -> List[str]`

Get the names of extracted features.

**Parameters:**

- `self`

**Returns:**

- Type: `List of feature names`

---

#### `get_feature_statistics(self) -> Dict[str, Any]`

Get statistics about the extracted features.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary with feature statistics`

---

#### `get_vocabulary(self) -> Dict[str, int]`

Get the vocabulary mapping.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary mapping terms to indices`

---

#### `load_extractor(self, models_dir: Optional[str] = None) -> 'TfidfExtractor'`

Load a previously fitted TF-IDF extractor.

**Parameters:**

- `self`
- `models_dir` (typing.Optional[str]) = None
  - Directory containing the extractor. If None, uses config directory.

**Returns:**

- Type: `Self for method chaining`

---

#### `save_extractor(self, models_dir: Optional[str] = None) -> None`

Save the fitted TF-IDF extractor.

**Parameters:**

- `self`
- `models_dir` (typing.Optional[str]) = None
  - Directory to save the extractor. If None, uses config directory.

**Returns:**

- Type: `None`

---


## 📦 features.topic_extractor

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/topic_extractor.py`

Topic modeling extractor for coffee review text analysis.

This module implements topic modeling following the thesis methodology:
- LDA (Latent Dirichlet Allocation) for probabilistic topic modeling
- NMF (Non-negative Matrix Factorization) for linear topic modeling
- 10 topics per model as specified in thesis
- Polars DataFrame output for efficient processing

### 🏗️ Classes

#### 🏗️ `TopicExtractor`

**Inherits from:** `BaseTopicExtractor`

Topic modeling extractor following thesis methodology.

From thesis: "LDA and NMF topic modeling (10 topics each)"

This extractor discovers latent topics in coffee reviews using both
LDA and NMF algorithms and outputs results as Polars DataFrames.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize topic extractor.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with parameters:

---

#### `extract_features(self, texts: List[str]) -> polars.dataframe.frame.DataFrame`

Extract topic features from texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of texts to process

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- Polars DataFrame with topic distribution features

---

#### `fit(self, texts: List[str]) -> 'TopicExtractor'`

Fit topic models to the training texts.

**Parameters:**

- `self`
- `texts` (typing.List[str])
  - List of training texts

**Returns:**

- Type: `TopicExtractor`
- Self for method chaining

---

#### `get_feature_names(self) -> List[str]`

Get topic feature names.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.List[str]`

---

#### `get_lda_topics(self) -> List[List[Tuple[str, float]]]`

Get LDA topics specifically.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.List[typing.List[typing.Tuple[str, float]]]`

---

#### `get_model_info(self) -> Dict[str, Any]`

Get information about the topic models.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary with model information

---

#### `get_nmf_topics(self) -> List[List[Tuple[str, float]]]`

Get NMF topics specifically.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.List[typing.List[typing.Tuple[str, float]]]`

---

#### `get_topic_count(self) -> int`

Get the total number of topics.

**Parameters:**

- `self`

**Returns:**

- Type: `<class 'int'>`

---

#### `get_topic_distribution(self, text: str) -> Dict[str, numpy.ndarray]`

Get topic distribution for a single text.

**Parameters:**

- `self`
- `text` (<class 'str'>)
  - Text to analyze

**Returns:**

- Type: `typing.Dict[str, numpy.ndarray]`
- Dictionary with topic distributions for each algorithm

---

#### `get_topics(self) -> List[List[Tuple[str, float]]]`

Get discovered topics with top words and weights.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.List[typing.List[typing.Tuple[str, float]]]`

---

#### `load_models(self, models_dir: Optional[str] = None) -> 'TopicExtractor'`

Load previously fitted models.

**Parameters:**

- `self`
- `models_dir` (typing.Optional[str]) = None
  - Directory containing the models. If None, uses config directory.

**Returns:**

- Type: `TopicExtractor`
- Self for method chaining

---

#### `print_topics(self, n_words: int = 10) -> None`

Print discovered topics in a readable format.

**Parameters:**

- `self`
- `n_words` (<class 'int'>) = 10
  - Number of top words to display per topic

**Returns:**

- Type: `None`

---


## 📦 models.base

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/base.py`

Base classes for machine learning models.

This module provides abstract base classes that define the interface for all
models in the coffee text analytics project.

### 🏗️ Classes

#### 🏗️ `BaseClassifier`

**Inherits from:** `BaseModel`

Base class for classification models.

**Properties:**

- `_repr_html_` (readable)
  - HTML representation of estimator.

This is redundant with the logic of `_repr_mimebundle_`. The latter
should be favorted in the long term, `_repr_html_` is only
implemented for consumers who do not interpret `_repr_mimbundle_`.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize the model with configuration.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional configuration dictionary

---

#### `fit(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> 'BaseModel'`

Fit the model to training data.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Training features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - Training targets

**Returns:**

- Type: `Self for method chaining`

---

#### `get_feature_importance(self) -> Dict[str, float]`

Get feature importance scores.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary mapping feature names to importance scores`

---

#### `get_metadata_routing(self)`

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

Returns
-------
routing : MetadataRequest
A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
routing information.

**Parameters:**

- `self`

---

#### `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
If True, will return the parameters for this estimator and
contained subobjects that are estimators.

Returns
-------
params : dict
Parameter names mapped to their values.

**Parameters:**

- `self`
- `deep` = True

---

#### `get_training_metrics(self) -> Dict[str, float]`

Get training metrics.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary of training metrics`

---

#### `predict(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame]) -> numpy.ndarray`

Make predictions using the fitted model.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Features to predict on

**Returns:**

- Type: `Predictions array`

---

#### `predict_proba(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame]) -> numpy.ndarray`

Predict class probabilities.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Features to predict on

**Returns:**

- Type: `Probability predictions`

---

#### `score(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> float`

Calculate accuracy score.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Test features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - True targets

**Returns:**

- Type: `Accuracy score`

---

#### `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

Parameters
----------
**params : dict
Estimator parameters.

Returns
-------
self : estimator instance
Estimator instance.

**Parameters:**

- `self`
- `params`

---

#### 🏗️ `BaseEnsembleModel`

**Inherits from:** `BaseModel`

Base class for ensemble models.

**Properties:**

- `_repr_html_` (readable)
  - HTML representation of estimator.

This is redundant with the logic of `_repr_mimebundle_`. The latter
should be favorted in the long term, `_repr_html_` is only
implemented for consumers who do not interpret `_repr_mimbundle_`.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize the model with configuration.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional configuration dictionary

---

#### `fit(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> 'BaseModel'`

Fit the model to training data.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Training features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - Training targets

**Returns:**

- Type: `Self for method chaining`

---

#### `get_base_models(self) -> List[models.base.BaseModel]`

Get the base models in the ensemble.

**Parameters:**

- `self`

**Returns:**

- Type: `List of base models`

---

#### `get_feature_importance(self) -> Dict[str, float]`

Get feature importance scores.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary mapping feature names to importance scores`

---

#### `get_metadata_routing(self)`

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

Returns
-------
routing : MetadataRequest
A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
routing information.

**Parameters:**

- `self`

---

#### `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
If True, will return the parameters for this estimator and
contained subobjects that are estimators.

Returns
-------
params : dict
Parameter names mapped to their values.

**Parameters:**

- `self`
- `deep` = True

---

#### `get_training_metrics(self) -> Dict[str, float]`

Get training metrics.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary of training metrics`

---

#### `predict(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame]) -> numpy.ndarray`

Make predictions using the fitted model.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Features to predict on

**Returns:**

- Type: `Predictions array`

---

#### `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

Parameters
----------
**params : dict
Estimator parameters.

Returns
-------
self : estimator instance
Estimator instance.

**Parameters:**

- `self`
- `params`

---

#### 🏗️ `BaseEvaluator`

**Inherits from:** `ABC`

Abstract base class for model evaluators.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize the evaluator with configuration.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional configuration dictionary

---

#### `compare_models(self, models: Dict[str, models.base.BaseModel], X_test: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y_test: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> Dict[str, Any]`

Compare multiple models.

**Parameters:**

- `self`
- `models` (typing.Dict[str, models.base.BaseModel])
  - Dictionary of models to compare
- `X_test` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Test features
- `y_test` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - Test targets

**Returns:**

- Type: `Dictionary of comparison results`

---

#### `evaluate_model(self, model: models.base.BaseModel, X_test: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y_test: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> Dict[str, float]`

Evaluate a single model.

**Parameters:**

- `self`
- `model` (<class 'models.base.BaseModel'>)
  - Model to evaluate
- `X_test` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Test features
- `y_test` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - Test targets

**Returns:**

- Type: `Dictionary of evaluation metrics`

---

#### 🏗️ `BaseModel`

**Inherits from:** `ABC`, `BaseEstimator`

Abstract base class for all machine learning models.

Provides common interface and functionality for model training, prediction,
and evaluation in the coffee text analytics project.

**Properties:**

- `_repr_html_` (readable)
  - HTML representation of estimator.

This is redundant with the logic of `_repr_mimebundle_`. The latter
should be favorted in the long term, `_repr_html_` is only
implemented for consumers who do not interpret `_repr_mimbundle_`.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize the model with configuration.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional configuration dictionary

---

#### `fit(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> 'BaseModel'`

Fit the model to training data.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Training features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - Training targets

**Returns:**

- Type: `Self for method chaining`

---

#### `get_feature_importance(self) -> Dict[str, float]`

Get feature importance scores.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary mapping feature names to importance scores`

---

#### `get_metadata_routing(self)`

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

Returns
-------
routing : MetadataRequest
A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
routing information.

**Parameters:**

- `self`

---

#### `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
If True, will return the parameters for this estimator and
contained subobjects that are estimators.

Returns
-------
params : dict
Parameter names mapped to their values.

**Parameters:**

- `self`
- `deep` = True

---

#### `get_training_metrics(self) -> Dict[str, float]`

Get training metrics.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary of training metrics`

---

#### `predict(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame]) -> numpy.ndarray`

Make predictions using the fitted model.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Features to predict on

**Returns:**

- Type: `Predictions array`

---

#### `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

Parameters
----------
**params : dict
Estimator parameters.

Returns
-------
self : estimator instance
Estimator instance.

**Parameters:**

- `self`
- `params`

---

#### 🏗️ `BaseRegressor`

**Inherits from:** `BaseModel`

Base class for regression models.

**Properties:**

- `_repr_html_` (readable)
  - HTML representation of estimator.

This is redundant with the logic of `_repr_mimebundle_`. The latter
should be favorted in the long term, `_repr_html_` is only
implemented for consumers who do not interpret `_repr_mimbundle_`.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize the model with configuration.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Optional configuration dictionary

---

#### `fit(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> 'BaseModel'`

Fit the model to training data.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Training features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - Training targets

**Returns:**

- Type: `Self for method chaining`

---

#### `get_feature_importance(self) -> Dict[str, float]`

Get feature importance scores.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary mapping feature names to importance scores`

---

#### `get_metadata_routing(self)`

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

Returns
-------
routing : MetadataRequest
A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
routing information.

**Parameters:**

- `self`

---

#### `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
If True, will return the parameters for this estimator and
contained subobjects that are estimators.

Returns
-------
params : dict
Parameter names mapped to their values.

**Parameters:**

- `self`
- `deep` = True

---

#### `get_training_metrics(self) -> Dict[str, float]`

Get training metrics.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary of training metrics`

---

#### `predict(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame]) -> numpy.ndarray`

Make predictions using the fitted model.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Features to predict on

**Returns:**

- Type: `Predictions array`

---

#### `score(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> float`

Calculate R² score.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Test features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - True targets

**Returns:**

- Type: `R² score`

---

#### `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

Parameters
----------
**params : dict
Estimator parameters.

Returns
-------
self : estimator instance
Estimator instance.

**Parameters:**

- `self`
- `params`

---


## 📦 models.evaluator

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/evaluator.py`

Model evaluation utilities for coffee rating prediction.

This module provides comprehensive evaluation capabilities for regression models
including cross-validation, performance metrics, and visualization.

### 🔢 Constants

#### `SHAP_AVAILABLE`
- **Type:** `bool`
- **Value:** `True`

### 🏗️ Classes

#### 🏗️ `CoffeeModelEvaluator`

**Inherits from:** `BaseEvaluator`

Comprehensive model evaluator for coffee rating prediction models.

Provides detailed evaluation metrics, cross-validation, and visualization
capabilities for regression models.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize model evaluator.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with parameters:

---

#### `compare_models(self, models: Dict[str, models.base.BaseModel], X_test: Union[numpy.ndarray, pandas.core.frame.DataFrame], y_test: Union[numpy.ndarray, pandas.core.series.Series]) -> Dict[str, Any]`

Compare multiple models on the same test set.

**Parameters:**

- `self`
- `models` (typing.Dict[str, models.base.BaseModel])
  - Dictionary mapping model names to fitted models
- `X_test` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
  - Test features
- `y_test` (typing.Union[numpy.ndarray, pandas.core.series.Series])
  - Test targets

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Comparison results

---

#### `cross_validate(self, model: models.base.BaseModel, X: Union[numpy.ndarray, pandas.core.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series], cv: int = None) -> Dict[str, Any]`

Perform cross-validation on a model.

**Parameters:**

- `self`
- `model` (<class 'models.base.BaseModel'>)
  - Model to evaluate
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
  - Features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series])
  - Targets
- `cv` (<class 'int'>) = None
  - Number of cross-validation folds (uses config default if None)

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Cross-validation results

---

#### `evaluate(self, model: models.base.BaseModel, X_test: Union[numpy.ndarray, pandas.core.frame.DataFrame], y_test: Union[numpy.ndarray, pandas.core.series.Series]) -> Dict[str, Any]`

Evaluate a model on test data.

**Parameters:**

- `self`
- `model` (<class 'models.base.BaseModel'>)
  - Fitted model to evaluate
- `X_test` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
  - Test features
- `y_test` (typing.Union[numpy.ndarray, pandas.core.series.Series])
  - Test targets

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary with evaluation results

---

#### `evaluate_model(self, model: models.base.BaseModel, X_test: Union[numpy.ndarray, pandas.core.frame.DataFrame], y_test: Union[numpy.ndarray, pandas.core.series.Series]) -> Dict[str, float]`

Evaluate a model on test data (abstract method implementation).

**Parameters:**

- `self`
- `model` (<class 'models.base.BaseModel'>)
  - Fitted model to evaluate
- `X_test` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
  - Test features
- `y_test` (typing.Union[numpy.ndarray, pandas.core.series.Series])
  - Test targets

**Returns:**

- Type: `typing.Dict[str, float]`
- Dictionary with evaluation metrics

---

#### `generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str`

Generate a comprehensive evaluation report.

**Parameters:**

- `self`
- `evaluation_results` (typing.Dict[str, typing.Any])
  - Results from evaluate method

**Returns:**

- Type: `<class 'str'>`
- Formatted report string

---

#### `plot_feature_importance(self, feature_importance: Dict[str, float], model_name: str = 'Model', top_n: int = 15, save_path: Optional[str] = None) -> matplotlib.figure.Figure`

Plot feature importance.

**Parameters:**

- `self`
- `feature_importance` (typing.Dict[str, float])
  - Dictionary mapping feature names to importance scores
- `model_name` (<class 'str'>) = Model
  - Name of the model for the title
- `top_n` (<class 'int'>) = 15
  - Number of top features to display
- `save_path` (typing.Optional[str]) = None
  - Path to save the plot (optional)

**Returns:**

- Type: `<class 'matplotlib.figure.Figure'>`
- Matplotlib figure

---

#### `plot_model_comparison(self, comparison_results: Dict[str, Any], metric: str = 'r2', save_path: Optional[str] = None) -> matplotlib.figure.Figure`

Plot comparison of multiple models.

**Parameters:**

- `self`
- `comparison_results` (typing.Dict[str, typing.Any])
  - Results from compare_models method
- `metric` (<class 'str'>) = r2
  - Metric to compare (default: 'r2')
- `save_path` (typing.Optional[str]) = None
  - Path to save the plot (optional)

**Returns:**

- Type: `<class 'matplotlib.figure.Figure'>`
- Matplotlib figure

---

#### `plot_predictions(self, y_true: Union[numpy.ndarray, pandas.core.series.Series], y_pred: numpy.ndarray, model_name: str = 'Model', save_path: Optional[str] = None) -> matplotlib.figure.Figure`

Plot predicted vs actual values.

**Parameters:**

- `self`
- `y_true` (typing.Union[numpy.ndarray, pandas.core.series.Series])
  - True values
- `y_pred` (<class 'numpy.ndarray'>)
  - Predicted values
- `model_name` (<class 'str'>) = Model
  - Name of the model for the title
- `save_path` (typing.Optional[str]) = None
  - Path to save the plot (optional)

**Returns:**

- Type: `<class 'matplotlib.figure.Figure'>`
- Matplotlib figure

---


## 📦 models.mnir

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/mnir.py`

Multinomial Inverse Regression (MNIR) implementation for coffee review analysis.

This module implements MNIR following the thesis methodology for analyzing
the relationship between text-based features and sensory attributes.

### 🔢 Constants

#### `SHAP_AVAILABLE`
- **Type:** `bool`
- **Value:** `True`

### 🏗️ Classes

#### 🏗️ `MultinomialInverseRegression`

**Inherits from:** `BaseRegressor`

Multinomial Inverse Regression (MNIR) implementation following thesis methodology.

MNIR is used to quantify the relationship between text-based features and sensory
attributes (acidity, body, aroma, aftertaste, flavor). The methodology follows:

1. Lasso regression feature selection (cv=5) to identify most relevant text predictors
2. Regression modeling to predict sensory attributes using selected features
3. Performance evaluation using MSE and R² metrics
4. SHAP analysis for feature interpretability

From thesis: "This approach was implemented following Lasso regression feature
selection, which helped identify the most relevant predictors from the
high-dimensional text data."

**Properties:**

- `_repr_html_` (readable)
  - HTML representation of estimator.

This is redundant with the logic of `_repr_mimebundle_`. The latter
should be favorted in the long term, `_repr_html_` is only
implemented for consumers who do not interpret `_repr_mimbundle_`.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize MNIR following thesis methodology.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with parameters:

---

#### `fit(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame], sensory_data: Dict[str, numpy.ndarray]) -> 'MultinomialInverseRegression'`

Fit MNIR models following thesis methodology.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
  - Text-based feature matrix (TF-IDF, BERT, GloVe, LDA topics, etc.)
- `sensory_data` (typing.Dict[str, numpy.ndarray])
  - Dict with sensory attributes {'aroma': scores, 'acid': scores, ...}

**Returns:**

- Type: `MultinomialInverseRegression`
- Self for method chaining

---

#### `generate_insights_report(self) -> str`

Generate a comprehensive insights report for all attributes.

**Parameters:**

- `self`

**Returns:**

- Type: `<class 'str'>`
- Formatted report string

---

#### `get_feature_importance(self) -> Dict[str, float]`

Get overall feature importance across all attributes.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, float]`
- Dictionary mapping feature names to importance scores

---

#### `get_metadata_routing(self)`

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

Returns
-------
routing : MetadataRequest
A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
routing information.

**Parameters:**

- `self`

---

#### `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
If True, will return the parameters for this estimator and
contained subobjects that are estimators.

Returns
-------
params : dict
Parameter names mapped to their values.

**Parameters:**

- `self`
- `deep` = True

---

#### `get_performance_summary(self) -> Dict[str, Dict[str, float]]`

Get performance summary for all attributes.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, typing.Dict[str, float]]`
- Dictionary with performance metrics for each attribute

---

#### `get_shap_summary(self, attribute: str) -> Optional[Dict[str, Any]]`

Get SHAP summary for a specific attribute.

**Parameters:**

- `self`
- `attribute` (<class 'str'>)
  - Sensory attribute

**Returns:**

- Type: `typing.Optional[typing.Dict[str, typing.Any]]`
- Dictionary with SHAP analysis results

---

#### `get_training_metrics(self) -> Dict[str, float]`

Get training metrics.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary of training metrics`

---

#### `load_model(self, filepath: str) -> 'MultinomialInverseRegression'`

Load a previously saved MNIR model.

**Parameters:**

- `self`
- `filepath` (<class 'str'>)
  - Path to the saved model

**Returns:**

- Type: `MultinomialInverseRegression`
- Self for method chaining

---

#### `predict(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame], attribute: str) -> numpy.ndarray`

Make predictions for a specific sensory attribute.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
  - Feature matrix
- `attribute` (<class 'str'>)
  - Sensory attribute to predict

**Returns:**

- Type: `<class 'numpy.ndarray'>`
- Predictions for the attribute

---

#### `predict_all_attributes(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame]) -> Dict[str, numpy.ndarray]`

Make predictions for all fitted sensory attributes.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
  - Feature matrix

**Returns:**

- Type: `typing.Dict[str, numpy.ndarray]`
- Dictionary mapping attributes to predictions

---

#### `save_model(self, filepath: str) -> None`

Save the fitted MNIR model.

**Parameters:**

- `self`
- `filepath` (<class 'str'>)
  - Path to save the model

**Returns:**

- Type: `None`

---

#### `score(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> float`

Calculate R² score.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Test features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - True targets

**Returns:**

- Type: `R² score`

---

#### `set_fit_request(self: models.mnir.MultinomialInverseRegression, *, sensory_data: Union[bool, NoneType, str] = '$UNCHANGED$') -> models.mnir.MultinomialInverseRegression`

Request metadata passed to the ``fit`` method.

Note that this method is only relevant if
``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
Please see :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

The options for each parameter are:

- ``True``: metadata is requested, and passed to ``fit`` if provided. The request is ignored if metadata is not provided.

- ``False``: metadata is not requested and the meta-estimator will not pass it to ``fit``.

- ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.

- ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.

The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
existing request. This allows you to change the request for some
parameters and not others.

.. versionadded:: 1.3

.. note::
This method is only relevant if this estimator is used as a
sub-estimator of a meta-estimator, e.g. used inside a
:class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.

Parameters
----------
sensory_data : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
Metadata routing for ``sensory_data`` parameter in ``fit``.

Returns
-------
self : object
The updated object.

**Parameters:**

- `self` (<class 'models.mnir.MultinomialInverseRegression'>)
- `sensory_data` (typing.Union[bool, NoneType, str]) = $UNCHANGED$

**Returns:**

- Type: `<class 'models.mnir.MultinomialInverseRegression'>`

---

#### `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

Parameters
----------
**params : dict
Estimator parameters.

Returns
-------
self : estimator instance
Estimator instance.

**Parameters:**

- `self`
- `params`

---

#### `set_predict_request(self: models.mnir.MultinomialInverseRegression, *, attribute: Union[bool, NoneType, str] = '$UNCHANGED$') -> models.mnir.MultinomialInverseRegression`

Request metadata passed to the ``predict`` method.

Note that this method is only relevant if
``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
Please see :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

The options for each parameter are:

- ``True``: metadata is requested, and passed to ``predict`` if provided. The request is ignored if metadata is not provided.

- ``False``: metadata is not requested and the meta-estimator will not pass it to ``predict``.

- ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.

- ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.

The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
existing request. This allows you to change the request for some
parameters and not others.

.. versionadded:: 1.3

.. note::
This method is only relevant if this estimator is used as a
sub-estimator of a meta-estimator, e.g. used inside a
:class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.

Parameters
----------
attribute : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
Metadata routing for ``attribute`` parameter in ``predict``.

Returns
-------
self : object
The updated object.

**Parameters:**

- `self` (<class 'models.mnir.MultinomialInverseRegression'>)
- `attribute` (typing.Union[bool, NoneType, str]) = $UNCHANGED$

**Returns:**

- Type: `<class 'models.mnir.MultinomialInverseRegression'>`

---


## 📦 models.regressors

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/regressors.py`

Individual regressor implementations for coffee rating prediction.

This module provides wrapped implementations of standard regression models
that integrate with the configuration system and follow the base model interface.

### 🔢 Constants

#### `XGBOOST_AVAILABLE`
- **Type:** `bool`
- **Value:** `True`

### 🏗️ Classes

#### 🏗️ `CoffeeLassoRegression`

**Inherits from:** `BaseRegressor`

Lasso Regression wrapper with hyperparameter tuning.

**Properties:**

- `_repr_html_` (readable)
  - HTML representation of estimator.

This is redundant with the logic of `_repr_mimebundle_`. The latter
should be favorted in the long term, `_repr_html_` is only
implemented for consumers who do not interpret `_repr_mimbundle_`.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize Lasso Regression model.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with parameters:

---

#### `fit(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series]) -> 'CoffeeLassoRegression'`

Fit Lasso Regression with hyperparameter tuning.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series])

**Returns:**

- Type: `CoffeeLassoRegression`

---

#### `get_feature_importance(self) -> Dict[str, float]`

Get feature importance (non-zero coefficients).

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, float]`

---

#### `get_metadata_routing(self)`

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

Returns
-------
routing : MetadataRequest
A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
routing information.

**Parameters:**

- `self`

---

#### `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
If True, will return the parameters for this estimator and
contained subobjects that are estimators.

Returns
-------
params : dict
Parameter names mapped to their values.

**Parameters:**

- `self`
- `deep` = True

---

#### `get_selected_features(self) -> List[str]`

Get names of features selected by Lasso.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.List[str]`

---

#### `get_training_metrics(self) -> Dict[str, float]`

Get training metrics.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary of training metrics`

---

#### `predict(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame]) -> numpy.ndarray`

Make predictions.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])

**Returns:**

- Type: `<class 'numpy.ndarray'>`

---

#### `score(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> float`

Calculate R² score.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Test features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - True targets

**Returns:**

- Type: `R² score`

---

#### `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

Parameters
----------
**params : dict
Estimator parameters.

Returns
-------
self : estimator instance
Estimator instance.

**Parameters:**

- `self`
- `params`

---

#### 🏗️ `CoffeeLinearRegression`

**Inherits from:** `BaseRegressor`

Linear Regression wrapper for coffee rating prediction.

Simple linear regression model with optional feature scaling.

**Properties:**

- `_repr_html_` (readable)
  - HTML representation of estimator.

This is redundant with the logic of `_repr_mimebundle_`. The latter
should be favorted in the long term, `_repr_html_` is only
implemented for consumers who do not interpret `_repr_mimbundle_`.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize Linear Regression model.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with parameters:

---

#### `fit(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series]) -> 'CoffeeLinearRegression'`

Fit Linear Regression model.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series])

**Returns:**

- Type: `CoffeeLinearRegression`

---

#### `get_feature_importance(self) -> Dict[str, float]`

Get feature importance (coefficients).

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, float]`

---

#### `get_metadata_routing(self)`

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

Returns
-------
routing : MetadataRequest
A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
routing information.

**Parameters:**

- `self`

---

#### `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
If True, will return the parameters for this estimator and
contained subobjects that are estimators.

Returns
-------
params : dict
Parameter names mapped to their values.

**Parameters:**

- `self`
- `deep` = True

---

#### `get_training_metrics(self) -> Dict[str, float]`

Get training metrics.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary of training metrics`

---

#### `predict(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame]) -> numpy.ndarray`

Make predictions.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])

**Returns:**

- Type: `<class 'numpy.ndarray'>`

---

#### `score(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> float`

Calculate R² score.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Test features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - True targets

**Returns:**

- Type: `R² score`

---

#### `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

Parameters
----------
**params : dict
Estimator parameters.

Returns
-------
self : estimator instance
Estimator instance.

**Parameters:**

- `self`
- `params`

---

#### 🏗️ `CoffeeRandomForest`

**Inherits from:** `BaseRegressor`

Random Forest Regressor wrapper with hyperparameter tuning.

**Properties:**

- `_repr_html_` (readable)
  - HTML representation of estimator.

This is redundant with the logic of `_repr_mimebundle_`. The latter
should be favorted in the long term, `_repr_html_` is only
implemented for consumers who do not interpret `_repr_mimbundle_`.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize Random Forest model.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with parameters:

---

#### `fit(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series]) -> 'CoffeeRandomForest'`

Fit Random Forest with optional hyperparameter tuning.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series])

**Returns:**

- Type: `CoffeeRandomForest`

---

#### `get_feature_importance(self) -> Dict[str, float]`

Get feature importance from Random Forest.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, float]`

---

#### `get_metadata_routing(self)`

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

Returns
-------
routing : MetadataRequest
A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
routing information.

**Parameters:**

- `self`

---

#### `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
If True, will return the parameters for this estimator and
contained subobjects that are estimators.

Returns
-------
params : dict
Parameter names mapped to their values.

**Parameters:**

- `self`
- `deep` = True

---

#### `get_training_metrics(self) -> Dict[str, float]`

Get training metrics.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary of training metrics`

---

#### `predict(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame]) -> numpy.ndarray`

Make predictions.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])

**Returns:**

- Type: `<class 'numpy.ndarray'>`

---

#### `score(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> float`

Calculate R² score.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Test features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - True targets

**Returns:**

- Type: `R² score`

---

#### `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

Parameters
----------
**params : dict
Estimator parameters.

Returns
-------
self : estimator instance
Estimator instance.

**Parameters:**

- `self`
- `params`

---

#### 🏗️ `CoffeeRidgeRegression`

**Inherits from:** `BaseRegressor`

Ridge Regression wrapper with hyperparameter tuning.

**Properties:**

- `_repr_html_` (readable)
  - HTML representation of estimator.

This is redundant with the logic of `_repr_mimebundle_`. The latter
should be favorted in the long term, `_repr_html_` is only
implemented for consumers who do not interpret `_repr_mimbundle_`.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize Ridge Regression model.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with parameters:

---

#### `fit(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series]) -> 'CoffeeRidgeRegression'`

Fit Ridge Regression with hyperparameter tuning.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series])

**Returns:**

- Type: `CoffeeRidgeRegression`

---

#### `get_feature_importance(self) -> Dict[str, float]`

Get feature importance (coefficients).

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, float]`

---

#### `get_metadata_routing(self)`

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

Returns
-------
routing : MetadataRequest
A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
routing information.

**Parameters:**

- `self`

---

#### `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
If True, will return the parameters for this estimator and
contained subobjects that are estimators.

Returns
-------
params : dict
Parameter names mapped to their values.

**Parameters:**

- `self`
- `deep` = True

---

#### `get_training_metrics(self) -> Dict[str, float]`

Get training metrics.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary of training metrics`

---

#### `predict(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame]) -> numpy.ndarray`

Make predictions.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])

**Returns:**

- Type: `<class 'numpy.ndarray'>`

---

#### `score(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> float`

Calculate R² score.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Test features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - True targets

**Returns:**

- Type: `R² score`

---

#### `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

Parameters
----------
**params : dict
Estimator parameters.

Returns
-------
self : estimator instance
Estimator instance.

**Parameters:**

- `self`
- `params`

---

#### 🏗️ `CoffeeXGBoost`

**Inherits from:** `BaseRegressor`

XGBoost Regressor wrapper with hyperparameter tuning.

**Properties:**

- `_repr_html_` (readable)
  - HTML representation of estimator.

This is redundant with the logic of `_repr_mimebundle_`. The latter
should be favorted in the long term, `_repr_html_` is only
implemented for consumers who do not interpret `_repr_mimbundle_`.

**Methods:**

#### `__init__(self, config: Optional[Dict[str, Any]] = None)`

Initialize XGBoost model.

**Parameters:**

- `self`
- `config` (typing.Optional[typing.Dict[str, typing.Any]]) = None
  - Configuration dictionary with parameters:

---

#### `fit(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series]) -> 'CoffeeXGBoost'`

Fit XGBoost with optional hyperparameter tuning.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series])

**Returns:**

- Type: `CoffeeXGBoost`

---

#### `get_feature_importance(self) -> Dict[str, float]`

Get feature importance from XGBoost.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, float]`

---

#### `get_metadata_routing(self)`

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

Returns
-------
routing : MetadataRequest
A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
routing information.

**Parameters:**

- `self`

---

#### `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
If True, will return the parameters for this estimator and
contained subobjects that are estimators.

Returns
-------
params : dict
Parameter names mapped to their values.

**Parameters:**

- `self`
- `deep` = True

---

#### `get_training_metrics(self) -> Dict[str, float]`

Get training metrics.

**Parameters:**

- `self`

**Returns:**

- Type: `Dictionary of training metrics`

---

#### `predict(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame]) -> numpy.ndarray`

Make predictions.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame])

**Returns:**

- Type: `<class 'numpy.ndarray'>`

---

#### `score(self, X: Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame], y: Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series]) -> float`

Calculate R² score.

**Parameters:**

- `self`
- `X` (typing.Union[numpy.ndarray, pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Test features
- `y` (typing.Union[numpy.ndarray, pandas.core.series.Series, polars.series.series.Series])
  - True targets

**Returns:**

- Type: `R² score`

---

#### `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

Parameters
----------
**params : dict
Estimator parameters.

Returns
-------
self : estimator instance
Estimator instance.

**Parameters:**

- `self`
- `params`

---


## 📦 utils.cache

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/../utils/cache.py`

Caching utilities for expensive operations in coffee text analytics.

This module provides caching capabilities for feature extraction, model training,
and other computationally expensive operations.

### 🔧 Functions

#### `cache_info() -> Dict[str, Any]`

Get cache information.

**Returns:**

- Type: `typing.Dict[str, typing.Any]`

---

#### `cached_function(cache_type: str = 'general', max_age_hours: int = 24)`

Decorator for caching function results.

**Parameters:**

- `cache_type` (<class 'str'>) = general
  - Type of cache to use
- `max_age_hours` (<class 'int'>) = 24
  - Maximum age of cache entries

**Returns:**

- Decorated function

---

#### `clear_all_cache()`

Clear all cached data.

---

#### `get_cache_manager() -> utils.cache.CacheManager`

Get global cache manager instance.

**Returns:**

- Type: `<class 'utils.cache.CacheManager'>`
- CacheManager instance

---

### 🏗️ Classes

#### 🏗️ `CacheManager`

Comprehensive cache manager for expensive operations.

**Methods:**

#### `__init__(self, cache_dir: Union[str, pathlib.Path] = 'cache', max_age_hours: int = 24)`

Initialize cache manager.

**Parameters:**

- `self`
- `cache_dir` (typing.Union[str, pathlib.Path]) = cache
  - Directory to store cache files
- `max_age_hours` (<class 'int'>) = 24
  - Maximum age of cache entries in hours

---

#### `cache_info(self) -> Dict[str, Any]`

Get information about cache usage.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary with cache statistics

---

#### `clear_cache(self, cache_type: str = None) -> None`

Clear cache files.

**Parameters:**

- `self`
- `cache_type` (<class 'str'>) = None
  - Specific cache type to clear, or None for all

**Returns:**

- Type: `None`

---

#### `get(self, key: str, cache_type: str = 'general') -> Optional[Any]`

Get item from cache.

**Parameters:**

- `self`
- `key` (<class 'str'>)
  - Cache key
- `cache_type` (<class 'str'>) = general
  - Type of cache (features, models, data, preprocessing)

**Returns:**

- Type: `typing.Optional[typing.Any]`
- Cached item or None if not found/expired

---

#### `get_or_compute(self, key: str, compute_func: Callable, cache_type: str = 'general', *args, **kwargs) -> Any`

Get item from cache or compute and cache it.

**Parameters:**

- `self`
- `key` (<class 'str'>)
  - Cache key
- `compute_func` (typing.Callable)
  - Function to compute the value
- `cache_type` (<class 'str'>) = general
  - Type of cache
- `args`
- `kwargs`

**Returns:**

- Type: `typing.Any`
- Cached or computed value

---

#### `set(self, key: str, value: Any, cache_type: str = 'general') -> None`

Store item in cache.

**Parameters:**

- `self`
- `key` (<class 'str'>)
  - Cache key
- `value` (typing.Any)
  - Item to cache
- `cache_type` (<class 'str'>) = general
  - Type of cache (features, models, data, preprocessing)

**Returns:**

- Type: `None`

---

#### 🏗️ `FeatureCache`

Specialized cache for feature extraction operations.

**Methods:**

#### `__init__(self, cache_manager: utils.cache.CacheManager)`

Initialize feature cache.

**Parameters:**

- `self`
- `cache_manager` (<class 'utils.cache.CacheManager'>)
  - CacheManager instance

---

#### `get_bert_features(self, texts: list, model_name: str, compute_func: Callable) -> Any`

Get or compute BERT features.

**Parameters:**

- `self`
- `texts` (<class 'list'>)
  - List of texts
- `model_name` (<class 'str'>)
  - BERT model name
- `compute_func` (typing.Callable)
  - Function to compute features

**Returns:**

- Type: `typing.Any`
- BERT features

---

#### `get_tfidf_features(self, texts: list, config: dict, compute_func: Callable) -> Any`

Get or compute TF-IDF features.

**Parameters:**

- `self`
- `texts` (<class 'list'>)
  - List of texts
- `config` (<class 'dict'>)
  - TF-IDF configuration
- `compute_func` (typing.Callable)
  - Function to compute features

**Returns:**

- Type: `typing.Any`
- TF-IDF features

---

#### `get_topic_features(self, texts: list, n_topics: int, compute_func: Callable) -> Any`

Get or compute topic modeling features.

**Parameters:**

- `self`
- `texts` (<class 'list'>)
  - List of texts
- `n_topics` (<class 'int'>)
  - Number of topics
- `compute_func` (typing.Callable)
  - Function to compute features

**Returns:**

- Type: `typing.Any`
- Topic features

---

#### 🏗️ `ModelCache`

Specialized cache for model training operations.

**Methods:**

#### `__init__(self, cache_manager: utils.cache.CacheManager)`

Initialize model cache.

**Parameters:**

- `self`
- `cache_manager` (<class 'utils.cache.CacheManager'>)
  - CacheManager instance

---

#### `get_trained_model(self, model_type: str, X_hash: str, y_hash: str, config: dict, compute_func: Callable) -> Any`

Get or compute trained model.

**Parameters:**

- `self`
- `model_type` (<class 'str'>)
  - Type of model
- `X_hash` (<class 'str'>)
  - Hash of training features
- `y_hash` (<class 'str'>)
  - Hash of training targets
- `config` (<class 'dict'>)
  - Model configuration
- `compute_func` (typing.Callable)
  - Function to train model

**Returns:**

- Type: `typing.Any`
- Trained model

---


## 📦 utils.cleaning

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/../utils/cleaning.py`

Utility functions for cleaning and preprocessing coffee review data.
Includes functions for:
- Price standardization (conversion to USD per kilogram)
- Country extraction and standardization
- Text preprocessing and cleaning
- Data saving and loading
- Data quality checks

### 🔧 Functions

#### `analyze_agtron_values(df: polars.dataframe.frame.DataFrame) -> None`

Analyze Agtron values to understand their distribution and format.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame with 'agtron' column

**Returns:**

- Type: `None`

---

#### `analyze_country_distribution(df: polars.dataframe.frame.DataFrame) -> polars.dataframe.frame.DataFrame`

Analyze and visualize country distribution.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame with 'country_of_origin' column

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`

---

#### `analyze_missing_values(df: polars.dataframe.frame.DataFrame) -> None`

Analyze missing values in the dataset.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame to analyze

**Returns:**

- Type: `None`

---

#### `analyze_numerical_columns(df: polars.dataframe.frame.DataFrame) -> None`

Analyze numerical columns in the DataFrame.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame to analyze

**Returns:**

- Type: `None`

---

#### `analyze_outliers(df: polars.dataframe.frame.DataFrame, column: str) -> None`

Analyze outliers in specified column.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame to analyze
- `column` (<class 'str'>)
  - Column name to check for outliers

**Returns:**

- Type: `None`

---

#### `analyze_price_distribution(df: polars.dataframe.frame.DataFrame) -> None`

Analyze and visualize price distribution.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame with 'price_per_kg' column

**Returns:**

- Type: `None`

---

#### `analyze_roast_standardization(df: polars.dataframe.frame.DataFrame) -> None`

Analyze the relationship between Agtron readings and roast levels.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)

**Returns:**

- Type: `None`

---

#### `apply_text_preprocessing(df: polars.dataframe.frame.DataFrame, text_columns: list, for_embeddings: bool = True) -> polars.dataframe.frame.DataFrame`

Apply text preprocessing to specified columns.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - Input DataFrame
- `text_columns` (<class 'list'>)
  - List of columns to preprocess
- `for_embeddings` (<class 'bool'>) = True
  - Whether preprocessing is for embeddings (True) or topic modeling (False)

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- DataFrame with preprocessed text columns

---

#### `check_column_consistency(*dfs: polars.dataframe.frame.DataFrame) -> None`

Check if all DataFrames have the same columns.

**Parameters:**

- `dfs` (<class 'polars.dataframe.frame.DataFrame'>)

**Returns:**

- Type: `None`

---

#### `check_missing_values(df: polars.dataframe.frame.DataFrame, name: str) -> None`

Check for missing values in DataFrame.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
- `name` (<class 'str'>)

**Returns:**

- Type: `None`

---

#### `check_target_variable(df: polars.dataframe.frame.DataFrame, target_col: str) -> None`

Check the quality of a target variable.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame containing the target
- `target_col` (<class 'str'>)
  - Name of target column

**Returns:**

- Type: `None`

---

#### `clean_dataset(df: polars.dataframe.frame.DataFrame, min_rating: float = 80.0) -> tuple[polars.dataframe.frame.DataFrame, dict]`

Clean dataset by handling missing values and outliers.

This function performs comprehensive data cleaning including:
- Removing rows with missing critical values
- Filtering by minimum rating threshold
- Standardizing country information
- Dropping irrelevant columns

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
- `min_rating` (<class 'float'>) = 80.0

**Returns:**

- Type: `tuple[pl.DataFrame, dict]: (cleaned_data, cleaning_statistics)
- cleaned_data: Filtered and cleaned DataFrame
- cleaning_statistics: Dict with removal counts and percentages`

**Examples:**

```python
>>> cleaned_df, stats = clean_dataset(raw_df, min_rating=85.0)
>>> print(f"Removed {stats['rows_removed']} rows")
```

---

#### `clean_price(price_str: str) -> Optional[float]`

Extract and standardize price to USD per kilogram.
Handles different units and currencies.

**Parameters:**

- `price_str` (<class 'str'>)
  - String containing price information

**Returns:**

- Type: `typing.Optional[float]`
- float: Standardized price in USD per kilogram
None: If price cannot be extracted or standardized

Conversion rates:
- 1 kilogram = 2.20462 pounds = 35.274 ounces = 1000 grams
- NT$ (Taiwan Dollar) to USD conversion rate: approximately 1 NT$ = 0.032 USD

---

#### `correct_country_name(country: str, origin: str) -> str`

Corrects country names based on specific rules.

**Parameters:**

- `country` (<class 'str'>)
  - Extracted country name
- `origin` (<class 'str'>)
  - Original origin string

**Returns:**

- Type: `<class 'str'>`
- str: Corrected country name

---

#### `drop_irrelevant_columns(df: polars.dataframe.frame.DataFrame) -> polars.dataframe.frame.DataFrame`

Drop columns that are not useful for analysis.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame to clean

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- DataFrame with irrelevant columns removed

---

#### `extract_and_correct_country(df: polars.dataframe.frame.DataFrame) -> polars.dataframe.frame.DataFrame`

Extracts and corrects country information from the 'origin' column.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame containing 'origin' column

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- DataFrame with new 'country_of_origin' column

---

#### `extract_country(origin: str) -> str`

Extracts country information from the 'origin' column.

**Parameters:**

- `origin` (<class 'str'>)
  - String containing origin information

**Returns:**

- Type: `<class 'str'>`
- str: Extracted country name(s) or 'ND' if none found

---

#### `load_parquet(path: Union[pathlib.Path, str], name: str) -> polars.dataframe.frame.DataFrame`

Load DataFrame from parquet file.

**Parameters:**

- `path` (typing.Union[pathlib.Path, str])
- `name` (<class 'str'>)

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`

---

#### `process_and_analyze_text(df: polars.dataframe.frame.DataFrame, desc_columns: list) -> tuple[polars.dataframe.frame.DataFrame, polars.dataframe.frame.DataFrame, polars.dataframe.frame.DataFrame]`

Process text data for different analysis types and provide summary.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame with text columns
- `desc_columns` (<class 'list'>)
  - List of description columns to process

**Returns:**

- Type: `tuple[polars.dataframe.frame.DataFrame, polars.dataframe.frame.DataFrame, polars.dataframe.frame.DataFrame]`
- Tuple of (embeddings_df, topic_modeling_df, sentiment_df)

---

#### `profile_dataset(df: polars.dataframe.frame.DataFrame, name: str = 'Dataset') -> None`

Perform comprehensive profiling of a DataFrame.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame to profile
- `name` (<class 'str'>) = Dataset
  - Name of the dataset for display purposes

**Returns:**

- Type: `None`

---

#### `save_parquet(df: polars.dataframe.frame.DataFrame, path: Union[pathlib.Path, str], name: str) -> None`

Save DataFrame to parquet file.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
- `path` (typing.Union[pathlib.Path, str])
- `name` (<class 'str'>)

**Returns:**

- Type: `None`

---

#### `standardize_prices(df: polars.dataframe.frame.DataFrame) -> polars.dataframe.frame.DataFrame`

Standardize all prices in the dataset to USD per kilogram.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame containing 'est_price' column

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- DataFrame with new 'price_per_kg' column

---

#### `standardize_roast_degree(df: polars.dataframe.frame.DataFrame) -> polars.dataframe.frame.DataFrame`

Standardize roast degree using Agtron values.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame with 'agtron' column

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`

---

#### `summarize_column_changes(original_df: polars.dataframe.frame.DataFrame, cleaned_df: polars.dataframe.frame.DataFrame) -> None`

Summarize changes in columns after cleaning.

**Parameters:**

- `original_df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame before dropping columns
- `cleaned_df` (<class 'polars.dataframe.frame.DataFrame'>)
  - DataFrame after dropping columns

**Returns:**

- Type: `None`

---


## 📦 utils.data_quality

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/../utils/data_quality.py`

Data quality analysis utilities for coffee review data.

This module provides comprehensive data quality analysis functions
that work with both Pandas and Polars DataFrames.

### 🔧 Functions

#### `analyze_data_quality(df: Union[pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame]) -> None`

Analyze data quality including missing values, duplicates, and value ranges.

This function provides comprehensive data quality analysis that works with
both Pandas and Polars DataFrames. It automatically converts Pandas to Polars
for consistent analysis.

**Parameters:**

- `df` (typing.Union[pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Input DataFrame (Pandas or Polars)

**Returns:**

- Type: `None`

---

#### `calculate_sensory_stats(df: Union[pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame]) -> polars.dataframe.frame.DataFrame`

Calculate summary statistics for sensory attributes.

Computes mean, median, and standard deviation for all sensory rating columns.

**Parameters:**

- `df` (typing.Union[pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Input DataFrame (Pandas or Polars)

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- pl.DataFrame: DataFrame with statistical summaries for sensory attributes

---

#### `get_data_overview(df: Union[pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame]) -> None`

Display comprehensive overview of the dataset.

Works with both Pandas and Polars DataFrames, providing detailed
information about columns, data types, and sample values.

**Parameters:**

- `df` (typing.Union[pandas.core.frame.DataFrame, polars.dataframe.frame.DataFrame])
  - Input DataFrame (Pandas or Polars)

**Returns:**

- Type: `None`

---


## 📦 utils.doc_generator

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/../utils/doc_generator.py`

API Documentation Generator for Coffee Text Analytics

This module automatically generates comprehensive API documentation
by inspecting modules, classes, and functions to extract docstrings,
signatures, and usage examples.

### 🔧 Functions

#### `generate_api_docs(src_path: str = 'src', output_file: str = 'API_DOCUMENTATION.md')`

Generate API documentation for the entire project.

**Parameters:**

- `src_path` (<class 'str'>) = src
  - Path to source code directory
- `output_file` (<class 'str'>) = API_DOCUMENTATION.md
  - Output documentation file

---

### 🏗️ Classes

#### 🏗️ `APIDocumentationGenerator`

Generates comprehensive API documentation for Python modules.

**Methods:**

#### `__init__(self, src_path: str = 'src')`

Initialize the documentation generator.

**Parameters:**

- `self`
- `src_path` (<class 'str'>) = src
  - Path to the source code directory

---

#### `discover_modules(self) -> List[str]`

Discover all Python modules in the source directory.

**Parameters:**

- `self`

**Returns:**

- Type: `typing.List[str]`
- List of module names

---

#### `extract_module_info(self, module_name: str) -> Dict[str, Any]`

Extract comprehensive information from a module.

**Parameters:**

- `self`
- `module_name` (<class 'str'>)
  - Name of the module to analyze

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary with module information

---

#### `generate_markdown_docs(self, output_file: str = 'API_DOCUMENTATION.md') -> str`

Generate comprehensive markdown documentation.

**Parameters:**

- `self`
- `output_file` (<class 'str'>) = API_DOCUMENTATION.md
  - Output file name

**Returns:**

- Type: `<class 'str'>`
- Generated markdown content

---


## 📦 utils.performance

**File:** `Unknown`



⚠️ **Import Error:**
```
Failed to import module: No module named 'psutil'
```


## 📦 utils.polars_utils

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/../utils/polars_utils.py`

Polars optimization utilities for efficient data processing.

This module provides utilities to minimize Polars ↔ Pandas conversions
and optimize data processing performance.

### 🔧 Functions

#### `analyze_memory(df: polars.dataframe.frame.DataFrame) -> Dict[str, Any]`

Convenience function for memory analysis.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)

**Returns:**

- Type: `typing.Dict[str, typing.Any]`

---

#### `efficient_pandas_apply(df: polars.dataframe.frame.DataFrame, column: str, func: <built-in function callable>) -> polars.dataframe.frame.DataFrame`

Convenience function for efficient apply operations.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
- `column` (<class 'str'>)
- `func` (<built-in function callable>)

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`

---

#### `optimize_memory(df: polars.dataframe.frame.DataFrame) -> polars.dataframe.frame.DataFrame`

Convenience function for memory optimization.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`

---

#### `prepare_for_sklearn(df: polars.dataframe.frame.DataFrame, features: List[str], target: str = None)`

Convenience function for sklearn preparation.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
- `features` (typing.List[str])
- `target` (<class 'str'>) = None

---

### 🏗️ Classes

#### 🏗️ `DataTypeOptimizer`

Utility class for optimizing data types and memory usage.

**Methods:**

#### `analyze_memory_usage(df: polars.dataframe.frame.DataFrame) -> Dict[str, Any]`

Analyze memory usage of DataFrame.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - Polars DataFrame

**Returns:**

- Type: `typing.Dict[str, typing.Any]`
- Dictionary with memory usage statistics

---

#### `optimize_dtypes(df: polars.dataframe.frame.DataFrame) -> polars.dataframe.frame.DataFrame`

Optimize data types for memory efficiency.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - Polars DataFrame

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- DataFrame with optimized data types

---

#### 🏗️ `PolarsOptimizer`

Utility class for optimizing Polars operations and minimizing conversions.

**Methods:**

#### `batch_convert_for_sklearn(df: polars.dataframe.frame.DataFrame, feature_columns: List[str], target_column: str = None) -> tuple[pandas.core.frame.DataFrame, typing.Optional[pandas.core.series.Series]]`

Efficiently convert Polars DataFrame to format needed for sklearn.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - Polars DataFrame
- `feature_columns` (typing.List[str])
  - List of feature column names
- `target_column` (<class 'str'>) = None
  - Target column name (optional)

**Returns:**

- Type: `tuple[pandas.core.frame.DataFrame, typing.Optional[pandas.core.series.Series]]`
- Tuple of (features_df, target_series)

---

#### `efficient_apply(df: polars.dataframe.frame.DataFrame, column: str, func: <built-in function callable>, new_column: str = None) -> polars.dataframe.frame.DataFrame`

Apply a function to a Polars column efficiently without full conversion.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - Polars DataFrame
- `column` (<class 'str'>)
  - Column name to apply function to
- `func` (<built-in function callable>)
  - Function to apply
- `new_column` (<class 'str'>) = None
  - Name for new column (defaults to original column)

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- DataFrame with function applied

---

#### `efficient_groupby_stats(df: polars.dataframe.frame.DataFrame, group_col: str, agg_cols: List[str], stats: List[str] = ['mean', 'count']) -> polars.dataframe.frame.DataFrame`

Perform efficient groupby operations using Polars native functions.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - Polars DataFrame
- `group_col` (<class 'str'>)
  - Column to group by
- `agg_cols` (typing.List[str])
  - Columns to aggregate
- `stats` (typing.List[str]) = ['mean', 'count']
  - Statistics to compute

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- Aggregated DataFrame

---

#### `lazy_text_processing(df: polars.dataframe.frame.DataFrame, text_columns: List[str], operations: List[str] = ['lowercase', 'strip']) -> polars.dataframe.frame.DataFrame`

Apply text processing operations using Polars lazy evaluation.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)
  - Polars DataFrame
- `text_columns` (typing.List[str])
  - Text columns to process
- `operations` (typing.List[str]) = ['lowercase', 'strip']
  - List of operations to apply

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- DataFrame with processed text

---

#### `memory_efficient_join(left: polars.dataframe.frame.DataFrame, right: polars.dataframe.frame.DataFrame, on: Union[str, List[str]], how: str = 'inner') -> polars.dataframe.frame.DataFrame`

Perform memory-efficient joins using Polars.

**Parameters:**

- `left` (<class 'polars.dataframe.frame.DataFrame'>)
  - Left DataFrame
- `right` (<class 'polars.dataframe.frame.DataFrame'>)
  - Right DataFrame
- `on` (typing.Union[str, typing.List[str]])
  - Column(s) to join on
- `how` (<class 'str'>) = inner
  - Join type

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- Joined DataFrame

---


## 📦 utils.utils

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/../utils/utils.py`

Utility functions for coffee review analysis.

### 🔧 Functions

#### `convert_pandas_to_polars(df: pandas.core.frame.DataFrame) -> polars.dataframe.frame.DataFrame`

Convert pandas DataFrame to Polars with proper type handling.

**Parameters:**

- `df` (<class 'pandas.core.frame.DataFrame'>)

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- pl.DataFrame: Converted Polars DataFrame

Note:
Handles common type conversions and null value representations.

---

#### `convert_polars_to_pandas(df: polars.dataframe.frame.DataFrame) -> pandas.core.frame.DataFrame`

Convert Polars DataFrame to pandas with proper type handling.

**Parameters:**

- `df` (<class 'polars.dataframe.frame.DataFrame'>)

**Returns:**

- Type: `<class 'pandas.core.frame.DataFrame'>`
- pd.DataFrame: Converted pandas DataFrame

Note:
Preserves data types where possible and handles Polars-specific types.

---

#### `load_dataset_from_utils() -> polars.dataframe.frame.DataFrame`

Load coffee review data from hardcoded utility path.

This is a utility function that loads data from a hardcoded path.
Consider using load_main_dataset() from data.loader for main pipeline.

**Returns:**

- Type: `<class 'polars.dataframe.frame.DataFrame'>`
- pl.DataFrame: Coffee review dataset

Note:
This function uses a hardcoded path and may not work in all environments.
Prefer load_main_dataset() for production use.

---


## 📦 visualization.plots

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/../utils/../visualization/plots.py`

Visualization functions for coffee review analysis.

### 🔢 Constants

#### `PATHS`
- **Type:** `dict`
- **Value:** `{'root': PosixPath('/Users/seijas/Code/coffee-text-analytics'), 'data': PosixPath('/Users/seijas/Code/coffee-text-analytics/data'), 'raw': PosixPath('/Users/seijas/Code/coffee-text-analytics/data/raw'), 'processed': PosixPath('/Users/seijas/Code/coffee-text-analytics/data/processed'), 'models': PosixPath('/Users/seijas/Code/coffee-text-analytics/models'), 'output': PosixPath('/Users/seijas/Code/coffee-text-analytics/output'), 'figures': PosixPath('/Users/seijas/Code/coffee-text-analytics/output/figures')}`

### 🔧 Functions

#### `plot_boxplots(data: polars.dataframe.frame.DataFrame, columns: List[str]) -> None`

Generate box plots for specified columns.

**Parameters:**

- `data` (<class 'polars.dataframe.frame.DataFrame'>)
- `columns` (typing.List[str])

**Returns:**

- Type: `None`

---

#### `plot_categorical_distributions(data: polars.dataframe.frame.DataFrame, columns: List[str]) -> None`

Plot distributions for categorical columns.

**Parameters:**

- `data` (<class 'polars.dataframe.frame.DataFrame'>)
- `columns` (typing.List[str])

**Returns:**

- Type: `None`

---

#### `plot_kde(data: polars.dataframe.frame.DataFrame, columns: List[str]) -> None`

Generate KDE plots for specified columns.

**Parameters:**

- `data` (<class 'polars.dataframe.frame.DataFrame'>)
- `columns` (typing.List[str])

**Returns:**

- Type: `None`

---

#### `save_figure(fig: plotly.graph_objs._figure.Figure, filename: str, path: pathlib.Path = PosixPath('/Users/seijas/Code/coffee-text-analytics/output/figures')) -> None`

Save a plotly figure to the figures directory.

**Parameters:**

- `fig` (<class 'plotly.graph_objs._figure.Figure'>)
  - Plotly figure object
- `filename` (<class 'str'>)
  - Name for the saved figure
- `path` (<class 'pathlib.Path'>) = /Users/seijas/Code/coffee-text-analytics/output/figures
  - Directory to save figure

**Returns:**

- Type: `None`

---


## 📦 visualization.visualize

**File:** `/Users/seijas/Code/coffee-text-analytics/src/data/../features/../models/../utils/../visualization/visualize.py`

Visualization utilities for coffee review data analysis.

### 🔢 Constants

#### `SEABORN_AVAILABLE`
- **Type:** `bool`
- **Value:** `True`

#### `WORDCLOUD_AVAILABLE`
- **Type:** `bool`
- **Value:** `False`

### 🔧 Functions

#### `create_visualizations(features_file, models_dir='models', output_dir='output/figures')`

Create visualizations from features and model results.

**Parameters:**

- `features_file`
  - Path to features CSV file
- `models_dir` = models
  - Directory containing models
- `output_dir` = output/figures
  - Directory to save visualizations

---

#### `plot_correlation_matrix(df, columns=None, title='Feature Correlation Matrix')`

Create a correlation matrix heatmap.

**Parameters:**

- `df`
  - DataFrame with features
- `columns` = None
  - Specific columns to include (if None, uses all numeric columns)
- `title` = Feature Correlation Matrix
  - Plot title

**Returns:**

- Matplotlib figure

---

#### `plot_feature_correlation_to_rating(df, target_column='rating', n_features=10, output_dir=None)`

Visualize features with strongest correlation to rating.

**Parameters:**

- `df`
  - DataFrame with features
- `target_column` = rating
  - Target column name
- `n_features` = 10
  - Number of top features to show
- `output_dir` = None
  - Directory to save outputs

**Returns:**

- Matplotlib figure

---

#### `plot_model_comparison(results_file, output_dir)`

Create visualizations comparing model performance.

**Parameters:**

- `results_file`
  - Path to model results JSON file
- `output_dir`
  - Directory to save outputs

---

#### `plot_rating_distribution(df, rating_column='rating', title=None)`

Visualize the distribution of coffee ratings.

**Parameters:**

- `df`
  - DataFrame containing rating data
- `rating_column` = rating
  - Name of rating column
- `title` = None
  - Custom title (default: Rating Distribution)

**Returns:**

- Matplotlib figure

---

#### `plot_word_clouds(model_file, vectorizer_file, output_dir, n_topics=10)`

Generate word clouds for topic models.

**Parameters:**

- `model_file`
  - Path to topic model file
- `vectorizer_file`
  - Path to vectorizer file
- `output_dir`
  - Directory to save outputs
- `n_topics` = 10
  - Number of topics to visualize

---

#### `save_figure(fig, filename, output_dir, dpi=300)`

Save a matplotlib figure to disk.

**Parameters:**

- `fig`
  - Matplotlib figure
- `filename`
  - Name for the file
- `output_dir`
  - Output directory
- `dpi` = 300
  - Resolution (dots per inch)

---

