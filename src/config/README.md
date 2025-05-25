# Configuration Management System

The Coffee Text Analytics project uses a comprehensive configuration management system that provides centralized, environment-aware configuration for all aspects of the analysis pipeline.

## Features

- **Environment-specific configurations** (development, production, testing, CI/CD)
- **Centralized parameter management** for models, features, data processing, and visualization
- **Configuration validation** with detailed error reporting and warnings
- **Command-line interface** for configuration management
- **Automatic directory creation** and path management
- **Dependency checking** and validation
- **Configuration export/import** capabilities

## Quick Start

```python
# Import the main configuration instance
from config import config

# Access configuration values
print(f"Environment: {config.environment}")
print(f"Data file: {config.paths.get_raw_data_path()}")
print(f"Target column: {config.models.target_column}")
print(f"Number of topics: {config.features.n_topics}")

# Get model hyperparameters
rf_params = config.get_model_params("random_forest")
print(f"Random Forest params: {rf_params}")
```

## Configuration Structure

The configuration system is organized into several components:

### PathConfig
Manages all file and directory paths:
```python
config.paths.root              # Project root directory
config.paths.raw               # Raw data directory
config.paths.processed         # Processed data directory
config.paths.models            # Models directory
config.paths.output            # Output directory
config.paths.figures           # Figures directory

# Convenience methods
config.paths.get_raw_data_path()      # Full path to raw data file
config.paths.get_processed_data_path() # Full path to processed data file
config.paths.get_features_data_path()  # Full path to features data file
```

### ModelConfig
Manages machine learning model settings:
```python
config.models.target_column           # Target variable name
config.models.text_columns           # List of text columns to analyze
config.models.models_to_train        # List of models to train
config.models.cv_folds              # Cross-validation folds
config.models.test_size             # Test set size
config.models.random_forest_params  # Random Forest hyperparameters
config.models.xgboost_params        # XGBoost hyperparameters
config.models.linear_params         # Linear model hyperparameters
config.models.mnir_params           # MNIR hyperparameters
```

### FeatureConfig
Manages feature extraction settings:
```python
config.features.tfidf_max_features   # TF-IDF maximum features
config.features.tfidf_ngram_range    # N-gram range for TF-IDF
config.features.n_topics             # Number of topics for topic modeling
config.features.bert_model_name      # BERT model name
config.features.bert_batch_size      # BERT batch size
config.features.glove_dimensions     # GloVe embedding dimensions
```

### DataConfig
Manages data processing settings:
```python
config.data.min_rating              # Minimum rating threshold
config.data.max_missing_percentage  # Maximum missing data percentage
config.data.standardize_countries   # Whether to standardize country names
config.data.standardize_prices      # Whether to standardize prices
```

### VisualizationConfig
Manages plotting and visualization settings:
```python
config.visualization.figure_width   # Default figure width
config.visualization.figure_height  # Default figure height
config.visualization.color_palette  # Color palette for plots
config.visualization.export_dpi     # Export DPI for figures
```

## Environment Configurations

The system supports multiple environments with different parameter sets:

### Development (default)
- Balanced settings for development work
- Moderate model complexity
- Console logging enabled
- Lower DPI for faster rendering

### Production
- Optimized for best performance
- Higher model complexity
- File logging enabled
- High-quality visualizations

### Testing
- Fast execution for testing
- Minimal model complexity
- Debug logging
- Low-quality visualizations

### CI/CD
- Minimal settings for continuous integration
- Only fast models
- Error-level logging only

## Using Different Environments

### Via Environment Variable
```bash
export COFFEE_ENV=production
python main.py
```

### Via Command Line
```bash
python main.py --environment production
```

### Programmatically
```python
from config import Config
from config.environments import apply_environment_config

# Create config for specific environment
config = Config()
config = apply_environment_config(config, "production")
```

## Configuration Validation

The system includes comprehensive validation:

```python
from config.validation import validate_config, print_config_summary

# Validate configuration
is_valid = validate_config(config)

# Print detailed summary
print_config_summary(config)
```

### Validation Features
- **Path validation**: Checks if directories exist and are writable
- **Parameter validation**: Ensures parameters are within valid ranges
- **Model validation**: Validates model hyperparameters
- **Environment validation**: Checks environment-specific settings
- **Dependency checking**: Verifies required packages are installed

## Command-Line Interface

The configuration system includes a CLI for management tasks:

```bash
# Show configuration summary
python -m config.cli --summary

# Validate configuration
python -m config.cli --validate

# Check dependencies
python -m config.cli --check-deps

# Show environment-specific configuration
python -m config.cli --environment production

# List available environments
python -m config.cli --list-environments

# Compare environments
python -m config.cli --compare development production

# Export configuration to JSON
python -m config.cli --export config.json

# Get specific configuration value
python -m config.cli --get models.target_column
```

## Customizing Configuration

### Creating Custom Environments
```python
from config.environments import create_custom_environment

# Create custom environment based on development
custom_config = create_custom_environment(
    base_environment="development",
    overrides={
        "features": {
            "n_topics": 20,
            "tfidf_max_features": 15000,
        },
        "models": {
            "cv_folds": 10,
        }
    }
)
```

### Runtime Configuration Overrides
```python
from config import config

# Temporarily override settings
config.features.n_topics = 15
config.models.cv_folds = 8

# Validate after changes
from config.validation import validate_config
is_valid = validate_config(config)
```

## Integration with Main Pipeline

The main pipeline automatically uses the configuration system:

```bash
# Run with default configuration
python main.py

# Run with specific environment
python main.py --environment production

# Override specific parameters
python main.py --n_topics 15 --models linear random_forest

# Validate configuration before running
python main.py --validate_config
```

## Best Practices

1. **Use environment variables** for deployment-specific settings
2. **Validate configuration** before running long pipelines
3. **Export configuration** for reproducibility
4. **Use appropriate environments** for different use cases
5. **Check dependencies** before running analysis

## Configuration Files

The configuration system consists of several files:

- `settings.py` - Main configuration classes and logic
- `validation.py` - Configuration validation utilities
- `environments.py` - Environment-specific presets
- `cli.py` - Command-line interface
- `__init__.py` - Module exports

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `src` is in your Python path
2. **Path errors**: Check that data files exist in expected locations
3. **Validation failures**: Use `--validate` to see specific issues
4. **Missing dependencies**: Use `--check-deps` to verify installations

### Getting Help

```bash
# Show CLI help
python -m config.cli --help

# Validate and see warnings
python -m config.cli --validate

# Check what's missing
python -m config.cli --check-deps
```

## Examples

### Basic Usage
```python
from config import config

# Load data using configured path
import polars as pl
df = pl.read_csv(config.paths.get_raw_data_path())

# Use configured text columns
text_cols = config.models.text_columns
print(f"Analyzing columns: {text_cols}")

# Get model parameters
rf_params = config.get_model_params("random_forest")
```

### Environment Switching
```python
from config import Config
from config.environments import apply_environment_config

# Start with development config
config = Config("development")
print(f"Dev topics: {config.features.n_topics}")

# Switch to production
config = apply_environment_config(config, "production")
print(f"Prod topics: {config.features.n_topics}")
```

### Configuration Export
```python
from config import config
import json

# Export current configuration
config_dict = config.to_dict()
with open("my_config.json", "w") as f:
    json.dump(config_dict, f, indent=2, default=str)
```

This configuration system provides a robust foundation for managing all aspects of the Coffee Text Analytics pipeline, ensuring consistency, reproducibility, and easy deployment across different environments. 