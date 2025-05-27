# Box-Cox Dual Pipeline Documentation

## Overview

The Box-Cox Dual Pipeline is a comprehensive feature that implements the thesis methodology for testing Box-Cox transformation on target variables. It trains models both with and without transformation, compares performance, and generates evidence-based recommendations.

## Thesis Alignment

Following the thesis approach:
- ✅ **Test Box-Cox transformation** on coffee ratings
- ✅ **Compare performance** across all models  
- ✅ **Document decision** with clear reasoning
- ✅ **Conclude appropriately** (thesis found no significant benefit)

## Usage

### Command Line Interface

```bash
# Enable Box-Cox dual pipeline
python main.py --steps train --box_cox_dual

# Enable single Box-Cox transformation
python main.py --steps train --box_cox

# Default behavior (no transformation)
python main.py --steps train
```

### Configuration

```python
# In src/config/settings.py
class ModelConfig:
    # Box-Cox transformation settings (thesis methodology)
    box_cox_enabled: bool = False  # Default: no transformation
    box_cox_dual_pipeline: bool = False  # Run comparison pipeline
    box_cox_config: Dict[str, Any] = {
        "lambda_range": (-2, 2),  # Range for lambda parameter search
        "method": "mle",  # Maximum likelihood estimation
        "alpha": 0.05,  # Significance level for normality tests
        "save_comparison": True,  # Save comparison results
        "random_state": 57,
    }
```

## Implementation Details

### Phase 1: Baseline Training (No Transformation)
- Trains all models on original target variable
- Evaluates performance using standard metrics (R², RMSE, MAE)
- Saves baseline models to `output/models_baseline/`

### Phase 2: Box-Cox Transformation
- Applies Box-Cox transformation to target variable
- Finds optimal lambda parameter using maximum likelihood
- Performs normality tests (Shapiro-Wilk) on original and transformed data
- Logs transformation statistics and improvements

### Phase 3: Box-Cox Model Training
- Trains all models on transformed target variable
- Makes predictions and inverse-transforms to original scale
- Evaluates performance against original target values
- Saves Box-Cox models to `output/models_boxcox/`

### Phase 4: Comparison and Analysis
- Compares performance metrics between baseline and Box-Cox approaches
- Calculates improvement statistics per model
- Generates overall improvement rate and average R² change

### Phase 5: Recommendation Generation
Following thesis methodology, the system recommends:
- **NO_TRANSFORMATION** if < 50% models improve or avg improvement < 0.01
- **NO_TRANSFORMATION** if improvement < 0.05 (complexity not justified)
- **NO_TRANSFORMATION** even with improvement (following thesis conclusion)

## Output Files

### Models
- `output/models_baseline/` - Models trained without transformation
- `output/models_boxcox/` - Models trained with transformation
- `output/models_boxcox/box_cox_transformer.pkl` - Fitted transformer

### Results
- `output/box_cox_dual_pipeline_results.json` - Comprehensive comparison report

### Example Results Structure
```json
{
  "methodology": "Box-Cox Dual Pipeline (Thesis Approach)",
  "baseline_results": {
    "summary_metrics": {
      "r2": {"ridge": 0.8999, "linear": 0.7965, ...},
      "rmse": {"ridge": 0.4104, "linear": 0.5852, ...}
    },
    "best_models": {"r2": "ridge", "rmse": "ridge", ...}
  },
  "boxcox_results": {
    "summary_metrics": {...},
    "best_models": {...}
  },
  "transformation_stats": {
    "lambda": 11.4093,
    "original_skewness": -0.4391,
    "transformed_skewness": -0.0153,
    "normality_improvement": true
  },
  "overall_stats": {
    "models_improved": 2,
    "total_models": 7,
    "improvement_rate": 0.286,
    "avg_r2_improvement": -0.4579
  },
  "recommendation": "NO_TRANSFORMATION",
  "recommendation_reason": "Box-Cox transformation shows minimal benefit..."
}
```

## Example Output

```
🎯 BOX-COX DUAL PIPELINE SUMMARY
============================================================
📊 Models tested: 7
📈 Models improved with Box-Cox: 2 (28.6%)
📊 Average R² improvement: -0.4579
🎯 Recommendation: NO_TRANSFORMATION
📝 Reason: Box-Cox transformation shows minimal benefit (only 2/7 models 
improved, avg R² improvement: -0.4579). Following thesis methodology: 
use baseline approach.
============================================================
```

## Technical Implementation

### Core Classes

#### `BoxCoxTransformer`
```python
from utils.transformations import BoxCoxTransformer

# Initialize with configuration
transformer = BoxCoxTransformer(config.models.box_cox_config)

# Fit and transform
y_transformed = transformer.fit_transform(y_train)
y_test_transformed = transformer.transform(y_test)

# Inverse transform predictions
predictions_original = transformer.inverse_transform(predictions_transformed)

# Save/load transformer
transformer.save_transformer("transformer.pkl")
loaded_transformer = BoxCoxTransformer.load_transformer("transformer.pkl")
```

#### `CoffeeModelEvaluator`
```python
from models import CoffeeModelEvaluator

evaluator = CoffeeModelEvaluator()

# Compare models with direct predictions (for Box-Cox)
results = evaluator.compare_models_with_predictions(predictions_dict, y_test)
```

### Dual Pipeline Function
```python
from utils.transformations import run_box_cox_dual_pipeline

results = run_box_cox_dual_pipeline(
    X_train, X_test, y_train, y_test, models_dict, config, logger
)
```

## Benefits

1. **Thesis Compliance**: Exactly follows thesis methodology for transformation testing
2. **Comprehensive Analysis**: Tests all models with both approaches
3. **Evidence-Based**: Makes recommendations based on actual performance data
4. **Reproducible**: Saves all models and results for future analysis
5. **Configurable**: Easy to enable/disable via command line or configuration

## Best Practices

1. **Use with larger samples** (5-10%) for reliable results
2. **Review transformation statistics** to understand data changes
3. **Consider computational cost** - dual pipeline takes ~2x training time
4. **Follow thesis conclusion** - transformation rarely provides significant benefit
5. **Document decisions** using generated reports for reproducibility

## Integration with Pipeline

The Box-Cox dual pipeline integrates seamlessly with the existing training pipeline:

```python
# In main.py train_models() function
if config.models.box_cox_dual_pipeline:
    # Run comprehensive dual pipeline
    dual_pipeline_results = run_box_cox_dual_pipeline(...)
    comparison_results = dual_pipeline_results["baseline_results"]
elif config.models.box_cox_enabled:
    # Run single Box-Cox transformation
    # ... transformation logic
else:
    # Standard training (default)
    comparison_results = evaluator.compare_models(trained_models, X_test, y_test)
```

This ensures backward compatibility while providing advanced transformation testing capabilities when needed. 