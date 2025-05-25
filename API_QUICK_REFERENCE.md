# 🚀 Coffee Text Analytics - Quick API Reference

*For complete documentation, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md)*

## 📋 **Essential Components**

### 🏗️ **Core Architecture**

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| **Features** | Text feature extraction | `CoffeeFeatureManager`, `TfidfExtractor`, `BertExtractor` |
| **Models** | Machine learning models | `CoffeeLinearRegression`, `MultinomialInverseRegression`, `CoffeeModelEvaluator` |
| **Data** | Data preprocessing | `preprocess_text()`, data cleaning utilities |
| **Config** | Configuration management | `Config`, environment presets |
| **Utils** | Optimization & utilities | `PolarsOptimizer`, `CacheManager`, `PerformanceProfiler` |

---

## 🔧 **Quick Start Examples**

### **1. Feature Extraction**
```python
from features import CoffeeFeatureManager

# Initialize feature manager
config = {
    "extractors": {"tfidf": True, "bert": False, "topics": True},
    "tfidf": {"max_features": 5000, "ngram_range": (1, 3)}
}
feature_manager = CoffeeFeatureManager(config)

# Fit and extract features
texts = ["Great coffee with fruity notes", "Smooth and balanced"]
feature_manager.fit(texts)
features_df = feature_manager.extract_all_features(df, text_columns=["desc_1"])
```

### **2. Model Training**
```python
from models import CoffeeLinearRegression, MultinomialInverseRegression

# Traditional regression
model = CoffeeLinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# MNIR (thesis methodology)
mnir = MultinomialInverseRegression()
mnir.fit(X_train, y_attributes_dict)  # Multiple target attributes
results = mnir.predict(X_test)
```

### **3. Performance Optimization**
```python
from utils.polars_utils import efficient_pandas_apply, optimize_memory
from utils.cache import get_cache_manager

# Efficient operations
optimized_df = efficient_pandas_apply(df, "text_col", preprocessing_func)
memory_optimized_df = optimize_memory(df)

# Caching
cache_manager = get_cache_manager()
result = cache_manager.get_or_compute("key", expensive_function, args)
```

### **4. Configuration Management**
```python
from config import Config, apply_environment_config

# Load configuration
config = Config()
config = apply_environment_config(config, "production")

# Access settings
max_features = config.features.tfidf_max_features
logging_level = config.logging.level
```

---

## 📦 **Module Overview**

### **🎯 Features Module** (`src/features/`)
- **`CoffeeFeatureManager`**: Orchestrates all feature extraction
- **`TfidfExtractor`**: TF-IDF vectorization with caching
- **`BertExtractor`**: BERT embeddings (requires transformers)
- **`TopicExtractor`**: LDA topic modeling
- **`SentimentExtractor`**: Sentiment analysis

### **🤖 Models Module** (`src/models/`)
- **`CoffeeLinearRegression`**: Linear regression with coffee-specific features
- **`CoffeeRandomForest`**: Random forest with hyperparameter tuning
- **`MultinomialInverseRegression`**: MNIR implementation (thesis methodology)
- **`CoffeeModelEvaluator`**: Comprehensive model evaluation

### **🔧 Utils Module** (`src/utils/`)
- **`PolarsOptimizer`**: Efficient DataFrame operations
- **`CacheManager`**: Multi-layer caching system
- **`PerformanceProfiler`**: Performance measurement and benchmarking
- **`cleaning.py`**: Data cleaning and preprocessing utilities

### **⚙️ Config Module** (`src/config/`)
- **`Config`**: Main configuration class
- **`environments.py`**: Environment-specific presets
- **`validation.py`**: Configuration validation

---

## 🎯 **Common Use Cases**

### **End-to-End Pipeline**
```python
# 1. Load and configure
from config import Config
config = Config()

# 2. Preprocess data
from data.preprocessing import preprocess_text
processed_texts = [preprocess_text(text) for text in raw_texts]

# 3. Extract features
from features import CoffeeFeatureManager
feature_manager = CoffeeFeatureManager(config.features)
feature_manager.fit(processed_texts)
features = feature_manager.extract_all_features(df, ["desc_1", "desc_2"])

# 4. Train model
from models import CoffeeLinearRegression
model = CoffeeLinearRegression()
model.fit(features, targets)

# 5. Evaluate
from models import CoffeeModelEvaluator
evaluator = CoffeeModelEvaluator()
metrics = evaluator.evaluate_model(model, X_test, y_test)
```

### **Performance Optimization**
```python
from utils.performance import PerformanceProfiler
from utils.cache import get_cache_manager

# Profile performance
profiler = PerformanceProfiler()
with profiler.measure("feature_extraction"):
    features = extract_features(texts)

# Use caching
cache = get_cache_manager()
features = cache.get_or_compute("features_key", extract_features, texts)
```

### **Configuration Management**
```python
from config import Config, apply_environment_config

# Development setup
config = Config()
config = apply_environment_config(config, "development")

# Production setup
config = apply_environment_config(config, "production")
```

---

## 🔍 **Key Design Patterns**

### **1. Component-Based Architecture**
- Each feature extractor is independent
- Models follow consistent interface
- Easy to add new components

### **2. Configuration-Driven**
- All parameters centralized in config
- Environment-specific overrides
- Validation and type checking

### **3. Performance-Optimized**
- Polars-first data processing
- Multi-layer caching system
- Memory optimization utilities

### **4. Comprehensive Testing**
- Integration tests for end-to-end workflows
- Performance benchmarking
- Component-level unit tests

---

## 📚 **Documentation Structure**

| File | Purpose |
|------|---------|
| `API_DOCUMENTATION.md` | Complete API reference (6,800+ lines) |
| `API_QUICK_REFERENCE.md` | This quick start guide |
| `REFACTORING_PLAN.md` | Architecture evolution plan |
| `PHASE_5_PERFORMANCE_SUMMARY.md` | Performance optimization details |

---

## 🎯 **Next Steps**

1. **Read the full API documentation** for detailed method signatures
2. **Check the configuration options** in `config/environments.py`
3. **Run the integration tests** to see working examples
4. **Use the performance tools** to optimize your workflows

**Happy coding! ☕️** 