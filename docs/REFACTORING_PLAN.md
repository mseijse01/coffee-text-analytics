# 🔧 Coffee Text Analytics - Refactoring Plan

## 📊 Current Status Assessment

### ✅ **What's Working Well**
- **Core functionality**: Text preprocessing, data structures, basic ML operations
- **MNIR implementation**: Correctly follows thesis methodology
- **Polars integration**: Modern data processing capabilities
- **Comprehensive documentation**: Well-documented methodology and findings
- **Test coverage**: Extensive test suite created (75% basic functionality working)

### ⚠️ **Issues Identified**

#### 🔴 **High Priority Issues**
1. **Code Duplication**
   - `analyze_data_quality()` function duplicated in `loader.py` and `utils.py`
   - Similar preprocessing logic scattered across modules
   - Redundant import patterns

2. **Dependency Management**
   - Missing dependencies (pyarrow, plotly)
   - SSL certificate issues with NLTK downloads
   - Inconsistent dependency versions

#### 🟡 **Medium Priority Issues**
3. **Large Monolithic Classes**
   - `CoffeeFeatureExtractor` class has multiple responsibilities (SRP violation)
   - `model_training.py` file is large and handles multiple concerns
   - Mixed abstraction levels

4. **Configuration Management**
   - Hardcoded parameters throughout codebase
   - No centralized configuration system
   - Environment-specific settings scattered

5. **Error Handling**
   - Inconsistent error handling patterns
   - No custom exception classes
   - Silent failures in some components

#### 🟢 **Low Priority Issues**
6. **Data Type Consistency**
   - Mixed Polars/Pandas usage could be streamlined
   - Unnecessary conversions between data types
   - Type hints could be more comprehensive

## 🎯 **Refactoring Strategy**

### **Phase 1: Foundation Cleanup** (🔴 High Priority - 2-3 hours)

#### **1.1 Code Deduplication**
```python
# Before: Duplicated in loader.py and utils.py
def analyze_data_quality(df: pl.DataFrame) -> None:
    # ... implementation ...

# After: Single implementation in utils/data_quality.py
from utils.data_quality import analyze_data_quality
```

#### **1.2 Dependency Resolution**
- Add missing dependencies to `requirements.txt`
- Fix SSL certificate issues for NLTK
- Create `requirements-dev.txt` for development dependencies

#### **1.3 Import Standardization**
- Standardize import patterns across modules
- Remove unused imports
- Group imports consistently

### **Phase 2: Configuration Management** (🟡 Medium Priority - 3-4 hours)

#### **2.1 Centralized Configuration**
```python
# src/config/settings.py
class Config:
    # Data processing
    TFIDF_MAX_FEATURES = 5000
    BERT_MAX_LENGTH = 512
    N_TOPICS_DEFAULT = 10
    
    # Model training
    RANDOM_STATE = 42
    LASSO_CV_FOLDS = 5
    
    # Paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    OUTPUT_DIR = "output"
```

#### **2.2 Environment-Specific Configs**
- Development, testing, production configurations
- Environment variable support
- Configuration validation

### **Phase 3: Component Separation** (🟡 Medium Priority - 6-8 hours)

#### **3.1 Feature Extraction Refactoring**
```python
# Before: Monolithic CoffeeFeatureExtractor
class CoffeeFeatureExtractor:
    def extract_tfidf_features(self, ...): ...
    def extract_bert_embeddings(self, ...): ...
    def extract_glove_embeddings(self, ...): ...
    def extract_topic_features(self, ...): ...
    def extract_sentiment_features(self, ...): ...

# After: Specialized extractors
from features.extractors import (
    TfidfExtractor,
    BertExtractor, 
    GloveExtractor,
    TopicExtractor,
    SentimentExtractor
)

class FeatureExtractionPipeline:
    def __init__(self):
        self.extractors = {
            'tfidf': TfidfExtractor(),
            'bert': BertExtractor(),
            'glove': GloveExtractor(),
            'topics': TopicExtractor(),
            'sentiment': SentimentExtractor()
        }
```

#### **3.2 Model Training Separation**
```python
# Before: Large model_training.py
# After: Separate files
src/models/
├── __init__.py
├── base.py              # Abstract base classes
├── traditional.py       # Linear, Ridge, Lasso, RF
├── ensemble.py          # XGBoost, other ensemble methods
├── mnir.py             # MNIR implementation
├── evaluation.py       # Model evaluation utilities
└── training_pipeline.py # Training orchestration
```

#### **3.3 Abstract Base Classes**
```python
# src/models/base.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y): ...
    
    @abstractmethod
    def predict(self, X): ...
    
    @abstractmethod
    def get_feature_importance(self): ...

class BaseExtractor(ABC):
    @abstractmethod
    def extract_features(self, texts): ...
    
    @abstractmethod
    def get_feature_names(self): ...
```

### **Phase 4: Error Handling Standardization** (🟡 Medium Priority - 4-5 hours)

#### **4.1 Custom Exception Classes**
```python
# src/exceptions.py
class CoffeeAnalyticsError(Exception):
    """Base exception for coffee analytics project."""
    pass

class DataLoadingError(CoffeeAnalyticsError):
    """Raised when data loading fails."""
    pass

class FeatureExtractionError(CoffeeAnalyticsError):
    """Raised when feature extraction fails."""
    pass

class ModelTrainingError(CoffeeAnalyticsError):
    """Raised when model training fails."""
    pass
```

#### **4.2 Consistent Error Handling**
```python
# Before: Silent failures
try:
    result = some_operation()
except Exception:
    return None

# After: Explicit error handling
try:
    result = some_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise FeatureExtractionError(f"Failed to extract features: {e}")
```

### **Phase 5: Performance Optimization** (🟢 Low Priority - 5-6 hours)

#### **5.1 Data Type Optimization**
- Establish clear Polars-first strategy
- Minimize Polars ↔ Pandas conversions
- Use lazy evaluation where possible

#### **5.2 Caching Implementation**
```python
# src/utils/cache.py
from functools import lru_cache
import pickle

class FeatureCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
    
    def get_or_compute(self, key, compute_func, *args, **kwargs):
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            return pickle.load(open(cache_file, 'rb'))
        
        result = compute_func(*args, **kwargs)
        pickle.dump(result, open(cache_file, 'wb'))
        return result
```

## 📋 **Implementation Checklist**

### **Phase 1: Foundation Cleanup**
- [ ] Consolidate `analyze_data_quality()` functions
- [ ] Add missing dependencies to requirements.txt
- [ ] Fix NLTK SSL certificate issues
- [ ] Standardize import patterns
- [ ] Remove code duplication
- [ ] Run tests to verify no regressions

### **Phase 2: Configuration Management**
- [ ] Create `src/config/settings.py`
- [ ] Move hardcoded parameters to config
- [ ] Add environment variable support
- [ ] Update all modules to use centralized config
- [ ] Add configuration validation
- [ ] Run tests to verify functionality

### **Phase 3: Component Separation**
- [ ] Create abstract base classes
- [ ] Split `CoffeeFeatureExtractor` into specialized classes
- [ ] Separate model training by type
- [ ] Create `FeatureExtractionPipeline` orchestrator
- [ ] Update imports and dependencies
- [ ] Run comprehensive tests

### **Phase 4: Error Handling**
- [ ] Create custom exception classes
- [ ] Implement consistent error handling patterns
- [ ] Add comprehensive logging
- [ ] Update error messages for clarity
- [ ] Test error scenarios

### **Phase 5: Performance Optimization**
- [ ] Optimize data type conversions
- [ ] Implement caching for expensive operations
- [ ] Profile performance bottlenecks
- [ ] Add lazy evaluation where beneficial
- [ ] Benchmark before/after performance

## 🧪 **Testing Strategy**

### **Continuous Testing Approach**
1. **Run tests before each phase**: Establish baseline
2. **Run tests after each change**: Verify no regressions
3. **Add new tests for new components**: Maintain coverage
4. **Performance testing**: Ensure optimizations work
5. **Integration testing**: Verify end-to-end functionality

### **Test Commands**
```bash
# Basic functionality test
python test_basic.py

# Comprehensive test suite (after fixing dependencies)
python run_tests.py

# Specific component testing
python -m unittest tests.test_feature_extraction -v
```

## 📊 **Success Metrics**

### **Code Quality Metrics**
- [ ] **Reduced duplication**: No duplicate functions
- [ ] **Improved modularity**: Single responsibility per class
- [ ] **Better error handling**: Comprehensive exception handling
- [ ] **Centralized configuration**: All parameters in config

### **Performance Metrics**
- [ ] **Faster feature extraction**: Benchmark improvements
- [ ] **Reduced memory usage**: Profile memory consumption
- [ ] **Better caching**: Avoid redundant computations

### **Maintainability Metrics**
- [ ] **Test coverage**: Maintain >90% coverage
- [ ] **Documentation**: Update docs for new structure
- [ ] **Type hints**: Add comprehensive type annotations
- [ ] **Logging**: Comprehensive logging throughout

## 🎯 **Expected Outcomes**

### **After Refactoring**
1. **Cleaner codebase**: Reduced duplication, better organization
2. **Easier maintenance**: Modular components, clear interfaces
3. **Better performance**: Optimized data processing, caching
4. **Improved reliability**: Comprehensive error handling, testing
5. **Enhanced extensibility**: Easy to add new features/models

### **Risk Mitigation**
- **Incremental approach**: Small, testable changes
- **Comprehensive testing**: Verify functionality at each step
- **Backup strategy**: Git branches for each phase
- **Rollback plan**: Ability to revert if issues arise

## 💡 **Next Steps**

1. **Fix basic dependencies** (pyarrow, plotly)
2. **Start with Phase 1** (code deduplication)
3. **Run tests after each change**
4. **Document changes as you go**
5. **Get feedback on each phase before proceeding**

---

**Total Estimated Time**: 20-25 hours
**Recommended Timeline**: 1-2 weeks (incremental approach)
**Risk Level**: 🟡 Medium (mitigated by testing and incremental approach) 