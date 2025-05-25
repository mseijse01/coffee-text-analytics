# Implementation Fix Plan
## Coffee Text Analytics - Fixing Core Implementation Issues

### **Overview**
This plan addresses the core implementation inconsistencies identified during testing, focusing on data type consistency, function behavior standardization, and proper documentation.

## **Priority 1: Critical Data Flow Issues** 🔴

### **Issue 1.1: Multiple `load_coffee_data` Functions**
**Problem**: Three different functions with same name but different behaviors:
- `src/data/loader.py`: `() -> pl.DataFrame` (config-based path)
- `src/utils/utils.py`: `() -> pl.DataFrame` (hardcoded path) 
- `src/data/preprocessing.py`: `(file_path: str) -> pd.DataFrame` (file parameter)

**Root Cause**: The preprocessing function was designed for pandas-based text operations, while the main pipeline uses Polars.

**Solution**: 
1. **Keep preprocessing function as pandas** (text operations are easier in pandas)
2. **Rename functions** to clarify their purpose:
   - `load_coffee_data()` → `load_main_dataset()` (loader.py)
   - `load_coffee_data()` → `load_dataset_from_utils()` (utils.py) 
   - `load_coffee_data(file_path)` → `load_csv_for_preprocessing(file_path)` (preprocessing.py)
3. **Add conversion utilities** for seamless pandas ↔ polars conversion

### **Issue 1.2: Return Type Inconsistency in `clean_dataset`**
**Problem**: `clean_dataset()` returns `(pl.DataFrame, dict)` but some callers expect just `pl.DataFrame`

**Solution**: 
- **Standardize on tuple return** - statistics are valuable for data quality monitoring
- **Update all callers** to handle tuple unpacking
- **Add clear docstring** explaining return format

### **Issue 1.3: Country Extraction Logic**
**Problem**: `extract_country_info()` doesn't properly extract countries from origin strings like "Ethiopia Yirgacheffe"

**Solution**: Fix the logic to prioritize known countries at string start

## **Priority 2: Function Behavior Standardization** 🟡

### **Issue 2.1: Text Cleaning Inconsistency**
**Problem**: `clean_text()` keeps punctuation but tests expect removal

**Solution**: 
- **Add parameter** `remove_punctuation: bool = True`
- **Update docstring** to clarify behavior
- **Ensure consistent behavior** across all text processing

### **Issue 2.2: Missing Docstrings and Type Hints**
**Problem**: Many functions lack clear documentation about pandas vs polars usage

**Solution**: Add comprehensive docstrings with:
- Clear parameter types (pd.DataFrame vs pl.DataFrame)
- Return type specifications
- Usage examples for complex functions
- Rationale for pandas vs polars choice

## **Priority 3: Data Type Strategy** 🟢

### **Strategy: Hybrid Pandas/Polars Approach**
**Rationale**: 
- **Polars**: Main pipeline, feature extraction, large data operations
- **Pandas**: Text preprocessing, complex string operations, compatibility with sklearn

**Implementation**:
1. **Clear boundaries**: Document which functions expect which type
2. **Conversion utilities**: Easy conversion between types
3. **Type hints**: Explicit type annotations everywhere
4. **Docstring standards**: Always specify expected DataFrame type

## **Detailed Implementation Steps**

### **Step 1: Fix `load_coffee_data` Functions**

```python
# src/data/loader.py
def load_main_dataset() -> pl.DataFrame:
    """
    Load the main coffee review dataset for analysis pipeline.
    
    Returns:
        pl.DataFrame: Main dataset optimized for Polars operations
    """

# src/data/preprocessing.py  
def load_csv_for_preprocessing(file_path: str) -> pd.DataFrame:
    """
    Load CSV data for text preprocessing operations.
    
    Uses pandas for easier text manipulation and sklearn compatibility.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        pd.DataFrame: Data optimized for text preprocessing
    """

# New utility function
def convert_pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas DataFrame to Polars with proper type handling."""
    
def convert_polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars DataFrame to pandas with proper type handling."""
```

### **Step 2: Fix `clean_dataset` Return Handling**

```python
# src/utils/cleaning.py
def clean_dataset(df: pl.DataFrame, min_rating: float = 80.0) -> tuple[pl.DataFrame, dict]:
    """
    Clean dataset by handling missing values and outliers.
    
    Args:
        df: Input Polars DataFrame
        min_rating: Minimum rating threshold
        
    Returns:
        tuple[pl.DataFrame, dict]: (cleaned_data, cleaning_statistics)
            - cleaned_data: Filtered and cleaned DataFrame
            - cleaning_statistics: Dict with removal counts and percentages
    """
```

### **Step 3: Fix Country Extraction**

```python
# src/data/preprocessing.py
def extract_country_info(location: str) -> str:
    """
    Extract country name from location string.
    
    Prioritizes known coffee-producing countries that appear at string start.
    
    Args:
        location: Location string (e.g., "Ethiopia Yirgacheffe")
        
    Returns:
        str: Extracted country name (e.g., "Ethiopia")
        
    Examples:
        >>> extract_country_info("Ethiopia Yirgacheffe")
        "Ethiopia"
        >>> extract_country_info("Colombia Huila")
        "Colombia"
    """
```

### **Step 4: Standardize Text Processing**

```python
# src/data/preprocessing.py
def clean_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Clean text by removing HTML tags, URLs, and optionally punctuation.
    
    Args:
        text: Input text
        remove_punctuation: Whether to remove punctuation marks
        
    Returns:
        str: Cleaned text
    """
```

## **Testing Strategy**

### **Test Updates Required**:
1. **Update import statements** to use renamed functions
2. **Fix return type expectations** to match implementation
3. **Add conversion tests** for pandas ↔ polars operations
4. **Test edge cases** in country extraction

### **New Tests to Add**:
1. **Data type conversion tests**
2. **Function behavior consistency tests**
3. **Documentation compliance tests**

## **Migration Path**

### **Phase 1**: Fix Critical Issues (2-3 hours)
1. Rename `load_coffee_data` functions
2. Fix `clean_dataset` return handling
3. Fix country extraction logic
4. Update imports in tests

### **Phase 2**: Standardize Documentation (1-2 hours)
1. Add comprehensive docstrings
2. Add type hints everywhere
3. Document pandas vs polars strategy

### **Phase 3**: Add Conversion Utilities (1 hour)
1. Create conversion functions
2. Add conversion tests
3. Update integration points

## **Success Criteria**
- ✅ All tests pass with correct implementation
- ✅ Clear data type boundaries documented
- ✅ No function name conflicts
- ✅ Consistent return types
- ✅ Proper country extraction behavior
- ✅ Clear pandas vs polars usage strategy 