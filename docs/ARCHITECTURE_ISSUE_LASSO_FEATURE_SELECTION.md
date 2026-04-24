# Architecture Issue: LASSO Feature Selection Breaks MNIR and Model Validation

**Created**: 2026-04-21
**Status**: Documented, Awaiting Architectural Redesign
**Priority**: High (blocks model quality validation)
**Assigned to**: Sonnet (future session)

## Executive Summary

The current LASSO feature selection architecture applies to **all features simultaneously** (text + sensory + categorical), which:
1. **Breaks MNIR**: Sensory columns (`aroma`, `acid`, `body`, `flavor`, `aftertaste`) exist in the selected feature matrix but are anonymized → MNIR can't locate them by name
2. **Prevents proper R² validation**: Linear models achieve R²=1.0 even with proper train/test split, suggesting hidden leakage or overfitting
3. **Creates redundant encoding**: Both original `roast`/`origin` AND encoded `roast_*`/`origin_*` features exist simultaneously

**Root cause**: LassoFeatureSelector treats all 2,940 text features as one block for selection, then outputs an anonymous 172-column feature matrix. Sensory and categorical features are lost/anonymized in the process.

---

## Problem Statement

### What We Observed

**Validation run (30% sample, 732 rows)**:
```
Train: 585 rows, 172 features
Test: 147 rows, 172 features

Results:
- Linear R²: 1.0000 ❌ (suspicious)
- Ridge R²: 1.0000 ❌ (suspicious)
- LASSO R²: 1.0000 ❌ (suspicious)
- RandomForest R²: 0.9968 ✓ (realistic)
- XGBoost R²: 0.9951 ✓ (realistic)
- MNIR R²: 0.0000 ❌ (no sensory columns found)

Warnings:
- MNIR: "Sensory attribute 'aroma' not found in data, skipping"
- MNIR: "Sensory attribute 'acid' not found in data, skipping"
(all 5 sensory attributes skipped)
```

**Key finding**: Tree-based models work correctly (0.995+), linear models overfit (1.0), MNIR fails.

### Root Cause Analysis

**File**: `src/features/feature_selector.py` (LassoFeatureSelector class)

**Current flow**:
1. Input: X (732 rows × 3,012 columns) = 2,940 text + 6 sensory + 66 categorical
2. `fit_select_features()` calls `_identify_feature_types()` to categorize columns
3. LASSO is fitted ONLY on text features (2,940 columns)
4. `transform()` returns selected features: 100 text + 6 sensory + 66 categorical = 172 columns
5. **Problem**: Output matrix is unnamed — column names lost
6. Result: Downstream code can't access sensory columns by name

**Code location**:
- `src/features/feature_selector.py:310-354` — `transform()` method
- `src/features/feature_selector.py:93-159` — `_identify_feature_types()` method

**Why it fails**:
```python
# Line 345 in feature_selector.py
X_selected = X_df[available_features]  # ← Returns DataFrame with correct columns
# BUT in validation scripts:
X_selected = selector.transform(X)
# This X_selected has columns like "tfidf_0", "bert_1", etc. (not "aroma", "acid")
```

The sensory features ARE selected (they pass through the selector), but they're named `tfidf_*` or `bert_*` in the original matrix, not `aroma`/`acid`. So they exist but can't be found by name.

---

## Impact

### 1. MNIR Model Can't Train
**File**: `src/models/mnir.py:49-50`
```python
sensory_cols = ["aroma", "acid", "body", "flavor", "aftertaste"]
available_sensory = [col for col in sensory_cols if col in processed_df.columns]
```

After LASSO selection, these column names don't exist in the feature matrix. MNIR skips with warnings.

### 2. R² = 1.0 for Linear Models
**Hypothesis**:
- Tree-based models handle anonymous features fine (they just use position)
- Linear models might be overfitting to sparse high-dimensional noise
- Or there's subtle data leakage in how features are engineered

**Validation approach**:
- 585 training samples, 172 features → 3.4 samples/feature
- Linear regression without regularization can overfit easily
- Need to investigate whether R²=1.0 is real signal or artifact

### 3. Double Categorical Encoding
In `validate_*_percent_and_save.py`:
```python
# Line: Convert categorical columns to numeric
for col in ["origin", "roast"]:
    if col in X.columns:
        X[col] = pd.Categorical(X[col]).codes  # ← Original column

# Later: CategoricalFeatureEncoder produces
# roast_Dark, roast_Light, roast_Medium, ... (one-hot encoded)
# origin is already converted to numeric codes

# Result: Both original AND encoded versions exist
```

This creates redundancy and confusion about which representation to use.

---

## Thesis Methodology Context

From CLAUDE.md and thesis methodology:
```
LASSO feature selection should:
1. Combine ALL text features from all desc columns
2. Apply single LASSO with CV to select best text features
3. Keep sensory and categorical features separate
4. Final feature set: selected_text + sensory + categorical
```

**Current state violates step 3**: All features are mixed together, then selected.

---

## Proposed Solution (Architectural)

### Option 1: Preserve Sensory Column Names (Recommended)

**Approach**: Refactor `LassoFeatureSelector` to:
1. Take X as DataFrame with named columns
2. During `fit_select_features()`:
   - Identify text, sensory, categorical columns
   - Fit LASSO ONLY on text columns
   - Store selected text feature indices
3. During `transform()`:
   - Select text features by stored indices
   - Preserve sensory column names exactly
   - Preserve categorical column names exactly
4. Output: Named DataFrame with 100 selected_text + 6 sensory + 66 categorical

**Why**: MNIR can then access sensory columns by name.

**Implementation files to modify**:
- `src/features/feature_selector.py` — refactor `fit_select_features()` and `transform()` to work with named columns
- `validate_15_percent_and_save.py` — already passes X as DataFrame; no changes needed
- `validate_30_percent_and_save.py` — same
- `src/pipeline/selection.py` — verify compatibility with main.py pipeline

### Option 2: Separate Pipeline for Sensory Features

**Approach**:
1. LASSO selects only text features (100 features)
2. Sensory features added separately, untouched
3. Categorical features added separately, untouched
4. Models receive: 100 text + 6 sensory + 66 categorical

**Why**: Cleaner separation of concerns.

**Tradeoff**: More refactoring, but better architecture long-term.

### Option 3: Add Sensory Reconstruction Logic to MNIR

**Approach**:
1. MNIR introspects the feature extraction pipeline
2. Reconstructs sensory columns from known locations
3. Doesn't require LassoFeatureSelector changes

**Why**: Quick fix, minimal refactoring.

**Tradeoff**: Fragile, violates single responsibility principle.

---

## Double Categorical Encoding Issue

**Problem code** (validate_*_percent_and_save.py, lines ~105-145):
```python
# Drop metadata columns (including original roast, origin, roaster)
metadata_cols = ["slug", "roaster", "name", "location", "review_date", "with_milk", "est_price", "agtron"]
metadata_to_drop = [col for col in metadata_cols if col in X.columns]
columns_to_drop.extend(metadata_to_drop)

# BUT THEN:
# Convert categorical columns to numeric (line ~140)
for col in ["origin", "roast"]:
    if col in X.columns:
        X[col] = pd.Categorical(X[col]).codes
```

**Issue**:
1. We're dropping `roaster` (metadata) but then converting `roast`/`origin` to numeric codes
2. CategoricalFeatureEncoder already created `roast_*` and `origin_*` one-hot features
3. Result: Both code-converted AND one-hot versions exist

**Fix**:
- Drop BOTH original AND encoded versions, or
- Keep ONLY encoded versions (no numeric code conversion)
- Or keep ONLY original (remove encoder output)

**Recommended**: Drop original categorical columns, keep only CategoricalFeatureEncoder output (one-hot/frequency-grouped).

---

## Validation Tests

### Test 1: MNIR Can Access Sensory Columns
```python
# After feature selection
assert "aroma" in X_selected.columns
assert "acid" in X_selected.columns
assert "body" in X_selected.columns
assert "flavor" in X_selected.columns
assert "aftertaste" in X_selected.columns
```

### Test 2: Linear R² is Reasonable
```python
# Run validation on 30% sample
# Linear R² should be < 0.95, not 1.0
# Or if R²=1.0, it indicates real structure (validate on 50%+ sample)
assert linear_r2 < 0.95 or "very_strong_signal"
```

### Test 3: No Double Categorical Encoding
```python
# After feature selection
original_categorical = [col for col in X_selected.columns if col in ["roast", "origin", "roaster"]]
encoded_categorical = [col for col in X_selected.columns if "_roast_" in col or "_origin_" in col or "_roaster_" in col]

# Should have either original OR encoded, not both
assert len(original_categorical) == 0 or len(encoded_categorical) == 0
```

### Test 4: MNIR Trains Successfully
```python
# After fix
mnir = MultinomialInverseRegression({})
mnir.fit(X_train, y_train)
assert mnir.performance_metrics is not None
assert len(mnir.performance_metrics) > 0  # Should have trained on some sensory attrs
```

---

## Files to Review Before Implementation

1. **src/features/feature_selector.py** (LassoFeatureSelector)
   - Lines 310-354: `transform()` method
   - Lines 93-159: `_identify_feature_types()` method
   - Lines 161-308: `fit_select_features()` method

2. **src/models/mnir.py** (MultinomialInverseRegression)
   - Lines 40-60: How it tries to access sensory columns
   - Determine exact expected interface

3. **validate_15_percent_and_save.py** and **validate_30_percent_and_save.py**
   - Understand current feature preparation flow
   - These scripts work with LassoFeatureSelector directly

4. **src/pipeline/selection.py** (main.py integration)
   - Verify how main.py uses LassoFeatureSelector
   - Ensure changes don't break main.py pipeline

---

## Recommended Starting Point for Sonnet

1. **Quick win**: Fix double categorical encoding in validation scripts (lines 140-145)
   - Remove numeric code conversion for origin/roast
   - Trust CategoricalFeatureEncoder's one-hot output

2. **Core fix**: Refactor `LassoFeatureSelector.transform()` to preserve column names
   - Keep text columns as selected
   - Preserve sensory columns as-is (don't anonymize)
   - Preserve categorical columns as-is

3. **Verification**: Run validation on 30% sample, verify:
   - MNIR trains successfully
   - Linear R² < 0.95 (or document why 1.0 is real)
   - No double categorical encoding

4. **Integration**: Test with main.py pipeline to ensure backward compatibility

---

## Decision Points for Sonnet

- **Option 1 (preserve names) vs Option 2 (separate pipeline)**: Which is cleaner?
- **Backward compatibility**: Should changes to LassoFeatureSelector be backward-compatible with main.py?
- **Test coverage**: Should unit tests be added to LassoFeatureSelector?
- **Documentation**: Update CLAUDE.md to document feature selection methodology?

---

## References

- **Thesis methodology**: CLAUDE.md, section "Key Design Decisions"
- **Feature selection logic**: `src/features/feature_selector.py`
- **MNIR implementation**: `src/models/mnir.py`
- **Validation scripts**: `validate_15_percent_and_save.py`, `validate_30_percent_and_save.py`
- **Main pipeline**: `src/pipeline/selection.py`, `main.py`
