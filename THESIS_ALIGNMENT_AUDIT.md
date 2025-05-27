# 🔍 Thesis Alignment Audit & Implementation Plan

## 🚨 Critical Issues Identified

### 1. **Text Column Handling Mismatch** 

#### ❌ **Current Implementation Issues**
- **Using combined text**: We're combining `desc_1`, `desc_2`, `desc_3` into a single text for feature extraction
- **Missing separate processing**: Thesis explicitly states using the three description fields **separately**
- **Wrong approach**: We merge text columns then extract features, losing the distinct semantic content of each description type

#### ✅ **Thesis Methodology (Expected)**
From thesis (page 234-279):
- **desc_1**: Tasting notes (specific flavor descriptors)
- **desc_2**: Contextual information (brewing, origin details)  
- **desc_3**: Reviewer's conclusion or "bottom line" (overall assessment)
- **Separate processing**: "By using them separately, we aimed to capture the different topics each description field focuses on"
- **TF-IDF per column**: "For each description column, the TF-IDF vectorizer produced a sparse matrix"

#### 🔧 **Required Fix**
- Extract TF-IDF features **separately** for each desc column
- Extract BERT embeddings **separately** for each desc column
- Extract GloVe embeddings **separately** for each desc column
- Apply sentiment analysis **separately** for each desc column
- Apply topic modeling **separately** for each desc column
- Result: `tfidf_desc_1_0`, `tfidf_desc_2_0`, `tfidf_desc_3_0`, etc.

### 2. **LASSO Feature Selection Methodology Mismatch**

#### ❌ **Current Implementation Issues**
- **Group-wise selection**: We're doing independent LASSO per feature group, but thesis shows LASSO applied to **combined feature sets**
- **Performance expectation**: Ridge (R² = 0.7739) outperforming on LASSO-selected features contradicts thesis methodology
- **Feature selection scope**: Thesis shows LASSO selecting from **all text features combined**, not group-wise

#### ✅ **Thesis Methodology (Expected)**
From thesis Table (page 580-590):
- **All Text Features + LASSO**: XGBoost R² = 0.683, Random Forest R² = 0.678
- **Flavor Features**: Linear R² = 0.998, SVR R² = 0.998, Random Forest R² = 0.995
- **LASSO should improve XGBoost performance**, not make Ridge the best model

### 2. **Model Performance Hierarchy Mismatch**

#### ❌ **Current Results (5% sample)**
```
Ridge:         R² = 0.7739  ← Best (WRONG!)
SVR:           R² = 0.7717
Linear:        R² = 0.6916
Lasso:         R² = 0.6405
Random Forest: R² = 0.2811  ← Should be much higher
XGBoost:       R² = 0.1427  ← Should be THE BEST
```

#### ✅ **Expected Results (from thesis)**
```
XGBoost:       R² = 0.683   ← Should be BEST
Random Forest: R² = 0.678   ← Should be second
Linear:        R² = 0.632   ← Baseline
SVR:           R² = 0.569   ← Lower than ensemble methods
```

### 3. **Data Column Usage Mismatch**

#### ❌ **Current Implementation Issues**
- **Missing all_text column**: We have `all_text` in raw data but not using it properly
- **Ignoring thesis data decisions**: Thesis explicitly mentions moving from `all_text` to separate `desc_1`, `desc_2`, `desc_3`
- **Column exclusion logic**: We're excluding `all_text` in cleaning but should be using separate desc columns

#### ✅ **Thesis Data Structure (Expected)**
From thesis data description:
- **all_text**: "Concatenated text of desc_1, desc_2 and desc_3" (initially used, then replaced)
- **desc_1**: Tasting notes (primary flavor descriptors)
- **desc_2**: Contextual information (brewing methods, origin details)
- **desc_3**: Reviewer's conclusion or "bottom line" assessment
- **Decision**: "use the three separate description fields instead" for better topic capture

### 4. **Feature Selection Strategy Issues**

#### ❌ **Current Approach**
- Group-wise LASSO (TF-IDF, BERT, topics separately)
- 94.9% reduction (4,797 → 244 features)
- Different alpha per group

#### ✅ **Thesis Approach**
- LASSO applied to **combined text features**
- Feature selection **before** model training
- Single optimal alpha for all text features
- Focus on **most predictive text features** across all types

### 5. **Text Preprocessing Pipeline Mismatch**

#### ❌ **Current Implementation Issues**
- **Single preprocessing pipeline**: We use one preprocessing approach for all models
- **Missing specialized pipelines**: No separate preprocessing for topic modeling vs embeddings
- **Stopword handling**: Not following thesis-specific requirements per model type

#### ✅ **Thesis Methodology (Expected)**
From thesis (page 270-290):
- **Two distinct preprocessing pipelines** based on model requirements:
  1. **For embeddings/sentiment**: Remove stopwords, retain punctuation
  2. **For topic modeling**: Retain stopwords, remove punctuation
- **Three separate datasets created**: embeddings, topic modeling, sentiment analysis
- **Explicit processing flags** applied per task

### 6. **Box-Cox Transformation Missing**

#### ❌ **Current Implementation Issues**
- **No Box-Cox transformation**: We don't apply Box-Cox to normalize target variable distribution
- **Missing skewness handling**: Thesis mentions target variable skewness was addressed

#### ✅ **Thesis Methodology (Expected)**
From thesis (page 200-210):
- **Box-Cox transformation applied** to coffee ratings due to skewness
- **Normalization for linear regression**: Ensures normally distributed residuals
- **Later discarded**: "transformation was ultimately discarded, as it did not improve the models tested"
- **Decision documented**: Should implement and test, then document decision

### 7. **Feature Naming Convention Mismatch**

#### ❌ **Current Implementation Issues**
- **Generic naming**: Features named `tfidf_0`, `bert_0`, `glove_0`
- **No column identification**: Can't tell which desc column features came from
- **Missing thesis naming**: Not following thesis feature naming convention

#### ✅ **Thesis Methodology (Expected)**
From thesis (page 279, 520):
- **Column-specific naming**: `tfidf_desc_1_0`, `bert_desc_1_0`, `glove_desc_1_0`
- **Clear source identification**: Each feature name indicates source column
- **Examples in results**: `tfidf_processed_desc_1_641`, `tfidf_processed_desc_1_2452`

### 8. **TF-IDF Configuration Mismatch**

#### ❌ **Current Implementation Issues**
- **Max features**: We use 5000, but apply to combined text
- **Per-column application**: Not extracting TF-IDF separately per desc column

#### ✅ **Thesis Methodology (Expected)**
From thesis (page 279):
- **5000 terms per column**: "vectorizer was configured to limit the feature space to 5000 terms"
- **Separate matrices**: "For each description column, the TF-IDF vectorizer produced a sparse matrix"
- **Column-specific features**: Each desc column gets its own 5000-feature TF-IDF space

### 9. **SHAP Analysis Implementation**

#### ✅ **Current Implementation (GOOD)**
- **SHAP available**: We have SHAP implementation in MNIR and evaluator
- **Feature importance**: SHAP values computed for interpretability
- **Thesis alignment**: Matches thesis extensive use of SHAP analysis

#### ⚠️ **Potential Enhancement**
- **Broader application**: Could extend SHAP to all models, not just MNIR
- **Visualization**: Could add SHAP summary plots as shown in thesis

### 10. **MNIR Implementation**

#### ✅ **Current Implementation (GOOD)**
- **MNIR available**: We have complete MNIR implementation
- **Sensory prediction**: Predicts acidity, body, aroma, aftertaste, flavor
- **SHAP integration**: Includes SHAP analysis for interpretability
- **Thesis alignment**: Matches thesis MNIR methodology exactly

## 📋 Implementation Plan

### Phase 1: Fix Text Column Processing (CRITICAL)

#### 1.1 **Implement Separate Description Column Processing** ✅ COMPLETED
- [x] **Modify feature extractors** to process `desc_1`, `desc_2`, `desc_3` separately ✅
- [x] **Update TF-IDF extraction** to create separate vectorizers per column ✅
- [x] **Update BERT embedding** to extract embeddings per column separately ✅
- [x] **Update GloVe embedding** to extract embeddings per column separately ✅
- [x] **Update sentiment analysis** to analyze each column separately ✅
- [x] **Update topic modeling** to model topics per column separately ✅

#### 1.2 **Update Feature Naming Convention** ✅ COMPLETED
- [x] Change from generic `tfidf_0`, `bert_0` to `tfidf_desc_1_0`, `bert_desc_1_0` ✅
- [x] Ensure feature names reflect source column: `glove_desc_2_0`, `sentiment_desc_3_pos` ✅
- [x] Update feature selection to handle column-specific features ✅

#### 1.3 **Validate Separate Processing**
- [ ] Verify each desc column produces distinct feature sets
- [ ] Check feature counts per column match thesis expectations
- [ ] Ensure no information loss from separate processing

### Phase 2: Fix LASSO Feature Selection Methodology

#### 2.1 **Correct LASSO Application**
- [ ] Apply LASSO to **combined text features** (not group-wise)
- [ ] Use single alpha optimization across all text features
- [ ] Maintain sensory features (aroma, acid, body, flavor, aftertaste) separately
- [ ] Keep categorical features (origin, roast) separately

#### 2.2 **Feature Selection Pipeline**
```python
# Correct approach:
# 1. Combine all text features from all 3 desc columns (TF-IDF + BERT + GloVe + topics + sentiment)
# 2. Apply single LASSO with CV to select best text features
# 3. Combine selected text features with sensory + categorical
# 4. Train models on this combined feature set
```

### Phase 3: Fix Text Preprocessing Pipelines

#### 3.1 **Implement Specialized Preprocessing**
- [ ] **Create embedding/sentiment pipeline**: Remove stopwords, retain punctuation
- [ ] **Create topic modeling pipeline**: Retain stopwords, remove punctuation
- [ ] **Generate three datasets**: embeddings, topic modeling, sentiment analysis
- [ ] **Apply appropriate pipeline** per feature extraction task

#### 3.2 **Box-Cox Transformation Implementation** ✅ COMPLETED
- [x] **Implement Box-Cox transformation** for target variable normalization ✅
- [ ] **Test impact on model performance** across all models
- [ ] **Document decision** to keep or discard (following thesis approach)
- [ ] **Handle skewness** in coffee ratings distribution

### Phase 4: Fix Model Performance and Hyperparameters

#### 4.1 **Expected Outcome After All Fixes**
- XGBoost should achieve R² ≈ 0.68-0.70 (best performance)
- Random Forest should achieve R² ≈ 0.65-0.68 (second best)
- Linear models should be baseline (R² ≈ 0.60-0.65)

### Phase 2: Fix Model Training and Evaluation

#### 2.1 **Hyperparameter Tuning Issues**
- [ ] XGBoost and Random Forest showing poor performance suggests overfitting on small sample
- [ ] Need proper hyperparameter tuning for small datasets
- [ ] Reduce model complexity for 5% sample size

#### 2.2 **Sample Size Considerations**
- [ ] 5% sample (122 samples) too small for complex models
- [ ] Tree-based models overfitting severely
- [ ] Need larger sample for proper validation (at least 10-20%)

#### 2.3 **Cross-Validation Strategy**
- [ ] Current 5-fold CV with 122 samples = ~24 samples per fold (too small)
- [ ] Reduce to 3-fold CV or use larger sample
- [ ] Stratified sampling working correctly

### Phase 3: Align with Thesis Feature Groups

#### 3.1 **Feature Group Strategy**
From thesis methodology:
- **Flavor Features**: aroma, acid, body, flavor, aftertaste (keep separate)
- **Categorical Features**: origin, roast, roaster (keep separate)  
- **Text Features**: ALL text-derived features combined for LASSO selection

#### 3.2 **LASSO Selection Process**
```python
# Thesis-aligned approach:
1. Extract all text features (~6000 features)
2. Apply LASSO with CV to select ~500-1000 most predictive
3. Combine with flavor features (5 features)
4. Combine with categorical features (~10-20 features)
5. Final feature set: ~500-1000 features total
6. Train models on this combined set
```

### Phase 4: Validation and Testing

#### 4.1 **Performance Validation**
- [ ] Test with larger sample (10-20%) to validate model hierarchy
- [ ] Ensure XGBoost > Random Forest > Linear models
- [ ] Validate feature importance aligns with thesis findings

#### 4.2 **Feature Importance Validation**
From thesis findings:
- **Text features** should dominate importance
- **Acidity** should be most important sensory feature
- **Origin** should be most important categorical feature

## 🔧 Technical Implementation Tasks

### Task 1: Refactor LASSO Feature Selector

#### Current Issues:
```python
# Wrong: Group-wise selection
groups = ["tfidf", "bert", "topics_lda", "topics_nmf", "sentiment", "glove"]
for group in groups:
    select_features_for_group(group)
```

#### Correct Implementation:
```python
# Right: Combined text feature selection
text_features = combine_all_text_features()  # ~6000 features
selected_text_features = lasso_cv_select(text_features)  # ~500-1000 features
final_features = selected_text_features + sensory_features + categorical_features
```

### Task 2: Fix Model Configuration

#### XGBoost Configuration for Small Samples:
```python
xgboost_params = {
    'n_estimators': 50,      # Reduce from default 100
    'max_depth': 3,          # Reduce from default 6
    'learning_rate': 0.1,    # Conservative learning rate
    'subsample': 0.8,        # Add regularization
    'colsample_bytree': 0.8, # Add regularization
}
```

#### Random Forest Configuration for Small Samples:
```python
rf_params = {
    'n_estimators': 50,      # Reduce from default 100
    'max_depth': 5,          # Limit depth
    'min_samples_split': 5,  # Increase minimum split
    'min_samples_leaf': 2,   # Increase minimum leaf
}
```

### Task 3: Update Evaluation Strategy

#### Sample Size Strategy:
- **Development**: Use 10-20% sample for proper validation
- **Testing**: Use 5% for quick iteration
- **Final validation**: Use full dataset

#### Cross-Validation Strategy:
- **Small samples (<500)**: 3-fold CV
- **Medium samples (500-2000)**: 5-fold CV
- **Large samples (>2000)**: 5-fold CV

## 🎯 Success Criteria

### Performance Targets (10-20% sample):
- **XGBoost**: R² > 0.65 (best performer)
- **Random Forest**: R² > 0.60 (second best)
- **Linear Regression**: R² > 0.55 (baseline)
- **Ridge/Lasso**: R² > 0.55 (regularized baselines)

### Feature Selection Targets:
- **Text features**: ~500-1000 selected from ~6000
- **Total features**: ~500-1020 (text + sensory + categorical)
- **Reduction ratio**: ~85-90% (not 95%+)

### Feature Importance Alignment:
- **Text features**: Dominate importance rankings
- **Acidity**: Top sensory feature
- **Origin**: Top categorical feature
- **BERT/LDA topics**: High importance among text features

## 📅 Implementation Timeline

### Week 1: Core Fixes
- [ ] Day 1-2: Refactor LASSO feature selector
- [ ] Day 3-4: Fix model configurations
- [ ] Day 5: Test with 10% sample

### Week 2: Validation
- [ ] Day 1-2: Full pipeline testing
- [ ] Day 3-4: Performance validation
- [ ] Day 5: Documentation updates

## 🚨 Priority Actions

### Immediate (Today):
1. **Fix text column processing** - implement separate desc_1, desc_2, desc_3 processing (CRITICAL)
2. **Fix feature naming convention** - use `tfidf_desc_1_0` format (CRITICAL)
3. **Fix TF-IDF configuration** - 5000 features per desc column, not combined (CRITICAL)
4. **Implement specialized preprocessing** - separate pipelines for embeddings vs topic modeling
5. **Fix LASSO methodology** - combine text features before selection  

### This Week:
1. **Implement Box-Cox transformation** - test and document decision
2. **Test with 10% sample** - get more reliable model performance  
3. **Reduce model complexity** - prevent overfitting on small samples
4. **Validate separate column processing** - ensure distinct feature extraction per column

### Next Steps:
1. **Validate performance hierarchy** - ensure XGBoost > Random Forest
2. **Check feature importance** - ensure text features dominate
3. **Extend SHAP analysis** - apply to all models, not just MNIR
4. **Update documentation** - reflect correct methodology

## 📊 Expected Outcomes

After fixes, we should see:
- **XGBoost as best performer** (R² > 0.65)
- **Random Forest as strong second** (R² > 0.60)
- **Text features dominating** importance rankings
- **Sensible feature reduction** (~85-90%, not 95%+)
- **Thesis-aligned results** matching expected performance hierarchy

---

**Status**: Ready for implementation  
**Priority**: CRITICAL - Core methodology alignment  
**Timeline**: 1-2 weeks for complete fix 