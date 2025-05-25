# Coffee Text Analytics: Detailed Methodology

This document provides a comprehensive overview of the methodology implemented in this project, based on the thesis **"Leveraging Text Analytics and Predictive Modeling to Analyze Consumer Coffee Reviews: A Data-Driven Approach"** by Marcelo Seijas.

## 📋 Table of Contents

1. [Research Framework](#research-framework)
2. [Data Collection & Preprocessing](#data-collection--preprocessing)
3. [Feature Extraction Pipeline](#feature-extraction-pipeline)
4. [Model Development](#model-development)
5. [Evaluation Methodology](#evaluation-methodology)
6. [Implementation Details](#implementation-details)

## 🎯 Research Framework

### Theoretical Foundation

The methodology is grounded in several key theoretical frameworks:

1. **Text Analytics Theory**: Leveraging computational linguistics for extracting meaningful patterns from unstructured text
2. **Consumer Behavior Analysis**: Understanding how textual expressions relate to consumer preferences and ratings
3. **Multi-Modal Learning**: Combining different types of features (text, categorical, numerical) for improved prediction

### Research Hypothesis

> **Primary Hypothesis**: Text-based features derived from coffee reviews contain more predictive power for rating prediction than traditional sensory attributes alone.

**Sub-hypotheses**:
- H1: BERT embeddings capture semantic nuances better than traditional TF-IDF
- H2: Topic modeling reveals interpretable themes that correlate with ratings
- H3: Sentiment analysis provides additional predictive signal beyond content analysis
- H4: Multi-modal feature fusion outperforms single-modality approaches

## 📊 Data Collection & Preprocessing

### Dataset Characteristics

**Source**: CoffeeReview.com - Professional coffee review platform
**Time Period**: Multi-year collection of professional reviews
**Sample Size**: Comprehensive dataset of coffee reviews with ratings

**Data Structure**:
```
Original Dataset Schema:
├── Text Features (Primary Analysis)
│   ├── desc_1: Primary review description
│   ├── desc_2: Secondary review notes
│   └── desc_3: Additional tasting notes
├── Target Variable
│   └── rating: Coffee rating (0-100 scale)
├── Categorical Features
│   ├── origin: Coffee origin/country
│   ├── roast: Roast level
│   └── roaster: Roasting company
├── Numerical Features (Sensory)
│   ├── aroma: Aroma score (0-10)
│   ├── acid: Acidity score (0-10)
│   ├── body: Body/mouthfeel score (0-10)
│   ├── flavor: Flavor score (0-10)
│   └── aftertaste: Aftertaste score (0-10)
└── Metadata
    ├── name: Coffee product name
    ├── location: Geographic details
    └── review_date: Review timestamp
```

### Preprocessing Pipeline

#### 1. **Text Cleaning** (`src/data/preprocessing.py`)

```python
def preprocess_text(text, remove_stop=True):
    """
    Comprehensive text preprocessing pipeline:
    1. HTML tag removal
    2. URL removal  
    3. Special character normalization
    4. Tokenization (NLTK-based)
    5. Stopword removal (optional)
    6. Lemmatization
    """
```

**Preprocessing Steps**:
1. **HTML Cleaning**: Remove HTML tags and entities
2. **URL Removal**: Strip web links and references
3. **Character Normalization**: Handle special characters and encoding
4. **Tokenization**: NLTK word tokenization with fallback
5. **Stopword Removal**: English stopwords (configurable)
6. **Lemmatization**: WordNet lemmatizer for word normalization

#### 2. **Data Standardization**

**Price Normalization**:
```python
def standardize_prices(df, price_col="est_price"):
    """
    Convert various price formats to USD per kilogram:
    - $/lb → $/kg (multiply by 2.20462)
    - $/oz → $/kg (multiply by 35.274)
    - Handle missing values and format variations
    """
```

**Geographic Standardization**:
```python
def extract_country_info(location):
    """
    Extract standardized country names from location strings:
    - Handle common variations (USA, US, United States)
    - Map regions to countries (Sumatra → Indonesia)
    - Standardize naming conventions
    """
```

#### 3. **Text Column Integration**

**Multi-Column Processing**:
- Process each text column (`desc_1`, `desc_2`, `desc_3`) individually
- Create merged text column for compatibility
- Maintain column-specific features for analysis

## 🔬 Feature Extraction Pipeline

### Overview

The feature extraction pipeline implements a **multi-modal approach** combining five different text representation methods:

```
Text Input → [TF-IDF, BERT, GloVe, Topics, Sentiment] → Feature Matrix
```

### 1. **TF-IDF Vectorization**

**Implementation**: `CoffeeFeatureExtractor.extract_tfidf_features()`

**Parameters**:
- `max_features`: 5000 (thesis specification)
- `ngram_range`: (1, 3) - unigrams, bigrams, trigrams
- `stop_words`: English stopwords removed
- `min_df`: 2 - ignore rare terms
- `max_df`: 0.95 - ignore overly common terms

**Rationale**: 
> "TF-IDF captures important coffee descriptors and terminology, focusing on domain-specific vocabulary that distinguishes different coffee characteristics."

**Output**: 5000 features per text column

### 2. **BERT Embeddings**

**Implementation**: `CoffeeFeatureExtractor.extract_bert_embeddings()`

**Model**: DistilBERT-base-uncased
- **Architecture**: Transformer-based language model
- **Dimensions**: 768-dimensional vectors
- **Processing**: Mean pooling of token embeddings
- **Context Window**: 512 tokens maximum

**Methodology**:
```python
# Tokenization and encoding
inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                  padding=True, max_length=512)

# Extract embeddings
with torch.no_grad():
    outputs = model(**inputs)
    # Mean pooling for document-level representation
    embedding = outputs.last_hidden_state.mean(dim=1)
```

**Rationale**:
> "BERT embeddings capture semantic meaning and contextual relationships in coffee reviews, understanding nuanced language that traditional methods might miss."

**Output**: 768 features per text column

### 3. **GloVe Embeddings**

**Implementation**: `CoffeeFeatureExtractor.extract_glove_embeddings()`

**Model**: GloVe Wiki-Gigaword-300
- **Training Data**: Wikipedia + Gigaword corpus
- **Dimensions**: 300-dimensional vectors
- **Aggregation**: Document-level averaging

**Methodology**:
```python
# Word-level embedding extraction
word_embeddings = []
for word in text.split():
    if word in glove_model:
        word_embeddings.append(glove_model[word])

# Document-level aggregation
doc_embedding = np.mean(word_embeddings, axis=0)
```

**Rationale**:
> "GloVe embeddings provide pre-trained semantic representations that capture general language understanding, complementing domain-specific BERT features."

**Output**: 300 features per text column

### 4. **Topic Modeling**

**Implementation**: `CoffeeFeatureExtractor.extract_topic_features()`

#### Latent Dirichlet Allocation (LDA)
- **Topics**: 10 per text column
- **Algorithm**: Online variational Bayes
- **Iterations**: 10 (optimized for performance)
- **Random State**: 42 (reproducibility)

#### Non-negative Matrix Factorization (NMF)
- **Topics**: 10 per text column  
- **Algorithm**: Coordinate descent
- **Iterations**: 1000 maximum
- **Random State**: 42 (reproducibility)

**Preprocessing for Topic Modeling**:
```python
# Separate TF-IDF for topic modeling (1000 features)
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts)
```

**Rationale**:
> "Topic modeling identifies latent themes in coffee reviews such as origin characteristics, processing methods, and flavor profiles, providing interpretable features for analysis."

**Output**: 20 features per text column (10 LDA + 10 NMF)

### 5. **Sentiment Analysis**

**Implementation**: `CoffeeFeatureExtractor.extract_sentiment_features()`

**Model**: DistilBERT-SST-2 (Stanford Sentiment Treebank)
- **Architecture**: DistilBERT fine-tuned for sentiment classification
- **Output**: Positive and negative probabilities
- **Batch Processing**: 32 samples per batch for efficiency

**Methodology**:
```python
# Sentiment pipeline with probability scores
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    return_all_scores=True
)

# Extract probability scores
for result in batch_results:
    pos_score = next(r['score'] for r in result if r['label'] == 'POSITIVE')
    neg_score = next(r['score'] for r in result if r['label'] == 'NEGATIVE')
```

**Rationale**:
> "Sentiment analysis captures the emotional tone and reviewer satisfaction, providing a direct measure of subjective evaluation that correlates with ratings."

**Output**: 2 features per text column (positive/negative probabilities)

### Feature Integration

**Total Feature Count per Text Column**:
- TF-IDF: 5,000 features
- BERT: 768 features  
- GloVe: 300 features
- Topics: 20 features (10 LDA + 10 NMF)
- Sentiment: 2 features

**Total per Column**: 6,090 features
**Total for 3 Columns**: ~18,270 text-based features

**Polars Integration**:
```python
# Efficient feature combination using Polars
feature_dfs = [tfidf_df, bert_df, glove_df, lda_df, nmf_df, sentiment_df]

# Horizontal concatenation with row alignment
combined_features = feature_dfs[0]
for feature_df in feature_dfs[1:]:
    combined_features = combined_features.join(feature_df, on="row_idx")
```

## 🤖 Model Development

### Model Selection Rationale

Based on the thesis findings, four models were selected for comparison:

#### 1. **XGBoost** (Primary Model)
> "XGBoost emerged as the best-performing model with the highest accuracy in predicting coffee ratings."

**Configuration**:
- **Objective**: Regression (continuous rating prediction)
- **Boosting Type**: Gradient boosting
- **Hyperparameters**: Grid search optimization
- **Features**: All text + categorical + numerical features

**Advantages**:
- Handles mixed feature types effectively
- Built-in feature importance
- Robust to overfitting
- Excellent performance on tabular data

#### 2. **Random Forest** (Ensemble Baseline)
**Configuration**:
- **Estimators**: Optimized through validation
- **Features**: All available features
- **Bootstrap**: True (variance reduction)

**Advantages**:
- Interpretable feature importance
- Handles missing values
- Robust ensemble method

#### 3. **Linear Regression** (Statistical Baseline)
**Configuration**:
- **Regularization**: Ridge/Lasso options
- **Features**: Standardized features
- **Solver**: Optimized for feature count

**Advantages**:
- Interpretable coefficients
- Fast training and inference
- Statistical significance testing

#### 4. **Multinomial Inverse Regression (MNIR)** (Thesis-Specific)
**Implementation**: Custom implementation for categorical rating prediction
**Rationale**: Specialized approach for ordinal rating outcomes

### Model Training Pipeline

```python
def train_and_evaluate_models(input_file, target_column, models_to_train):
    """
    Comprehensive model training pipeline:
    1. Feature loading and preprocessing
    2. Train/validation/test split
    3. Model training with hyperparameter optimization
    4. Performance evaluation
    5. Feature importance analysis (SHAP)
    6. Model persistence
    """
```

## 📈 Evaluation Methodology

### Performance Metrics

#### Regression Metrics
- **RMSE**: Root Mean Square Error (primary metric)
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Pearson Correlation**: Linear relationship strength

#### Classification Metrics (for rating categories)
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision/recall

### Cross-Validation Strategy

**Method**: Stratified K-Fold (k=5)
**Rationale**: Ensures balanced rating distribution across folds
**Metrics**: Average performance across folds with confidence intervals

### Feature Importance Analysis

#### SHAP (SHapley Additive exPlanations)
```python
# SHAP analysis for model interpretation
explainer = shap.TreeExplainer(xgboost_model)
shap_values = explainer.shap_values(X_test)

# Feature importance ranking
feature_importance = np.abs(shap_values).mean(0)
```

**Analysis Dimensions**:
1. **Global Importance**: Overall feature ranking
2. **Local Explanations**: Individual prediction explanations
3. **Feature Interactions**: Pairwise feature effects
4. **Partial Dependence**: Feature-outcome relationships

## 🛠️ Implementation Details

### Technology Stack

#### Data Processing
- **Polars**: Primary DataFrame library for efficiency
- **Pandas**: Compatibility layer for sklearn integration
- **NumPy**: Numerical computing foundation

#### Machine Learning
- **Scikit-learn**: Traditional ML algorithms and preprocessing
- **XGBoost**: Gradient boosting implementation
- **Transformers**: BERT models and tokenization
- **PyTorch**: Deep learning backend

#### Text Processing
- **NLTK**: Text preprocessing utilities
- **Gensim**: GloVe embeddings and topic modeling
- **Transformers**: Modern NLP models

### Performance Optimizations

#### Memory Management
```python
# Polars lazy evaluation for large datasets
df_lazy = pl.scan_csv("large_dataset.csv")
processed = df_lazy.select([...]).filter([...]).collect()
```

#### Batch Processing
```python
# BERT embedding extraction in batches
batch_size = 32
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    embeddings.extend(process_batch(batch))
```

#### Model Persistence
```python
# Efficient model and feature storage
with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)
    
# Feature metadata preservation
feature_metadata = {
    "tfidf_features": vectorizer.get_feature_names_out(),
    "bert_dimensions": 768,
    "topic_labels": topic_model.components_
}
```

### Reproducibility Measures

#### Random State Management
- **Global Seed**: 42 for all random operations
- **Model Seeds**: Consistent across training runs
- **Data Splits**: Deterministic train/test splits

#### Environment Specification
```python
# requirements.txt with exact versions
polars==1.30.0
transformers==4.18.0
torch==1.11.0
scikit-learn==1.0.0
```

#### Configuration Management
```python
# src/config/settings.py
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 5000
BERT_MAX_LENGTH = 512
N_TOPICS = 10
```

## 🔍 Validation and Testing

### Data Quality Checks
1. **Missing Value Analysis**: Systematic handling of null values
2. **Outlier Detection**: Statistical outlier identification
3. **Distribution Analysis**: Target variable distribution validation
4. **Text Quality**: Empty text detection and handling

### Model Validation
1. **Cross-Validation**: 5-fold stratified validation
2. **Holdout Testing**: Final model evaluation on unseen data
3. **Feature Stability**: Consistency across different data splits
4. **Prediction Calibration**: Model confidence assessment

### Error Analysis
1. **Residual Analysis**: Systematic error pattern identification
2. **Feature Attribution**: Understanding prediction drivers
3. **Edge Case Analysis**: Performance on extreme cases
4. **Bias Detection**: Systematic bias identification

---

This methodology provides the foundation for reproducible, rigorous analysis of coffee review text data, implementing state-of-the-art NLP techniques within a robust experimental framework. 