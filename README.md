# Coffee Text Analytics: A Data-Driven Approach

A comprehensive text analytics and predictive modeling framework for analyzing consumer coffee reviews. This project implements the methodology described in the thesis **"Leveraging Text Analytics and Predictive Modeling to Analyze Consumer Coffee Reviews: A Data-Driven Approach"** by Marcelo Seijas, Erasmus University Rotterdam.

## Current Status

**Phase 2.2 Complete** - Ready for scaling or advanced research  
**Latest Results**: XGBoost R²=0.9453, Ridge R²=0.9259  
**Thesis Compliance**: 100% methodology alignment achieved  
**Quick Start**: `python validate_15_percent_methodology.py` (4-minute validation)

## Research Overview

This study investigates the key sensory and non-sensory attributes that drive consumer preferences by analyzing coffee reviews from CoffeeReview.com using a combination of text analytics, sentiment analysis, Multinomial Inverse Regression (MNIR), and machine learning.

### Key Research Questions

1. What are the key factors that influence coffee ratings?
2. How do text-based features compare to traditional sensory attributes?
3. Can advanced NLP techniques improve rating prediction accuracy?
4. What insights can topic modeling reveal about coffee review themes?

### Research Methodology

This implementation follows the exact methodology described in the thesis:

> "A diverse set of features, including flavor attributes, categorical variables such as country of origin and roast level, and text-based features derived from BERT embeddings, GloVe vectors, and LDA topics, were used to predict coffee ratings."

## Key Features

### Modern Data Processing with Polars
- **Polars-First Approach**: Efficient data processing with lazy evaluation
- **Hybrid Compatibility**: Seamless conversion to Pandas when needed for sklearn
- **Performance Optimization**: Leverages Polars' memory efficiency

### Advanced Text Analytics Pipeline

#### Multi-Modal Feature Extraction
- **TF-IDF Vectorization**: 200 features per desc column (600 total) with unigrams, bigrams, and trigrams
- **BERT Embeddings**: 768-dimensional semantic representations using DistilBERT (2304 total)
- **GloVe Embeddings**: 300-dimensional pre-trained word vectors (900 total)
- **Topic Modeling**: LDA and NMF for thematic analysis (30 total topics)
- **Sentiment Analysis**: DistilBERT-based positive/negative sentiment scoring (6 total)
- **LASSO Feature Selection**: 279 selected from 3,840 text features (92.7% reduction)

#### Machine Learning Models
- **XGBoost**: Best performance (R²=0.9453) with hyperparameter optimization
- **Ridge Regression**: Excellent performance (R²=0.9259) 
- **LASSO Regression**: Strong performance (R²=0.8897)
- **Random Forest**: Good ensemble performance (R²=0.8675)
- **Linear Regression**: Baseline model (R²=0.8173)
- **MNIR**: Multinomial Inverse Regression for text-sensory correlation analysis

#### Enhanced Experiment Tracking
- **MLflow Integration**: Comprehensive experiment tracking and model registry
- **Optuna Optimization**: TPE algorithm with intelligent pruning (5-10x speedup)
- **SHAP Analysis**: Automated feature importance and model interpretation
- **Performance Metrics**: MAE, RMSE, R² with statistical validation

## Key Findings

### Model Performance Results

**Current Validation Results (15% Sample):**

```
Model           R²       RMSE     MAE      Status
--------------------------------------------------
XGBoost         0.9453   0.4103   0.2152   Best
Ridge           0.9259   0.4775   0.3801   Excellent  
LASSO           0.8897   0.5825   0.4623   Strong
Random Forest   0.8675   0.6386   0.3590   Good
Linear          0.8173   0.7497   0.6101   Baseline
```

**MNIR Analysis Results:**
```
Sensory Attribute    R²       MSE      Performance
--------------------------------------------------
Acidity             0.9389   0.5343   Excellent
Aftertaste          0.8420   0.0375   Strong  
Body                0.7966   0.0508   Good
Aroma               0.7834   0.0816   Good
Flavor              0.5789   0.0433   Moderate
```

### Feature Importance Insights

**Text features dominate:** BERT embeddings and TF-IDF features were found to be the most predictive of coffee ratings, followed by sentiment scores and topic features.

**Topic analysis reveals:** Distinct themes like origin characteristics, processing methods, flavor profiles, and brewing recommendations.

**Sentiment-rating correlation:** Strong relationship between sentiment and ratings - positive sentiment correlates with higher ratings (8.5+), while negative sentiment associates with lower ratings (<7.0).

## Installation & Setup

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for BERT embeddings)
- CUDA-compatible GPU (optional, for faster processing)

### Installation Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd coffee-text-analytics
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python run_tests.py  # Run test suite
```

## Usage

### Quick Start - 15% Validation (Recommended)

**Fastest way to validate thesis methodology (4 minutes):**
```bash
python validate_15_percent_methodology.py
```

This gives you complete thesis validation with XGBoost R²=0.9453, all models + MNIR analysis.

### Quick Reference
- **Current Achievement**: 100% thesis methodology compliance
- **Best Model**: XGBoost (R²=0.9453) 
- **Feature Selection**: 279 selected from 3,840 text features (92.7% reduction)
- **MNIR Performance**: Acidity R²=0.9389, Body R²=0.7966
- **Infrastructure**: MLflow + Optuna + SHAP analysis ready

### Advanced Usage Options

#### Scale to Larger Samples
```bash
python validate_15_percent_methodology.py --sample_size=50   # 50% sample
python validate_15_percent_methodology.py --sample_size=100  # Full dataset  
```

#### Enhanced Hyperparameter Optimization
```bash
python validate_15_percent_methodology.py --mode=full --trials=50  # Research-grade optimization
```

#### MLflow Experiment Tracking
```bash
mlflow ui --port 5000  # View results in browser
```

### Complete Pipeline

Run the full methodology:
```bash
python main.py --steps all
```

### Step-by-Step Execution

#### 1. Data Preprocessing
```bash
python main.py --steps preprocess
```
Cleans text columns, extracts country info, standardizes prices.

#### 2. Feature Extraction
```bash
python main.py --steps features
```
Runs TF-IDF, BERT, sentiment, and topic extraction.

#### 3. Model Training
```bash
python main.py --steps train
```
Trains all models with hyperparameter optimization and SHAP analysis.

#### 4. Results Visualization
```bash
python main.py --steps visualize
```
Creates performance comparisons and feature importance charts.

### Custom Configuration

#### Specify Models
```bash
python main.py --models xgboost random_forest mnir --steps train
```

#### Adjust Features
```bash
python main.py --text_columns desc_1 desc_2 desc_3 --steps features
```

#### Environment Settings
```bash
COFFEE_ENV=production python main.py --steps all  # Production settings
COFFEE_ENV=testing python main.py --steps all     # Testing settings
```

## Data Schema

The dataset from CoffeeReview.com includes:

#### Text Features (Primary Analysis)
- `desc_1`: Primary review description
- `desc_2`: Secondary review notes  
- `desc_3`: Additional tasting notes

#### Target Variable
- `rating`: Coffee rating score (0-100 scale)

#### Categorical Features
- `origin`: Coffee origin/country
- `roast`: Roast level (light, medium, dark)
- `roaster`: Coffee roasting company

#### Numerical Features (Sensory Attributes)
- `est_price`: Estimated price per pound
- `aroma`: Aroma score (0-10)
- `acid`: Acidity score (0-10)
- `body`: Body/mouthfeel score (0-10)
- `flavor`: Flavor score (0-10)
- `aftertaste`: Aftertaste score (0-10)

## Feature Engineering

### Component-Based Architecture

```python
# Example using the new architecture
from src.features.feature_manager import CoffeeFeatureManager

# Initialize feature manager
feature_manager = CoffeeFeatureManager({
    'extractors': ['tfidf', 'sentiment', 'topic']
})

# Extract features
feature_manager.fit(training_texts)
features_df = feature_manager.extract_all_features(
    df=coffee_data,  # Polars DataFrame
    text_columns=['desc_1', 'desc_2', 'desc_3']
)
```

### Feature Dimensions

Per text column:
- **TF-IDF Features**: 5,000 dimensions
- **BERT Embeddings**: 768 dimensions
- **Sentiment Features**: 2 dimensions
- **Topic Features**: 20 dimensions (10 LDA + 10 NMF)

**Total Features per Text Column**: ~5,790 dimensions  
**Total for 3 Text Columns**: ~17,370 text-based features

## Dependencies & Technology Stack

### Core Data Processing
- **Polars** `>=0.15.0`: Modern DataFrame library
- **Pandas** `>=1.4.0`: Compatibility layer for sklearn
- **NumPy** `>=1.20.0`: Numerical computing

### Machine Learning & NLP
- **Scikit-learn** `>=1.0.0`: ML algorithms and preprocessing
- **XGBoost** `>=1.5.0`: Gradient boosting (best model)
- **Transformers** `>=4.18.0`: BERT embeddings
- **PyTorch** `>=1.11.0`: Deep learning backend
- **Gensim** `>=4.1.0`: Topic modeling
- **NLTK** `>=3.7.0`: Text preprocessing

### Experiment Tracking & Optimization
- **MLflow** `>=2.0.0`: Experiment tracking
- **Optuna** `>=3.0.0`: Hyperparameter optimization
- **SHAP** `>=0.40.0`: Model interpretation

### Visualization
- **Plotly** `>=5.0.0`: Interactive visualizations
- **Matplotlib** `>=3.5.0`: Basic plotting
- **Seaborn** `>=0.11.0`: Statistical visualizations

## Testing & Quality Assurance

### Comprehensive Test Suite
```bash
python run_tests.py
```

**Test Coverage:**
- 96.7% pass rate - Robust and stable codebase
- Data processing tests - Polars/Pandas integration validated
- Integration tests - End-to-end pipeline validation
- Performance tests - Memory and speed optimization
- Thesis compliance validation - 15% methodology validator proven

### Code Quality
- Component-based design - Modular, extensible architecture
- Zero import conflicts - Clean dependency management
- MLflow + Optuna integration - Enhanced experiment tracking
- Professional documentation - Academic-grade documentation
- Thesis methodology compliance - 100% validation achieved

## Important Notes

### Model Persistence Behavior
The pipeline always trains new models and overwrites existing models in the `models/` directory. Before running on new data:

```bash
# Option 1: Clear models directory
rm -rf models/*.pkl

# Option 2: Backup existing models
mkdir models_backup_$(date +%Y%m%d)
cp models/*.pkl models_backup_$(date +%Y%m%d)/

# Then run pipeline
python main.py --steps all
```

### Model Files Created
- `models/tfidf_vectorizer.pkl` - TF-IDF vocabulary
- `models/lda_model.pkl` - LDA topic model
- `models/nmf_model.pkl` - NMF topic model  
- `models/linear_model.pkl` - Linear regression
- `models/random_forest_model.pkl` - Random forest
- `models/xgboost_model.pkl` - XGBoost
- `models/mnir_model.pkl` - MNIR model

## Documentation

### Current Documentation
- **CURRENT_STATUS.md**: Up-to-date project status and achievements
- **STRATEGIC_IMPLEMENTATION_PLAN.md**: Master plan with progress
- **docs/thesis.md**: Complete thesis document
- **docs/findings.md**: Research findings and insights
- **docs/methodology.md**: Detailed research methodology

### Historical Archive
- **docs/archive/**: Complete development history for reference

## Clean Output Directories

The `clean_outputs.py` utility helps ensure fresh pipeline runs:

```bash
# Interactive mode with confirmation
python clean_outputs.py

# Clean all without confirmation
python clean_outputs.py --confirm

# Preview what would be deleted
python clean_outputs.py --dry-run

# Choose specific directories
python clean_outputs.py --selective
```

## Contributing

We welcome contributions that extend the thesis methodology:

### Research Extensions
- Additional embedding models (RoBERTa, ELECTRA)
- Advanced topic modeling (BERTopic, Top2Vec)
- Cross-domain validation studies
- Temporal analysis of review trends

### Technical Improvements
- GPU acceleration for BERT processing
- Distributed processing capabilities
- Real-time inference pipeline
- Web interface for exploration

## License & Citation

This project is part of an academic thesis and is provided for educational and research purposes.

### Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{seijas2024coffee,
  title={Leveraging Text Analytics and Predictive Modeling to Analyze Consumer Coffee Reviews: A Data-Driven Approach},
  author={Seijas, Marcelo},
  year={2024},
  school={Erasmus University Rotterdam},
  department={Erasmus School of Economics},
  program={Data Science and Marketing Analytics},
  supervisor={O'Neill, Eoghan},
  secondassessor={Brüggemann, Sean}
}
```

---

**Thesis Supervisor**: Eoghan O'Neill  
**Second Assessor**: Sean Brüggemann  
**Institution**: Erasmus University Rotterdam, Erasmus School of Economics  
**Program**: Data Science and Marketing Analytics  
**Year**: 2024