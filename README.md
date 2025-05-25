# Coffee Text Analytics: A Data-Driven Approach

A comprehensive text analytics and predictive modeling framework for analyzing consumer coffee reviews. This project implements the methodology described in the thesis **"Leveraging Text Analytics and Predictive Modeling to Analyze Consumer Coffee Reviews: A Data-Driven Approach"** by Marcelo Seijas, Erasmus University Rotterdam.

## 🎯 Research Overview

### Thesis Abstract

> In the growing specialty coffee market, understanding the factors that influence consumer ratings can provide valuable insights for producers, marketers, and retailers. This study investigates the key sensory and non-sensory attributes that drive consumer preferences by analyzing coffee reviews from CoffeeReview.com using a combination of text analytics, sentiment analysis, Multinomial Inverse Regression (MNIR), and machine learning.

### Key Research Questions

1. **What are the key factors that influence coffee ratings?**
2. **How do text-based features compare to traditional sensory attributes?**
3. **Can advanced NLP techniques improve rating prediction accuracy?**
4. **What insights can topic modeling reveal about coffee review themes?**

### Research Methodology

This implementation follows the exact methodology described in the thesis:

> "A diverse set of features, including flavor attributes, categorical variables such as country of origin and roast level, and text-based features derived from BERT embeddings, GloVe vectors, and LDA topics, were used to predict coffee ratings."

## 🚀 Key Features & Innovation

### Modern Data Processing with Polars
- **Polars-First Approach**: Showcases modern data processing with Polars for efficiency
- **Hybrid Compatibility**: Seamless conversion to Pandas when needed for sklearn
- **Performance Optimization**: Leverages Polars' lazy evaluation and memory efficiency

### Advanced Text Analytics Pipeline

#### 1. **Multi-Modal Feature Extraction**
- **TF-IDF Vectorization**: 5000 features with unigrams, bigrams, and trigrams
- **BERT Embeddings**: 768-dimensional semantic representations using DistilBERT
- **GloVe Embeddings**: 300-dimensional pre-trained word vectors
- **Topic Modeling**: LDA and NMF for thematic analysis (10 topics each)
- **Sentiment Analysis**: DistilBERT-based positive/negative sentiment scoring

#### 2. **Machine Learning Models**
- **XGBoost**: Primary model (best performance in thesis)
- **Random Forest**: Ensemble method for comparison
- **Linear Regression**: Baseline model
- **MNIR**: Multinomial Inverse Regression (Lasso feature selection + regression for sensory attribute prediction)

#### 3. **Model Interpretation**
- **SHAP Values**: Feature importance analysis
- **Topic Visualization**: LDA topic interpretation
- **Performance Metrics**: Comprehensive evaluation
- **MNIR Analysis**: Text-sensory relationship quantification with R² = 0.95 (acidity), R² = 0.94 (body)

## 📊 Key Thesis Findings

### Model Performance Results

From the thesis research:

> "XGBoost emerged as the best-performing model with the highest accuracy in predicting coffee ratings."

**Performance Hierarchy (Rating Prediction):**
1. **XGBoost** - Best overall performance
2. **Random Forest** - Strong ensemble performance  
3. **Linear Regression** - Baseline comparison

**Additional Analysis:**
- **MNIR** - Text-sensory relationship quantification (R² = 0.95 for acidity, R² = 0.94 for body)

### Feature Importance Insights

#### Text Features Dominate
> "Text-based features (BERT, TF-IDF) were found to be the most predictive of coffee ratings."

**Feature Ranking by Importance:**
1. **BERT Embeddings** - Capture semantic meaning and context
2. **TF-IDF Features** - Important coffee terminology and descriptors
3. **Sentiment Scores** - Reviewer emotional response
4. **Topic Features** - Thematic content analysis
5. **Traditional Attributes** - Sensory scores (aroma, body, etc.)

#### Topic Analysis Revelations
> "LDA reveals distinct themes like origin characteristics and flavor profiles."

**Discovered Topics Include:**
- **Origin Characteristics**: Geographic and terroir influences
- **Processing Methods**: Wet/dry processing, fermentation
- **Flavor Profiles**: Fruity, nutty, chocolatey, floral notes
- **Brewing Recommendations**: Extraction methods and preparation
- **Quality Assessments**: Overall satisfaction and recommendations

### Sentiment-Rating Correlation
> "Strong relationship between sentiment and ratings was observed."

- **Positive Sentiment**: Strongly correlates with higher ratings (8.5+)
- **Negative Sentiment**: Associated with lower ratings (<7.0)
- **Neutral Reviews**: Often focus on technical aspects rather than enjoyment

## 🏗️ Project Architecture

```
coffee-text-analytics/
├── 📁 data/
│   ├── raw/                    # Raw coffee review data
│   │   └── coffee_clean.csv    # CoffeeReview.com dataset
│   └── processed/              # Polars-processed data files
├── 📁 src/
│   ├── data/                   # Data loading and preprocessing
│   │   ├── loader.py          # Polars-based data loading
│   │   └── preprocessing.py   # Text preprocessing pipeline
│   ├── features/              # Feature extraction (Polars-based)
│   │   └── feature_extraction.py  # Complete thesis methodology
│   ├── models/                # Model training and evaluation
│   │   └── model_training.py  # ML implementations + MNIR
│   ├── utils/                 # Utility functions
│   │   ├── cleaning.py        # Data cleaning utilities
│   │   └── utils.py           # General utilities
│   ├── visualization/         # Plotting and visualization
│   │   ├── plots.py           # Statistical plots
│   │   └── visualize.py       # Advanced visualizations
│   └── config/                # Configuration
│       └── settings.py        # Project settings
├── 📁 models/                 # Saved model artifacts
├── 📁 output/                 # Analysis outputs
│   └── figures/              # Generated plots and figures
├── 📁 notebooks/              # Jupyter notebooks for exploration
├── 📁 docs/                   # Documentation
│   ├── methodology.md         # Detailed thesis methodology
│   ├── findings.md           # Research findings and insights
│   └── api.md                # API documentation
├── main.py                   # Main execution script (Polars-based)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🛠️ Installation & Setup

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

4. **Download NLTK data:**
```bash
python download_nltk.py
```

5. **Verify installation:**
```bash
python -c "import polars as pl; import transformers; print('Setup complete!')"
```

## 🚀 Usage

### Quick Start - Complete Pipeline

Run the full thesis methodology:
```bash
python main.py --steps all
```

### Step-by-Step Execution

#### 1. **Data Preprocessing** (Polars-based)
```bash
python main.py --steps preprocess
```
- Cleans and preprocesses text columns (`desc_1`, `desc_2`, `desc_3`)
- Extracts country information from origin data
- Standardizes price information
- Creates merged text columns for analysis

#### 2. **Feature Extraction** (Thesis Methodology)
```bash
python main.py --steps features
```
Implements the complete feature extraction pipeline:
- **TF-IDF**: 5000 features per text column
- **BERT**: 768 dimensions per text column  
- **GloVe**: 300 dimensions per text column
- **Topics**: 10 LDA + 10 NMF topics per text column
- **Sentiment**: Positive/negative probabilities per text column

#### 3. **Model Training** (XGBoost + MNIR)
```bash
python main.py --models xgboost random_forest mnir --steps train
```
- Trains all models from thesis (XGBoost, Random Forest, Linear Regression, MNIR)
- Performs hyperparameter optimization
- Generates SHAP feature importance analysis
- Saves trained models for inference

#### 4. **Results Visualization**
```bash
python main.py --steps visualize
```
- Creates thesis-quality visualizations
- Topic modeling plots
- Feature importance charts
- Model performance comparisons
- Sentiment analysis visualizations

### Custom Configuration

#### Specify Text Columns
```bash
python main.py --text_columns desc_1 desc_2 desc_3 --steps features
```

#### Adjust Topic Modeling
```bash
python main.py --n_topics 15 --steps features
```

#### Train Specific Models
```bash
python main.py --models xgboost random_forest mnir --steps train
```

## 📈 Data Schema

### Coffee Review Dataset Structure

The dataset from CoffeeReview.com includes:

#### **Text Features** (Primary Analysis)
- `desc_1`: Primary review description
- `desc_2`: Secondary review notes  
- `desc_3`: Additional tasting notes

#### **Target Variable**
- `rating`: Coffee rating score (0-100 scale)

#### **Categorical Features**
- `origin`: Coffee origin/country
- `roast`: Roast level (light, medium, dark)
- `roaster`: Coffee roasting company

#### **Numerical Features** (Sensory Attributes)
- `est_price`: Estimated price per pound
- `aroma`: Aroma score (0-10)
- `acid`: Acidity score (0-10)
- `body`: Body/mouthfeel score (0-10)
- `flavor`: Flavor score (0-10)
- `aftertaste`: Aftertaste score (0-10)

#### **Metadata**
- `name`: Coffee product name
- `location`: Detailed location information
- `review_date`: Date of review

## 🔬 Feature Engineering Pipeline

### Text Processing Workflow

```python
# Example of the thesis methodology implementation
from src.features.feature_extraction import CoffeeFeatureExtractor

# Initialize with Polars
extractor = CoffeeFeatureExtractor()

# Extract all features as described in thesis
features_df = extractor.extract_all_features(
    df=coffee_data,  # Polars DataFrame
    text_columns=['desc_1', 'desc_2', 'desc_3'],
    n_topics=10
)
```

### Feature Dimensions

Per text column, the pipeline generates:

1. **TF-IDF Features**: 5,000 dimensions
   - Unigrams, bigrams, trigrams
   - Stop word removal and frequency filtering
   - Coffee-specific terminology capture

2. **BERT Embeddings**: 768 dimensions
   - Semantic representations using DistilBERT
   - Mean pooling of token embeddings
   - Context-aware semantic understanding

3. **GloVe Embeddings**: 300 dimensions
   - Pre-trained word vectors (Wikipedia + Gigaword)
   - Document-level averaging
   - Captures word-level semantics

4. **Topic Features**: 20 dimensions (10 LDA + 10 NMF)
   - Latent Dirichlet Allocation topics
   - Non-negative Matrix Factorization topics
   - Thematic content analysis

5. **Sentiment Features**: 2 dimensions
   - Positive sentiment probability
   - Negative sentiment probability
   - DistilBERT-based classification

**Total Features per Text Column**: 6,090 dimensions
**Total for 3 Text Columns**: ~18,270 text-based features

## 🎯 Research Contributions

### 1. **Multi-Modal Feature Fusion**
> "Combining different text representations improves performance"

- Demonstrates synergy between TF-IDF, BERT, and GloVe
- Shows complementary strengths of different embedding approaches
- Validates ensemble feature approach for text analysis

### 2. **Domain-Specific Insights**
> "Topic insights reveal distinct themes like origin characteristics and flavor profiles"

- Identifies key themes in coffee reviews
- Maps sensory language to rating patterns
- Provides actionable insights for coffee industry

### 3. **Modern Data Processing**
- Showcases Polars for efficient data manipulation
- Demonstrates hybrid Polars/Pandas approach
- Optimizes memory usage for large-scale text processing

### 4. **Methodological Rigor**
- Implements complete academic methodology
- Provides reproducible research pipeline
- Enables extension and validation studies

## 📚 Dependencies & Technology Stack

### Core Data Processing
- **Polars** `>=0.15.0`: Modern DataFrame library for efficient processing
- **Pandas** `>=1.4.0`: Compatibility layer for sklearn integration
- **NumPy** `>=1.20.0`: Numerical computing foundation

### Machine Learning & NLP
- **Scikit-learn** `>=1.0.0`: Traditional ML algorithms and preprocessing
- **XGBoost** `>=1.5.0`: Gradient boosting (best-performing model)
- **Transformers** `>=4.18.0`: BERT embeddings and sentiment analysis
- **PyTorch** `>=1.11.0`: Deep learning backend for transformers
- **Gensim** `>=4.1.0`: GloVe embeddings and topic modeling
- **NLTK** `>=3.7.0`: Text preprocessing utilities

### Visualization & Analysis
- **Matplotlib** `>=3.5.0`: Basic plotting functionality
- **Seaborn** `>=0.11.0`: Statistical visualizations
- **SHAP** `>=0.40.0`: Model interpretation and feature importance
- **WordCloud** `>=1.8.0`: Topic visualization

## 🤝 Contributing

We welcome contributions that extend the thesis methodology:

### Research Extensions
- Additional embedding models (RoBERTa, ELECTRA)
- Advanced topic modeling (BERTopic, Top2Vec)
- Cross-domain validation studies
- Temporal analysis of review trends

### Technical Improvements
- GPU acceleration for BERT processing
- Distributed processing with Dask
- Real-time inference pipeline
- Web interface for exploration

### Contribution Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-extension`)
3. Implement changes with tests
4. Update documentation
5. Submit pull request with detailed description

## 📄 License & Citation

### Academic Use
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

### Software Citation

```bibtex
@software{seijas2024coffee_software,
  title={Coffee Text Analytics: Implementation of Multi-Modal Text Analysis for Consumer Review Prediction},
  author={Seijas, Marcelo},
  year={2024},
  url={https://github.com/username/coffee-text-analytics},
  note={Implementation of thesis methodology using Polars and modern NLP techniques}
}
```

## 🔗 Related Work & References

### Key Academic References
- **Text Analytics**: Silge & Robinson (2017) - Text Mining with R
- **BERT Embeddings**: Devlin et al. (2018) - BERT: Pre-training of Deep Bidirectional Transformers
- **Topic Modeling**: Blei et al. (2003) - Latent Dirichlet Allocation
- **XGBoost**: Chen & Guestrin (2016) - XGBoost: A Scalable Tree Boosting System

### Industry Applications
- **Coffee Industry**: Specialty Coffee Association rating standards
- **Review Analysis**: Amazon, Yelp review prediction systems
- **Sentiment Analysis**: Social media monitoring and brand analysis

---

**Thesis Supervisor**: Eoghan O'Neill  
**Second Assessor**: Sean Brüggemann  
**Institution**: Erasmus University Rotterdam, Erasmus School of Economics  
**Program**: Data Science and Marketing Analytics  
**Year**: 2024