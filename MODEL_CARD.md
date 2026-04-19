# Model Card: Coffee Review Rating Predictor

## Model Details

**Model Name:** XGBoost Coffee Rating Predictor  
**Model Type:** Gradient Boosted Decision Tree Regression  
**Version:** 1.0  
**Date:** 2026-04-20  
**Authors:** Marcelo Seijas (Erasmus University Rotterdam)  
**Repository:** https://github.com/mseijse01/coffee-text-analytics

---

## Intended Use

This model predicts coffee quality ratings (80–100 scale) from consumer review text using advanced NLP and machine learning techniques.

**Primary Use Cases:**
- Research: Analyzing factors that influence coffee ratings
- Educational: Portfolio demonstration of text analytics and ML engineering
- Analysis: Understanding which textual features correlate with coffee quality

**NOT Intended For:**
- Commercial coffee grading or quality certification
- Production recommendation systems without domain expert review
- Generalization to rating systems outside the 80–100 scale

---

## Training Data

**Source:** CoffeeReview.com reviews (publicly available dataset)  
**Original Size:** ~6,400 reviews  
**Final Dataset:** ~2,440 reviews after quality filtering  
**Features Used:** Text columns (`desc_1`, `desc_2`, `desc_3`), categorical (country, roast level), numeric attributes  
**Target:** Coffee rating (80–100 scale, continuous)  

**Data Split:**
- Training: 70% (1,708 samples)
- Testing: 30% (732 samples)
- **Stratification:** By rating bins to ensure representative splits

---

## Feature Engineering

**Total Raw Features:** ~3,840  
**Selected Features:** 279 (92.7% dimensionality reduction via LASSO)

**Feature Types:**

1. **TF-IDF Vectorization** (600 features)
   - Unigrams, bigrams, trigrams across three description columns
   - Captures domain-specific vocabulary patterns

2. **BERT Embeddings** (2,304 features)
   - DistilBERT 768-dim vectors, averaged across descriptions
   - Semantic representation of review text

3. **GloVe Embeddings** (900 features)
   - Pre-trained 300-dim word vectors
   - Complementary semantic encoding

4. **Topic Modeling** (30 features)
   - LDA and NMF: thematic structure (e.g., origin, processing, flavor profiles)

5. **Sentiment Analysis** (6 features)
   - DistilBERT sentiment (positive/negative scores)
   - Captures reviewer tone

6. **Categorical & Numeric** (15+ features)
   - Country of origin, roast level, acidity, body, aroma, aftertaste, flavor

**Feature Selection:** LASSO regression with 5-fold cross-validation to identify the 279 most predictive features.

---

## Performance

**Evaluation Metrics (30% Test Set):**

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| **XGBoost** | **0.9453** | **0.4103** | **0.2152** |
| Ridge | 0.9259 | 0.4775 | 0.3801 |
| LASSO | 0.8897 | 0.5825 | 0.4623 |
| Random Forest | 0.8675 | 0.6386 | 0.3590 |
| Linear Regression | 0.8173 | 0.7497 | 0.6101 |

**Performance Interpretation:**
- R² = 0.9453: Model explains 94.5% of variance in coffee ratings
- RMSE = 0.41: Average prediction error ±0.41 points on 80–100 scale
- MAE = 0.22: Median absolute error of 0.22 points (very accurate)

**Supporting Model:**
- **MNIR (Multinomial Inverse Regression):** Included for interpretability analysis of sensory attributes (acidity, body, aroma, aftertaste, flavor) — not optimized for prediction performance.

---

## Limitations

1. **Single Source Bias:** Data comes exclusively from CoffeeReview.com; results may not generalize to other review platforms (e.g., Amazon, specialty retailers)

2. **Language & Cultural Bias:** All reviews in English; may not represent preferences in other markets or languages

3. **Subjective Target Variable:** Coffee ratings are reviewer opinions, not objective quality measures. Inter-rater agreement is imperfect.

4. **Rating Scale Specificity:** Model trained on 80–100 scale; extrapolation outside this range is not recommended

5. **Temporal Bias:** Training data spans a fixed time window; newer coffee products and reviewing trends may differ

6. **Feature Dependency:** Model relies on specific text patterns learned from CoffeeReview.com reviews; different review formats may degrade performance

---

## Ethical Considerations

**Positive:**
- No PII (personally identifiable information) used in training
- No protected attributes (race, gender, age) in model inputs
- Transparent methodology documented in research

**Concerns:**
- **Reviewer Bias:** Training data reflects biases of coffee reviewers (e.g., preference for single-origin, specialty beans), not universal consumer preferences
- **Accessibility:** Model designed for research; commercialization without domain expert review could mislead consumers
- **Interpretability Limits:** While SHAP analysis is available, deep model reasoning requires expert interpretation

---

## How to Use

### Installation
```bash
pip install -r requirements.txt
```

### Loading the Model
```python
import pickle
import numpy as np

# Load trained XGBoost model
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example: predict rating from 279-dimensional feature vector
features = np.random.randn(279)  # Your feature vector
prediction = model.predict([features])
print(f"Predicted rating: {prediction[0]:.2f}")
```

### Running the Pipeline
```bash
# Full pipeline: preprocess → features → select → train
python main.py --steps all

# Or train existing model on new data
python main.py --steps features train --models xgboost
```

### Validation
```bash
# Verify thesis compliance (15% sample, ~4 min)
python validate_15_percent_methodology.py
```

---

## References

- **Thesis:** "Leveraging Text Analytics and Predictive Modeling to Analyze Consumer Coffee Reviews: A Data-Driven Approach," Marcelo Seijas, Erasmus University Rotterdam
- **Dataset:** CoffeeReview.com (publicly accessible)
- **Methodology:** Stratified sampling, LASSO feature selection, XGBoost with Optuna hyperparameter optimization, SHAP analysis

---

**Last Updated:** 2026-04-20  
**Maintained By:** Marcelo Seijas
