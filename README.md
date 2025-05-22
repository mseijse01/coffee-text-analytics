# Coffee Text Analytics

A data science project that analyzes coffee reviews using natural language processing and machine learning to understand what factors influence coffee ratings and consumer preferences.

## Project Overview

This project applies text analytics and machine learning techniques to analyze coffee reviews and predict coffee ratings. It demonstrates how to extract meaningful insights from unstructured text data in the domain of coffee reviews.

### Key Features

- **Text Preprocessing**: Clean and normalize coffee review text
- **Feature Extraction**:
  - Topic modeling (LDA & NMF) to discover latent themes
  - Sentiment analysis to quantify opinions
  - Text-based feature engineering
- **Predictive Modeling**: Multiple regression models to predict coffee ratings
- **Visualization**: Insightful visualizations of coffee review data and analysis results

## Project Structure

```
coffee-text-analytics/
│
├── data/               # Data directory
│   ├── raw/            # Raw, unprocessed data
│   └── processed/      # Cleaned and processed data
│
├── models/             # Trained models and model objects
│
├── notebooks/          # Jupyter notebooks for exploration and analysis
│
├── output/             # Generated output
│   └── figures/        # Visualization outputs
│
├── src/                # Source code
│   ├── data/           # Data loading and preprocessing
│   ├── features/       # Feature extraction code
│   ├── models/         # Model training code
│   └── visualization/  # Visualization utilities
│
├── main.py             # Main entry point for the pipeline
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/coffee-text-analytics.git
cd coffee-text-analytics
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Complete Pipeline

Run the entire analysis pipeline:

```bash
python main.py --steps all
```

This will execute:
1. Data preprocessing
2. Feature extraction
3. Model training
4. Result visualization

### Individual Pipeline Components

Run specific parts of the pipeline:

```bash
# Preprocess data only
python main.py --steps preprocess

# Extract features from preprocessed data
python main.py --steps features

# Train models on extracted features
python main.py --steps train

# Generate visualizations
python main.py --steps visualize
```

### Custom Dataset

You can use your own coffee review dataset:

```bash
python main.py --input_file path/to/your/data.csv --text_columns description notes
```

## Results

The project generates several outputs:

- **Processed Data**: Cleaned and preprocessed text data
- **Feature Data**: Extracted features from text
- **Trained Models**: Regression models to predict coffee ratings
- **Visualizations**: Various plots showing:
  - Rating distributions
  - Topic model results
  - Feature importance
  - Model performance comparison
  - Correlation analysis

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.