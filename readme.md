# Coffee Text Analytics

Coffee Text Analytics is a portfolio project that leverages natural language processing and advanced feature engineering to analyze coffee reviews. The project uncovers insights into coffee quality and ratings by combining numerical data with rich textual descriptions.

---

## Overview

This project aims to understand what makes a great cup of coffee by analyzing a dataset of coffee reviews. It uses multiple techniques to extract meaningful features from both structured data (e.g., ratings, sensory scores, pricing) and unstructured text (descriptive reviews). The work is organized into several key phases:

- **Exploratory Data Analysis:** Investigating distributions, correlations, and patterns in ratings, sensory attributes, pricing, and geographical origins.
- **Data Cleaning & Preprocessing:** Standardizing prices, handling missing values, extracting country information, and cleaning textual data.
- **Feature Engineering:** Extracting textual features using:
  - **TF-IDF:** To capture important coffee descriptors.
  - **BERT Embeddings:** For rich semantic representations of the review texts.
  - **Topic Modeling (LDA & NMF):** To discover themes across reviews.
  - **Sentiment Analysis:** To quantify reviewer sentiment using DistilBERT.
- **Visualization & Insights:** Creating interactive and static plots to communicate key findings.

---

## Dataset

The dataset is stored in the `data/raw/` folder as `coffee_clean.csv` and contains:
- **Ratings:** Overall scores for the coffee.
- **Sensory Attributes:** Scores for aroma, acid, body, flavor, and aftertaste.
- **Pricing Information:** Including an estimated price that is standardized to USD per kilogram.
- **Text Descriptions:** Detailed coffee reviews in three columns (`desc_1`, `desc_2`, `desc_3`).
- **Additional Metadata:** Such as roast level and country of origin.

---

## Project Structure