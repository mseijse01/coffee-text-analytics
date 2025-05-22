#!/usr/bin/env python3
"""
Script to visualize LDA topics from the coffee text analytics project.
"""

import pickle
import sys

# Load the models
try:
    print("Loading LDA model...")
    with open("models/lda_model.pkl", "rb") as f:
        lda_model = pickle.load(f)

    print("Loading TF-IDF vectorizer...")
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Print topics
    print("\nTop 10 keywords for each LDA topic:\n")
    print("-" * 60)

    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Topic #{topic_idx + 1}:")
        top_keywords = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        print(f"  {', '.join(top_keywords)}")
        print("-" * 60)

except FileNotFoundError:
    print("Error: Model files not found. Run the pipeline first with:")
    print("python3 main.py --steps preprocess features")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
