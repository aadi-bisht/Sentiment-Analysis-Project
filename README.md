# Sentiment Analysis of Yelp Reviews

This repository contains a sentiment analysis project focused on Yelp reviews. Using Natural Language Processing (NLP), the model is trained to detect emotional tones in text, helping machines understand human language. Useful for extracting insights from customer feedback and improving user experience.

## Overview

This project explores various machine learning techniques to perform sentiment analysis on Yelp reviews. The models predict four key attributes from review text: `stars`, `useful`, `cool`, and `funny`. Each group member implemented a different classification technique:
- Neural Network (DistilBERT)
- Probabilistic (Naive Bayes)
- Non-Parametric (Decision Tree)

## Models Implemented

- **Neural Network (DistilBERT):** Fine-tuned transformer models using Hugging Face's Trainer API.
- **Naive Bayes:** Custom probabilistic model using Count Vectorizer for bag-of-words representation.
- **Decision Tree:** Non-parametric model using Count Vectorizer and GridSearchCV for hyperparameter tuning.

## Dataset

- **Source:** [`yelp_academic_dataset_review.json`](https://www.yelp.com/dataset)
- **Preprocessing Steps:**
  - Cleaning and normalization
  - Tokenization and vectorization (CountVectorizer, TF-IDF)
  - Removing invalid or missing labels
  - Splitting: 80% training, 10% validation, 10% testing

## Evaluation Metrics

- **Micro F1-Score** is used to evaluate the classification performance across all four labels.

## How to Run

### Requirements

Python 3.x and the following packages:

```
pip install argparse scikit-learn pandas numpy matplotlib nltk torch
```

Run a Model
```
python models.py -t <testset.json> -c <classifier> [-v <vectorizer.pickle>] [-m <model.pickle>]
```
Example (Naive Bayes - Stars):

```python models.py -t test.json -c nb -v pickles/Count_Vectorizer_top_15_dropped.pickle -m pickles/NaiveBayes_Stars_top_15_dropped.pickle```

Example (Decision Tree):
```python models.py -t test.csv -c dt```

## Experiments

Each model includes two experiments to evaluate performance under different settings:

### DistilBERT
- Epoch variation
- Input sequence length adjustment

### Naive Bayes
- Removing top-K frequent words
- Ablation study by dropping reviews with specific star ratings

### Decision Tree
- CountVectorizer vs TF-IDF comparison
- Evaluation on a hand-annotated review dataset

## Results

- **Best F1-score:** ~0.61 with optimized Naive Bayes
- DistilBERT performed best for edge sentiment classes (1 and 5 stars)
- Decision Tree worked best with unigrams and tuned depth

## Contributions

- **Kris Chan:** Decision Tree modeling, TF-IDF vectorization, hand-annotated dataset
- **Angus Lin:** Naive Bayes modeling, stopword and ablation experiments
- **Aadi Bisht:** DistilBERT fine-tuning, training strategy experiments
