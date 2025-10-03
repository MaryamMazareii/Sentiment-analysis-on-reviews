Sentiment Analysis with BERT

This repository contains a sentiment analysis project using BERT (Bidirectional Encoder Representations from Transformers) implemented in TensorFlow. The model classifies text reviews into three categories: positive, neutral, and negative.

Features:

Preprocessing of text reviews (tokenization, padding, attention masks)

Label encoding for classification

BERT-based model fine-tuned for sentiment analysis

Train/test split with evaluation metrics including:

Accuracy

Confusion Matrix (with heatmap visualization)

Precision, Recall, F1-score using classification_report

Inference on new examples

Optionally includes early stopping to prevent overfitting

Libraries Used:

TensorFlow

Transformers (Hugging Face)

scikit-learn

pandas, numpy, seaborn, matplotlib
