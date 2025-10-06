This project focuses on comparing the performance of three deep learning models — LSTM, BERT, and RoBERTa — for sentiment analysis on textual reviews.
The goal is to determine which model achieves the best performance in predicting positive, neutral, and negative sentiments.
Objectives

Build and train three different models for sentiment classification:

* LSTM – A sequential deep learning baseline.

* BERT – A transformer-based model with contextual embeddings.

* RoBERTa – An optimized version of BERT trained on a larger corpus.

Evaluate and compare their accuracy, precision, recall, and F1-score.

Visualize model performance metrics for clearer comparison.

The dataset consists of text reviews labeled with sentiment categories.
Each entry contains:

review: textual content

sentiment: sentiment label (negative, neutral, or positive)

LSTM

Tokenized text data and padded sequences.

Embedding + LSTM + Dense layers.

Loss: Categorical Crossentropy, Optimizer: Adam.

BERT

Tokenized using BertTokenizer (bert-base-uncased).

Fine-tuned TFBertForSequenceClassification model.

Optimized with learning rate 3e-5 and early stopping.

RoBERTa

Tokenized using RobertaTokenizer (roberta-base).

Fine-tuned TFRobertaForSequenceClassification.

Same setup as BERT for fair comparison.

Training & Evaluation

Each model was trained on the same training/validation split.
After training:

Predictions were generated on the validation set.

classification_report and confusion matrices were used for evaluation.

Metrics were visualized with bar plots (precision, recall, F1).

| Model   | Accuracy     | Precision | Recall | macro F1-Score |
| ------- | ------------ | --------- | ------ | -------------- |
| LSTM    | *42.3%*      | *13%*     | *33%*  | *19%*          |
| BERT    | *98%*        | *98%*     | *98%*  | *98%*          |
| RoBERTa | *98.7%*      | *99%*     | *99%*  | *99%*          |


How to Run

1. Clone the repository
git clone https://github.com/MaryamMazareii/Sentiment-analysis-on-reviews.git

cd Sentiment-analysis-on-reviews

2. Install dependencies:
pip install -r requirements.txt

3. Run each notebook or script:
* LSTM_Model.ipynb
* BERT_Model.ipynb
* RoBERTa_Model.ipynb

4. View and compare metrics in the generated output cells or plots.

Visualizations

Example performance plot showing precision, recall, and F1-score for each sentiment class.

Future Work

* Expand dataset with more diverse reviews.
* Experiment with hyperparameter tuning and dropout.
* Test additional transformer models (DistilBERT, ALBERT, etc.).

Author:
Maryam Mazarei
Computer Engineering Student | Deep Learning Enthusiast
