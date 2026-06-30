# AI-Powered Fake News Detection Using Text Classification

A machine learning project that builds a complete fake news detection pipeline from scratch using Natural Language Processing (NLP) and classical Machine Learning algorithms.

The project performs manual text preprocessing, multiple feature extraction techniques, model training, evaluation, and comparison of both parametric and non-parametric classifiers.

---

## Objective

The objective of this project is to classify news articles as **Real** or **Fake** using text classification techniques without relying on an end-to-end pre-built fake news detection system.

The implementation includes:

- Manual text preprocessing
- Exploratory Data Analysis (EDA)
- Bag-of-Words (BoW)
- Manual TF-IDF implementation
- TF-IDF Vectorization
- Word2Vec embeddings
- Model training and comparison
- Performance evaluation
- Confusion matrices
- Test sample generation

---

## Dataset

**Dataset:** ISOT Fake News Dataset

Source:
- Kaggle – Fake News Detection Datasets by Emine Bozkus

Original files:

- `True.csv`
- `Fake.csv`

After preprocessing:

- Total Articles: **38,646**
- Real Articles: **21,192**
- Fake Articles: **17,454**

---

## Project Structure

```
AI-Powered-Fake-News-Detection/
│
├── data/
│   └── raw/
│       ├── Fake.csv
│       └── True.csv
│
├── notebooks/
│   └── FakeNewsDetection.ipynb
│
├── output/
│   ├── model_results.csv
│   └── test_data_samples.csv
│
├── report/
│   ├── AI_Powered_Fake_News_Detection_Report.pdf
│   ├── report.tex
│   └── assets/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Preprocessing

The preprocessing pipeline includes:

- Lowercasing
- URL removal
- HTML tag removal
- Punctuation removal
- Digit removal
- Space normalization
- Manual tokenization
- Manual stopword removal
- Reuters dateline removal (to reduce dataset leakage)

---

## Feature Extraction

Three feature representations were implemented:

- Bag-of-Words (BoW)
- TF-IDF
- Word2Vec

The primary classifier uses TF-IDF features with 5,000 dimensions.

---

## Machine Learning Models

The following models were implemented and evaluated:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest
- Multi-layer Perceptron (Neural Network)

---

## Results

| Model | Accuracy |
|--------|----------|
| Neural Network | **98.89%** |
| Random Forest | **98.49%** |
| Logistic Regression | **98.40%** |
| KNN | **68.46%** |

Neural Network achieved the highest overall performance with an F1-score of **0.9898**.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Gensim
- Matplotlib
- Seaborn

---

## Files

| File | Description |
|------|-------------|
| FakeNewsDetection.ipynb | Complete implementation |
| AI_Powered_Fake_News_Detection_Report.pdf | IEEE-style project report |
| model_results.csv | Model evaluation metrics |
| test_data_samples.csv | Sample predictions |

---

## Future Improvements

- Transformer-based models (BERT, RoBERTa)
- Hyperparameter optimization
- Multi-source datasets
- Dense document embeddings
- Improved document-level Word2Vec features

---

## Author

Snigdha