

# 📊 Financial Sentiment Analysis with FinBERT

This project applies **Natural Language Processing (NLP)** techniques to analyze sentiment in financial text (headlines & news).
We benchmark a **baseline machine learning model (TF-IDF + Logistic Regression)** against **FinBERT**, a transformer model pre-trained specifically for financial sentiment classification.

---

## 🚀 Project Overview

Financial markets are heavily influenced by **news sentiment**. Positive or negative tone in financial text can drive investor decisions, stock prices, and market volatility.
This notebook explores:

1. **Dataset** → [Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news), containing sentences labeled as *positive, neutral,* or *negative*.
2. **Baseline Model** → TF-IDF + Logistic Regression.
3. **Advanced Model** → [FinBERT](https://huggingface.co/ProsusAI/finbert), a domain-specific BERT variant trained on financial texts.
4. **Evaluation** → Accuracy, precision, recall, F1-score, and confusion matrices.
5. **Comparison** → Performance gap between baseline vs FinBERT.
6. **Interpretability** → Example predictions showing how models classify financial headlines.

---

## 📂 Dataset

* **Source**: Kaggle → [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
* **Size**: \~5,000 labeled sentences
* **Labels**:

  * `positive`
  * `neutral`
  * `negative`

---

## ⚙️ Methodology

### 1️⃣ Baseline: TF-IDF + Logistic Regression

* Convert sentences to TF-IDF vectors (unigrams + bigrams).
* Train Logistic Regression with balanced class weights.
* Evaluate on accuracy, precision, recall, F1-score.

### 2️⃣ FinBERT: Transformer Model

* Load **ProsusAI/FinBERT** from Hugging Face.
* Tokenize sentences with padding/truncation.
* Perform forward pass through FinBERT to get sentiment logits.
* Apply **softmax** to compute probabilities.
* Assign predicted label (positive / neutral / negative).

---

## 📊 Results

### 🔹 Model Performance

| Model                  | Accuracy | Notes                       |
| ---------------------- | -------- | --------------------------- |
| Baseline (TF-IDF + LR) | \~75–80% | Traditional ML approach     |
| FinBERT (Hugging Face) | \~85–90% | Domain-specific transformer |

*(Exact numbers depend on random seed & train/test split.)*

### 🔹 Confusion Matrices

Both models are evaluated with confusion matrices to analyze misclassifications:

* **Baseline**: Struggles to differentiate *neutral* vs *positive*.
* **FinBERT**: Much better at distinguishing sentiment tones.

---

## 🔍 Example Predictions

```
Text: "The company reported higher than expected quarterly earnings."
True Label: positive
FinBERT Prediction: positive ✅

Text: "The firm issued a profit warning due to weak demand."
True Label: negative
FinBERT Prediction: negative ✅

Text: "The board will review the current strategy next month."
True Label: neutral
FinBERT Prediction: neutral ✅
```

---

## 🛠️ Tech Stack

* **Python**
* **Pandas / NumPy / Scikit-Learn** → preprocessing & baseline model
* **Matplotlib / Seaborn** → visualization
* **Hugging Face Transformers** → FinBERT model & tokenizer
* **PyTorch** → model inference

---

## 📌 Key Takeaways

* **Baseline models** are simple and interpretable but limited in performance.
* **FinBERT**, being pre-trained on financial text, provides significant accuracy improvement.
* NLP models can be directly applied to corporate finance research, trading strategies, and market analysis.

---

## 📈 Future Work

* Fine-tune FinBERT on additional financial datasets (e.g., earnings call transcripts).
* Incorporate **time-series market data** (e.g., stock returns) to link sentiment → price movement.
* Deploy as a **real-time sentiment analysis API** for financial news feeds.

---

## 🤝 Acknowledgments

* [Financial PhraseBank Dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
* [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)

---

