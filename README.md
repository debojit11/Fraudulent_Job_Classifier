# Fraudulent_Job_Classifier

## 🌐 Live Demo
🚀 Try the app here: [fraud-job-classifier.streamlit.app](https://fraud-job-classifier.streamlit.app/)

✅ **Deployed on Streamlit Community Cloud**

## 📋 Overview
This project implements machine learning models to detect fraudulent job postings. Using natural language processing (NLP) techniques, multiple models have been trained and evaluated to identify potentially deceptive job advertisements, helping job seekers avoid scams.

## 🎯 Problem Statement
Online job platforms are often targeted by scammers posting fake job opportunities. These fraudulent postings can lead to identity theft, financial loss, and wasted time for job seekers. This project aims to create a reliable AI system that can automatically flag suspicious job postings.

## 🚀 Features
- Multiple ML models for fraud detection:
  - DistilBERT transformer model
  - Logistic Regression with TF-IDF
  - Embedding-based models (Random Forest, Gradient Boosting, MLP, XGBoost)
- Interactive web application with Streamlit
- Bulk processing capabilities for CSV files
- Model comparison dashboard
- Explainability features using LIME

## 💻 Technology Stack
- Python 3.8+
- PyTorch & Transformers
- Scikit-learn
- Sentence Transformers
- Streamlit
- Pandas & NumPy
- Matplotlib & Seaborn

## 📊 Model Performance
| Model | Accuracy | F1-Score (Fraud) |
|-------|----------|------------------|
| DistilBERT | 1.00 | 1.00 |
| Random Forest (Embeddings) | 1.00 | 1.00 |
| XGBoost (Embeddings) | 1.00 | 1.00 |
| MLP Classifier (Embeddings) | 0.99 | 0.99 |
| Logistic Regression (TF-IDF) | 0.97 | 0.96 |
| Gradient Boosting (Embeddings) | 0.94 | 0.93 |

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/job-fraud-detection.git
   cd job-fraud-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained models or train your own:
   ```bash
   # To train models
   python train.py
   ```

## 🖥️ Usage

### Running the Web App
```bash
streamlit run app.py
```

### Using the API
```python
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load the model
tokenizer = DistilBertTokenizerFast.from_pretrained("models/tokenizer")
model = DistilBertForSequenceClassification.from_pretrained("models/fraud_job_distilbert")

# Function for prediction
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0].tolist()

# Example
job_description = "Work from home opportunity! Make $5000 weekly with no experience!"
result = predict(job_description)
print(f"Fraud probability: {result[1]:.2f}")
```

## 🌐 Web Application Features
The Streamlit web application provides:

1. **Job Post Analyzer**: Analyze a single job posting with all models
2. **Model Performance Dashboard**: Compare metrics across models
3. **Bulk CSV Checker**: Process multiple job descriptions at once
4. **Feedback Form**: Submit feedback about the application

## 📁 Project Structure
```
job-fraud-detection/
├── app.py                      # Streamlit web application
├── train.ipynb                 # TF-IDF and DistilBERT training notebook
├── train_with_embed.ipynb      # Embedding models training notebook
├── test.ipynb                  # Testing and explainability notebook
├── requirements.txt            # Project dependencies
├── models/                     # Saved ML models
│   ├── tokenizer/              # DistilBERT tokenizer
│   ├── fraud_job_distilbert/   # DistilBERT model
│   ├── job_fraud_logistic.joblib
│   ├── embedding_random_forest_job_fraud.joblib
│   ├── embedding_gradient_boosting_job_fraud.joblib
│   ├── embedding_mlp_job_fraud.joblib
│   └── embedding_xgboost_job_fraud.joblib
└── README.md                   # Project documentation
```

## 🔄 Data Pipeline
1. **Data Preprocessing**:
   - Text cleaning (removing URLs, special characters)
   - Lowercasing
   - Text normalization

2. **Feature Extraction**:
   - TF-IDF vectorization
   - Sentence embeddings (all-MiniLM-L6-v2)

3. **Model Training**:
   - Multiple models trained with different techniques
   - Performance evaluation and model selection

4. **Inference**:
   - Real-time prediction using the trained models
   - Ensemble decision-making for higher accuracy

## 🙌 Future Improvements
- Implement active learning to improve model accuracy over time
- Add multilingual support for global job markets
- Integrate with email systems for automated scanning
- Develop browser extensions for real-time checking while browsing job sites

## 👏 Acknowledgements
- HuggingFace for the transformers library
- Sentence-Transformers for the embedding models
- Streamlit for the web application framework