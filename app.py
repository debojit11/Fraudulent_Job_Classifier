import streamlit as st
import re
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

st.set_page_config(page_title="Job Fraud Detector", layout="centered")

# Sidebar navigation
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", ["🔍 Analyze a Job Post", "📊 Model Performance Dashboard", "🗂️ Bulk CSV Checker", "📝 Feedback"])

# ==== Load Models ====
@st.cache_resource
def load_bert_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/tokenizer", local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained("models/fraud_job_distilbert", local_files_only=True)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_logistic_model():
    return joblib.load("models/job_fraud_logistic.joblib")

@st.cache_resource
def load_embedding_models():
    models = {
        "Random Forest": joblib.load("models/embedding_random_forest_job_fraud.joblib"),
        "Gradient Boosting": joblib.load("models/embedding_gradient_boosting_job_fraud.joblib"),
        "MLP Classifier": joblib.load("models/embedding_mlp_job_fraud.joblib"),
        "XGBoost": joblib.load("models/embedding_xgboost_job_fraud.joblib")
    }
    return models

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# def clean_text(text):
#     text = str(text)
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)   # Remove URLs
#     text = re.sub(r'\@\w+|\#', '', text)   # Remove mentions, hashtags
#     text = re.sub(r"[^a-zA-Z0-9\s]", '', text)   # Remove most special characters
#     text = re.sub(r'mso\S+|ascii\S+|font\S+|times new roman|calibri|minorlatin|engt|ntte\d+q|c\s+', '', text) # Remove specific formatting strings
#     text = re.sub(r'email\w+|phonew+', '', text) # Remove potential email and phone placeholders
#     text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
#     return text.strip()


# Load resources
tokenizer, bert_model = load_bert_model()
logreg_model = load_logistic_model()
embedding_models = load_embedding_models()
embedder = load_embedder()

# ==== Prediction Functions ====
def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0].tolist()

def predict_logreg(text):
    probs = logreg_model.predict_proba([text])[0]
    return probs.tolist()

def predict_embedding_model(model, embedding_vector):
    probs = model.predict_proba([embedding_vector])[0]
    return probs.tolist()

# ==== PAGE 1: Analyze Job Post ====
if page == "🔍 Analyze a Job Post":
    st.title("🕵️‍♂️ Job Fraud Detector")
    st.write("Enter a job description and compare predictions from different models:")

    if "text" not in st.session_state:
        st.session_state.text = ""
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False

    st.session_state.text = st.text_area("Job Description", value=st.session_state.text, height=200)

    if st.button("Analyze"):
        if st.session_state.text.strip() == "":
            st.warning("Please enter a job description!")
        else:
            st.session_state.analyzed = True

    if st.session_state.analyzed:
        raw_text = st.session_state.text
        text = raw_text
        st.subheader("🔍 Results")


# Make all predictions first
        bert_probs = predict_bert(text)
        logreg_probs = predict_logreg(text)
        embedding_vector = embedder.encode(text)

        # Collect fraud probabilities from all models
        fraud_probs = {
            "DistilBERT": bert_probs[1],
            "Logistic Regression (TF-IDF)": logreg_probs[1]
        }
        for name, model in embedding_models.items():
            probs = predict_embedding_model(model, embedding_vector)
            fraud_probs[name] = probs[1]

        predictions = {model: (prob > 0.5) for model, prob in fraud_probs.items()}
        fraud_count = sum(predictions.values())
        non_fraud_count = len(predictions) - fraud_count

        # Final Verdict
        if fraud_count >= 4:
            st.markdown(
                '<h3 style="text-align:center;">This job is likely to be <span style="color:red; font-size: 2em;"><b><i>Fraud</i></b></span></h3>',
                unsafe_allow_html=True
            )
        elif non_fraud_count >= 4:
            st.markdown(
                '<h3 style="text-align:center;">This job is likely to be <span style="color:green; font-size: 2em;"><b><i>Legit</i></b></span></h3>',
                unsafe_allow_html=True
            )

        # === DistilBERT ===
        st.markdown("**🤖 DistilBERT**")
        bert_probs = predict_bert(text)
        st.write(f"Not Fraud: **{bert_probs[0]:.2f}**")
        st.write(f"Fraud: **{bert_probs[1]:.2f}**")
        st.progress(bert_probs[1])

        # === Logistic Regression (TF-IDF) ===
        st.markdown("**📊 Logistic Regression (TF-IDF)**")
        logreg_probs = predict_logreg(text)
        st.write(f"Not Fraud: **{logreg_probs[0]:.2f}**")
        st.write(f"Fraud: **{logreg_probs[1]:.2f}**")
        st.progress(logreg_probs[1])

        # === Embedding Vector ===
        embedding_vector = embedder.encode(text)

        # === Embedding-Based Models ===
        for name, model in embedding_models.items():
            st.markdown(f"**🧠 {name}**")
            probs = predict_embedding_model(model, embedding_vector)
            st.write(f"Not Fraud: **{probs[0]:.2f}**")
            st.write(f"Fraud: **{probs[1]:.2f}**")
            st.progress(probs[1])

        st.success("All models evaluated! ✅")

        # === Disagreement Detector ===
        fraud_probs = {
            "DistilBERT": bert_probs[1],
            "Logistic Regression (TF-IDF)": logreg_probs[1]
        }

        for name, model in embedding_models.items():
            probs = predict_embedding_model(model, embedding_vector)
            fraud_probs[name] = probs[1]

        predictions = {model: (prob > 0.5) for model, prob in fraud_probs.items()}
        fraud_count = sum(predictions.values())
        non_fraud_count = len(predictions) - fraud_count

        st.subheader("⚔️ Model Disagreement Detector")

        if fraud_count == 0 or non_fraud_count == 0:
            st.success("✅ All models agree: " + ("Fraud" if fraud_count > 0 else "Not Fraud"))
        else:
            st.warning("⚠️ Models disagree!")
            st.write(f"{fraud_count} model(s) predict **Fraud**, {non_fraud_count} predict **Not Fraud**")
            st.markdown("**🧪 Individual Predictions:**")
            for model, is_fraud in predictions.items():
                label = "Fraud" if is_fraud else "Not Fraud"
                st.write(f"- **{model}**: {label}")

# ==== PAGE 2: Model Performance Dashboard ====
elif page == "📊 Model Performance Dashboard":
    st.title("📊 Model Performance Dashboard")
    st.write("Compare evaluation metrics across all trained models.")

    metrics_data = {
        "Model": [
            "DistilBERT",
            "Logistic Regression (TF-IDF)",
            "Random Forest (Embeddings)",
            "Gradient Boosting (Embeddings)",
            "MLP Classifier (Embeddings)",
            "XGBoost (Embeddings)"
        ],
        "Precision (0)": [1.00, 0.97, 1.00, 0.94, 1.00, 1.00],
        "Precision (1)": [1.00, 0.96, 1.00, 0.95, 0.98, 1.00],
        "Recall (0)":    [1.00, 0.97, 1.00, 0.97, 0.99, 1.00],
        "Recall (1)":    [1.00, 0.96, 1.00, 0.91, 1.00, 1.00],
        "F1-Score (0)":  [1.00, 0.97, 1.00, 0.95, 0.99, 1.00],
        "F1-Score (1)":  [1.00, 0.96, 1.00, 0.93, 0.99, 1.00],
        "Accuracy":      [1.00, 0.97, 1.00, 0.94, 0.99, 1.00]
    }

    df_metrics = pd.DataFrame(metrics_data)

    st.dataframe(df_metrics, use_container_width=True)

    metric_option = st.selectbox(
        "Select a metric to visualize:",
        ["Accuracy", "Precision (0)", "Precision (1)", "Recall (0)", "Recall (1)", "F1-Score (0)", "F1-Score (1)"]
    )

    st.markdown(f"### 🔬 Comparing: {metric_option}")
    fig, ax = plt.subplots()
    ax.bar(df_metrics["Model"], df_metrics[metric_option], color="skyblue")
    ax.set_ylabel(metric_option)
    ax.set_ylim(0.85, 1.05)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    st.subheader("🧮 Confusion Matrix Viewer")

    confusion_data = {
        "DistilBERT": np.array([[3394, 9], [0, 2269]]),
        "Logistic Regression (TF-IDF)": np.array([[3302, 101], [91, 2178]]),
        "Random Forest (Embeddings)": np.array([[3402, 1], [0, 2269]]),
        "Gradient Boosting (Embeddings)": np.array([[3288, 115], [209, 2060]]),
        "MLP Classifier (Embeddings)": np.array([[3366, 37], [0, 2269]]),
        "XGBoost (Embeddings)": np.array([[3398, 5], [0, 2269]])
    }

    selected_model = st.selectbox("Select a model to view its confusion matrix:", list(confusion_data.keys()))
    cm = confusion_data[selected_model]

    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_title(f"Confusion Matrix - {selected_model}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig_cm)

# ==== PAGE 3: CSV Upload and Bulk Check ====
elif page == "🗂️ Bulk CSV Checker":
    st.title("🗂️ Bulk Job Description Checker")
    st.markdown("**📌 The uploaded CSV file must contain only one column named `description`.**")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'description' column", type=["csv"])

    model_name = st.selectbox("Choose a model", [
        "DistilBERT", "Logistic Regression (TF-IDF)",
        "Random Forest", "Gradient Boosting", "MLP Classifier", "XGBoost"
    ])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'description' not in df.columns:
            st.error("CSV must contain a 'description' column.")
        else:
            df['cleaned_description'] = df['description'].apply(clean_text)

            with st.spinner("Predicting..."):
                if model_name == "DistilBERT":
                    df['fraud_probability'] = df['cleaned_description'].apply(lambda x: predict_bert(str(x))[1])
                elif model_name == "Logistic Regression (TF-IDF)":
                    df['fraud_probability'] = df['cleaned_description'].apply(lambda x: predict_logreg(str(x))[1])
                else:
                    model = embedding_models[model_name]
                    df['fraud_probability'] = df['cleaned_description'].apply(lambda x: predict_embedding_model(model, embedder.encode(str(x)))[1])

                df['prediction'] = df['fraud_probability'].apply(lambda x: "Fraud" if x > 0.5 else "Not Fraud")
                st.success("✅ Predictions complete!")
                st.dataframe(df[["description", "fraud_probability", "prediction"]], use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    st.markdown("💡 *For confusing or edge-case descriptions (e.g., short, vague, or unclear job posts), we recommend trying **Logistic Regression (TF-IDF)** or **MLP Classifier** for more robust results.*")




elif page == "📝 Feedback":
    st.title("📝 Feedback")
    st.write("We'd love to hear what you think! Please rate your experience and share suggestions below:")

    # Embed the Google Form
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSeyNnIS8XyY6eMB6SECoqyXrsN6mPpqTGumo7b_rns1IccALw/viewform?embedded=true"
    components.iframe(form_url, height=900)
