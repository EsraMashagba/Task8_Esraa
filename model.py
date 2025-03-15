import torch 
import pandas as pd
import pickle
import mlflow
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm  # Progress bar
import os  # To check file existence
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Model & Tokenizer (only when needed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model.eval()

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\b\d+\b", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Function to classify news
def classify_news(texts, batch_size=8):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

        with torch.no_grad():
            outputs = finbert_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            impact_scores = probs.max(dim=1)[0].cpu().numpy()  # Take the highest probability as impact score

            # Convert numeric labels to sentiment strings
            sentiment_labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
            classified_results = [(sentiment_labels[pred], float(score)) for pred, score in zip(preds, impact_scores)]
            predictions.extend(classified_results)

        torch.cuda.empty_cache()
    
    return predictions

# Ensure dataset processing only runs when executing model.py directly
if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset("EsrMash/news")
    df = dataset["train"].to_pandas()
    print(f"Dataset loaded. Shape: {df.shape}")

    # Preprocess data
    print("Preprocessing data...")
    df["news"] = df["news"].apply(clean_text)
    df = df.dropna(subset=["news", "sentiment"])
    sentiment_mapping = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    df["sentiment_label"] = df["sentiment"].str.upper().map(sentiment_mapping)

    if df.empty:
        raise ValueError("Dataset is empty after preprocessing!")

    print(f"Data after preprocessing: {df.shape}")

    # Test classification
    df_sample = df.sample(n=5000, random_state=42)  # Test with 5K articles
    df_sample["predicted_sentiment"] = classify_news(df_sample["news"].tolist(), batch_size=16)

    # Evaluate
    sample_accuracy = accuracy_score(df_sample["sentiment_label"], df_sample["predicted_sentiment"])
    print(f"Sample Test Accuracy: {sample_accuracy:.2f}")
    print(classification_report(
        df_sample["sentiment_label"], df_sample["predicted_sentiment"], 
        target_names=["negative", "neutral", "positive"], 
        zero_division=1
    ))

    # Save model
    print("Saving model...")
    model_path = "finbert_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(finbert_model, f)
    print(f"Model saved as {model_path}")

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_metric("sample_accuracy", sample_accuracy)
        mlflow.log_param("batch_size", 16)
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)
        else:
            print(f"Warning: {model_path} not found, skipping artifact logging.")

    print("Model and metrics logged in MLflow.")
