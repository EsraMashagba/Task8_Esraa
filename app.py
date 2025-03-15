from flask import Flask, request, jsonify
import torch
import pickle
from transformers import AutoTokenizer
import torch.nn.functional as F
from pydantic import BaseModel
from model import classify_news
from ranking import rank_news

# Load model and tokenizer
print("Loading model...")
with open("finbert_model.pkl", "rb") as f:
    finbert_model = pickle.load(f)
finbert_model.eval()

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# Request Model (Not used in Flask directly)
class NewsRequest(BaseModel):
    news: str

# API Root
@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"message": "Welcome to the Financial News Ranking API!"})

# API Endpoint for News Classification
@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if "news" not in data:
        return jsonify({"error": "Missing 'news' field"}), 400
    
    sentiment, impact_score = classify_news([data["news"]])[0]
    return jsonify({"news": data["news"], "sentiment": sentiment, "impact_score": impact_score})

# API Endpoint for News Ranking
@app.route("/rank", methods=["POST"])
def rank():
    data = request.get_json()
    if not isinstance(data, list) or not all("news" in item for item in data):
        return jsonify({"error": "Invalid input format. Expected a list of objects with 'news' field."}), 400

    ranked_news = rank_news([item["news"] for item in data])
    return jsonify({"ranked_news": ranked_news})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    

