"""
Spam Classifier — Backend API
==============================
Flask REST API that wraps the TF-IDF + Multinomial Naive Bayes
spam/ham classifier and exposes it to the frontend.

Endpoints
---------
GET  /api/health              -> service + model status
GET  /api/stats                -> accuracy, per-class metrics, confusion matrix
GET  /api/messages              -> the labeled sample dataset the model trained on
POST /api/predict     {text}    -> {label, confidence: {ham, spam}}
POST /api/retrain     {rows?}   -> retrains on default data (or supplied rows), returns fresh stats

Run
---
    pip install flask flask-cors scikit-learn pandas --break-system-packages
    python app.py
    # serves on http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)
CORS(app)  # allow the static frontend (served from a different origin) to call this API

# ---------------------------------------------------------------------------
# Data + model
# ---------------------------------------------------------------------------

def load_sample_data():
    data = {
        "text": [
            "Win a free iPhone now, click this link immediately!!!",
            "Congratulations! You have won a $1000 Walmart gift card.",
            "URGENT: Your account has been suspended, verify now.",
            "Claim your free lottery prize before it expires today!",
            "Limited time offer, buy cheap meds online no prescription",
            "You have been selected for a free cruise, call now",
            "Get rich quick with this one simple trick, act now",
            "Hot singles in your area want to meet you tonight",
            "Hey, are we still meeting for lunch tomorrow?",
            "Please find attached the report you requested yesterday.",
            "Can you send me the meeting notes from today's call?",
            "Reminder: your dentist appointment is on Friday at 3pm.",
            "Let's catch up this weekend, it's been a while.",
            "The project deadline has been moved to next Monday.",
            "Thanks for your help with the presentation earlier.",
            "Mom, I'll be home late tonight, don't wait for dinner.",
            "Your Amazon order has shipped and will arrive tomorrow.",
            "I've attached the invoice for last month's services.",
            "Can we reschedule our 1:1 to Thursday afternoon?",
            "Free entry in a weekly contest, text WIN to 80085 now",
        ],
        "label": [
            "spam", "spam", "spam", "spam", "spam",
            "spam", "spam", "spam", "ham", "ham",
            "ham", "ham", "ham", "ham", "ham",
            "ham", "ham", "ham", "ham", "spam",
        ],
    }
    return pd.DataFrame(data)


def train_model(df):
    """Trains the pipeline and returns (model, stats_dict)."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True)),
        ("classifier", MultinomialNB()),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"]).tolist()

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": report,
        "confusion_matrix": {
            "labels": ["ham", "spam"],
            "matrix": cm,  # rows = actual [ham, spam], cols = predicted [ham, spam]
        },
        "train_size": len(X_train),
        "test_size": len(X_test),
        "class_counts": df["label"].value_counts().to_dict(),
    }
    return model, stats


# Train once at startup; /api/retrain can refresh this later.
_current_df = load_sample_data()
_model, _stats = train_model(_current_df)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_trained": _model is not None})


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(_stats)


@app.route("/api/messages", methods=["GET"])
def messages():
    return jsonify(_current_df.to_dict(orient="records"))


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty."}), 400

    prediction = _model.predict([text])[0]
    probabilities = _model.predict_proba([text])[0]
    confidence = {cls: float(p) for cls, p in zip(_model.classes_, probabilities)}

    return jsonify({
        "text": text,
        "label": prediction,
        "confidence": confidence,
    })


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Optionally accepts {"rows": [{"text": ..., "label": "spam"|"ham"}, ...]}
    to append new labeled examples before retraining. Without a body, just
    retrains on the default sample dataset (useful to reset)."""
    global _current_df, _model, _stats

    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows")

    df = load_sample_data()
    if rows:
        valid_rows = [
            r for r in rows
            if isinstance(r, dict) and r.get("text") and r.get("label") in ("spam", "ham")
        ]
        if valid_rows:
            df = pd.concat([df, pd.DataFrame(valid_rows)], ignore_index=True)

    _current_df = df
    _model, _stats = train_model(_current_df)

    return jsonify({"message": "Model retrained.", "stats": _stats})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
