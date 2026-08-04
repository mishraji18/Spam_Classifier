"""
Spam Email Classifier — Supervised Learning
=============================================
Uses TF-IDF vectorization + Multinomial Naive Bayes to classify
emails/messages as "spam" or "ham" (not spam).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True)),
        ("classifier", MultinomialNB()),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

    return model


def predict_message(model, message):
    prediction = model.predict([message])[0]
    probability = model.predict_proba([message])[0]
    classes = model.classes_
    confidence = dict(zip(classes, probability))
    return prediction, confidence


def main():
    df = load_sample_data()
    print(f"Loaded {len(df)} labeled messages ({df['label'].value_counts().to_dict()})\n")

    model = train_model(df)

    print("\n=== Testing on New Messages ===")
    test_messages = [
        "Claim your free prize now, limited time only!",
        "Hey, can you review my code before the standup?",
        "You've won a free vacation, click here to claim",
        "Let's grab coffee sometime next week",
    ]

    for msg in test_messages:
        label, confidence = predict_message(model, msg)
        print(f"\nMessage: \"{msg}\"")
        print(f"Prediction: {label.upper()}")
        print(f"Confidence: {', '.join(f'{k}={v:.2f}' for k, v in confidence.items())}")


if __name__ == "__main__":
    main()