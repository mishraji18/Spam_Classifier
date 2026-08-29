# Spam Classifier — Sorting Room

A small full-stack app around your existing TF-IDF + Multinomial Naive Bayes
spam/ham classifier: a Flask JSON API on the backend, and a single-page
"mail sorting desk" UI on the frontend that stamps each message SPAM or HAM. 
 
```
spam-classifier-app/ 
├── backend/ 
│   ├── app.py            # Flask API (wraps your original spam_classifier.py logic)
│   └── requirements.txt
└── frontend/
    └── index.html         # static UI — no build step, just open in a browser
```

## Run the backend

```bash
cd backend
pip install -r requirements.txt        # or: pip install -r requirements.txt --break-system-packages
python app.py
```

Serves at `http://localhost:5000`. On startup it trains the same pipeline as
`spam_classifier.py` (TF-IDF → MultinomialNB) on the built-in 20-message
sample dataset.

## Run the frontend

Just open `frontend/index.html` in a browser (double-click it, or serve the
folder with any static server, e.g. `python -m http.server` from inside
`frontend/`). It talks to the backend at `http://localhost:5000` — edit the
`API_BASE` constant near the top of the `<script>` block in `index.html` if
your backend runs elsewhere.

## API

| Method | Route            | Body                                    | Returns                                  |
|--------|------------------|------------------------------------------|-------------------------------------------|
| GET    | `/api/health`    | –                                        | `{status, model_trained}`                 |
| GET    | `/api/stats`     | –                                        | accuracy, per-class report, confusion matrix |
| GET    | `/api/messages`  | –                                        | the labeled training rows                 |
| POST   | `/api/predict`   | `{"text": "..."}`                        | `{text, label, confidence: {ham, spam}}`  |
| POST   | `/api/retrain`   | `{"rows": [{"text":..., "label":...}]}` (optional) | retrains, returns fresh `/api/stats` shape |

## Notes

- CORS is enabled on the backend so the static frontend (a different origin,
  e.g. `file://` or a local dev server) can call it directly.
- `/api/retrain` lets you grow the training set at runtime — pass new
  `{text, label}` rows and it appends them to the base 20-message dataset
  before refitting. There's no persistence layer, so a server restart resets
  it back to the original sample data; swap `load_sample_data()` for a real
  dataset/DB read if you want it to stick.
- The 20-row sample dataset is intentionally tiny (it's what your original
  script shipped with), so accuracy on the held-out test split will look
  noisy — this is meant as a working scaffold, not a production-grade model.
