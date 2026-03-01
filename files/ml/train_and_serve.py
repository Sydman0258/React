
import json
import os
import pickle
import io
import csv
import random
import math
from datetime import datetime, timedelta
from flask import Flask, request, jsonify

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

MODEL_PATH = "model.pkl"
ENCODERS_PATH = "encoders.pkl"


TEAMS = [
    "Manchester City", "Arsenal", "Liverpool", "Chelsea",
    "Manchester United", "Tottenham", "Newcastle", "Aston Villa",
    "Brighton", "West Ham", "Brentford", "Fulham",
    "Crystal Palace", "Wolves", "Bournemouth", "Nottingham Forest",
    "Everton", "Burnley", "Sheffield United", "Luton Town"
]

TEAM_STRENGTH = {
    "Manchester City": 0.90, "Arsenal": 0.82, "Liverpool": 0.85,
    "Chelsea": 0.76, "Manchester United": 0.74, "Tottenham": 0.73,
    "Newcastle": 0.70, "Aston Villa": 0.69, "Brighton": 0.67,
    "West Ham": 0.63, "Brentford": 0.62, "Fulham": 0.60,
    "Crystal Palace": 0.57, "Wolves": 0.56, "Bournemouth": 0.55,
    "Nottingham Forest": 0.54, "Everton": 0.52, "Burnley": 0.45,
    "Sheffield United": 0.43, "Luton Town": 0.40
}


def generate_dataset(n_matches=3000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    records = []

    for i in range(n_matches):
        home = random.choice(TEAMS)
        away = random.choice([t for t in TEAMS if t != home])

        hs = TEAM_STRENGTH[home]
        as_ = TEAM_STRENGTH[away]
        home_adv = 0.08  # home advantage

        # Simulate goals
        home_goals = np.random.poisson(max(0.3, (hs - as_) * 3 + 1.4 + home_adv))
        away_goals = np.random.poisson(max(0.3, (as_ - hs) * 3 + 1.0))

        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"

        # Feature engineering
        records.append({
            "home_team": home,
            "away_team": away,
            "home_strength": hs,
            "away_strength": as_,
            "strength_diff": hs - as_,
            "home_goals_scored_avg": round(hs * 2.1 + np.random.normal(0, 0.2), 2),
            "home_goals_conceded_avg": round((1 - hs) * 1.5 + np.random.normal(0, 0.2), 2),
            "away_goals_scored_avg": round(as_ * 1.8 + np.random.normal(0, 0.2), 2),
            "away_goals_conceded_avg": round((1 - as_) * 1.6 + np.random.normal(0, 0.2), 2),
            "home_form": round(hs + np.random.normal(0, 0.1), 2),  # last 5 games pts ratio
            "away_form": round(as_ + np.random.normal(0, 0.1), 2),
            "home_win_rate": round(hs * 0.6 + np.random.normal(0, 0.05), 2),
            "away_win_rate": round(as_ * 0.5 + np.random.normal(0, 0.05), 2),
            "head_to_head_home_wins": random.randint(0, 5),
            "head_to_head_draws": random.randint(0, 3),
            "home_position": int(21 - hs * 20),
            "away_position": int(21 - as_ * 20),
            "result": result
        })

    return pd.DataFrame(records)


def get_feature_columns():
    return [
        "home_strength", "away_strength", "strength_diff",
        "home_goals_scored_avg", "home_goals_conceded_avg",
        "away_goals_scored_avg", "away_goals_conceded_avg",
        "home_form", "away_form",
        "home_win_rate", "away_win_rate",
        "head_to_head_home_wins", "head_to_head_draws",
        "home_position", "away_position"
    ]


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
model = None
label_encoder = None
accuracy = 0.0
feature_importances = {}

def train_model():
    global model, label_encoder, accuracy, feature_importances

    print("📊 Generating dataset...")
    df = generate_dataset(3000)

    features = get_feature_columns()
    X = df[features]
    y = df["result"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    print("🤖 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    importances = model.feature_importances_
    feature_importances = dict(zip(features, [round(float(v), 4) for v in importances]))

    print(f"✅ Model trained! Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    return accuracy


def load_model():
    global model, label_encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ENCODERS_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        return True
    return False


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "accuracy": round(accuracy, 4)})


@app.route("/train", methods=["POST"])
def train():
    acc = train_model()
    return jsonify({
        "success": True,
        "accuracy": round(acc, 4),
        "accuracy_pct": f"{acc:.1%}",
        "feature_importances": feature_importances,
        "model": "RandomForestClassifier(n_estimators=200)"
    })


@app.route("/teams", methods=["GET"])
def get_teams():
    teams = [{"name": t, "strength": TEAM_STRENGTH[t]} for t in TEAMS]
    teams.sort(key=lambda x: -x["strength"])
    return jsonify({"teams": teams})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if model is None:
        return jsonify({"error": "Model not trained yet. Call /train first."}), 400

    data = request.get_json()
    home = data.get("home_team")
    away = data.get("away_team")

    if home not in TEAM_STRENGTH or away not in TEAM_STRENGTH:
        return jsonify({"error": "Unknown team"}), 400

    hs = TEAM_STRENGTH[home]
    as_ = TEAM_STRENGTH[away]

    features_dict = {
        "home_strength": hs,
        "away_strength": as_,
        "strength_diff": hs - as_,
        "home_goals_scored_avg": round(hs * 2.1, 2),
        "home_goals_conceded_avg": round((1 - hs) * 1.5, 2),
        "away_goals_scored_avg": round(as_ * 1.8, 2),
        "away_goals_conceded_avg": round((1 - as_) * 1.6, 2),
        "home_form": round(hs + np.random.normal(0, 0.05), 2),
        "away_form": round(as_ + np.random.normal(0, 0.05), 2),
        "home_win_rate": round(hs * 0.6, 2),
        "away_win_rate": round(as_ * 0.5, 2),
        "head_to_head_home_wins": data.get("h2h_home_wins", 2),
        "head_to_head_draws": data.get("h2h_draws", 1),
        "home_position": int(21 - hs * 20),
        "away_position": int(21 - as_ * 20),
    }

    X = pd.DataFrame([features_dict])[get_feature_columns()]
    proba = model.predict_proba(X)[0]
    classes = label_encoder.classes_  # e.g. ['A', 'D', 'H']

    result_map = {}
    for cls, prob in zip(classes, proba):
        result_map[cls] = round(float(prob), 4)

    predicted_idx = int(np.argmax(proba))
    predicted = classes[predicted_idx]
    labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

    # Expected goals (Poisson-based)
    exp_home_goals = round(max(0.3, (hs - as_) * 3 + 1.4 + 0.08), 2)
    exp_away_goals = round(max(0.3, (as_ - hs) * 3 + 1.0), 2)

    return jsonify({
        "home_team": home,
        "away_team": away,
        "prediction": predicted,
        "prediction_label": labels[predicted],
        "confidence": round(float(proba[predicted_idx]), 4),
        "probabilities": {
            "home_win": result_map.get("H", 0),
            "draw": result_map.get("D", 0),
            "away_win": result_map.get("A", 0),
        },
        "expected_goals": {
            "home": exp_home_goals,
            "away": exp_away_goals
        },
        "features_used": features_dict
    })


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Predict multiple upcoming fixtures at once."""
    if model is None:
        return jsonify({"error": "Model not trained yet."}), 400

    fixtures = request.get_json().get("fixtures", [])
    results = []
    for fix in fixtures:
        home = fix.get("home_team")
        away = fix.get("away_team")
        date = fix.get("date", "")
        if home not in TEAM_STRENGTH or away not in TEAM_STRENGTH:
            continue
        hs = TEAM_STRENGTH[home]
        as_ = TEAM_STRENGTH[away]
        features_dict = {
            "home_strength": hs, "away_strength": as_, "strength_diff": hs - as_,
            "home_goals_scored_avg": round(hs * 2.1, 2),
            "home_goals_conceded_avg": round((1 - hs) * 1.5, 2),
            "away_goals_scored_avg": round(as_ * 1.8, 2),
            "away_goals_conceded_avg": round((1 - as_) * 1.6, 2),
            "home_form": round(hs, 2), "away_form": round(as_, 2),
            "home_win_rate": round(hs * 0.6, 2), "away_win_rate": round(as_ * 0.5, 2),
            "head_to_head_home_wins": 2, "head_to_head_draws": 1,
            "home_position": int(21 - hs * 20), "away_position": int(21 - as_ * 20),
        }
        X = pd.DataFrame([features_dict])[get_feature_columns()]
        proba = model.predict_proba(X)[0]
        classes = label_encoder.classes_
        result_map = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        pred = classes[int(np.argmax(proba))]
        labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
        results.append({
            "home_team": home, "away_team": away, "date": date,
            "prediction": pred, "prediction_label": labels[pred],
            "confidence": round(float(np.max(proba)), 4),
            "probabilities": {
                "home_win": result_map.get("H", 0),
                "draw": result_map.get("D", 0),
                "away_win": result_map.get("A", 0),
            }
        })

    return jsonify({"predictions": results})


@app.route("/stats", methods=["GET"])
def stats():
    """Return model statistics and feature importances."""
    return jsonify({
        "model_trained": model is not None,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": f"{accuracy:.1%}",
        "feature_importances": dict(sorted(feature_importances.items(), key=lambda x: -x[1])),
        "teams_count": len(TEAMS),
        "training_samples": 3000
    })


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🏆 Football Prediction ML Service")
    print("=" * 40)
    if not load_model():
        print("No saved model found. Training now...")
        train_model()
    else:
        print("✅ Loaded saved model")
    print(f"🚀 Starting Flask on port 5001")
    app.run(port=5001, debug=False)
