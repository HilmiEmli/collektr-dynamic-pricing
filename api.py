from __future__ import annotations

import json
import os
from typing import Any

from flask import Flask, jsonify, request

from src.config import CSV_SEPARATOR, DATA_PATH, DATE_COLUMN, ENTITY_COLUMN, MODEL_DIR, PRICE_COLUMN
from src.dynamic_pricing import load_pricing_data, predict_tomorrow, train_models


app = Flask(__name__)


@app.get("/")
def index() -> tuple[Any, int]:
    return jsonify(
        {
            "name": "Pokemon AI Dynamic Pricing API",
            "status": "running",
            "endpoints": {
                "GET /health": "Check API status.",
                "POST /metrics": "Read model metrics.",
                "POST /predict": "Predict all cards or one card.",
                "POST /train": "Retrain the models.",
            },
        }
    ), 200


@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.post("/metrics")
def metrics() -> tuple[Any, int]:
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return jsonify({"metrics": {}}), 200
    return jsonify({"metrics": json.loads(metrics_path.read_text(encoding="utf-8"))}), 200


@app.post("/train")
def train() -> tuple[Any, int]:
    try:
        df = load_pricing_data(DATA_PATH, DATE_COLUMN, PRICE_COLUMN, CSV_SEPARATOR)
        result = train_models(df, DATE_COLUMN, PRICE_COLUMN, MODEL_DIR, ENTITY_COLUMN)
        return jsonify({"best_model": result.model_name, "metrics": result.metrics}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/predict")
def predict() -> tuple[Any, int]:
    try:
        payload = request.get_json(silent=True) or {}
        df = load_pricing_data(DATA_PATH, DATE_COLUMN, PRICE_COLUMN, CSV_SEPARATOR)
        return jsonify({"predictions": predict_tomorrow(df, MODEL_DIR, payload.get("item"))}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
