from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

from src.config import CSV_SEPARATOR, DATA_PATH, DATE_COLUMN, ENTITY_COLUMN, MODEL_DIR, PRICE_COLUMN
from src.dynamic_pricing import load_pricing_data, predict_tomorrow, train_models


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
MAX_CUSTOM_HISTORY_ROWS = 5000


@app.get("/")
def index() -> tuple[Any, int]:
    return jsonify(
        {
            "name": "Pokemon AI Dynamic Pricing API",
            "status": "running",
            "endpoints": {
                "GET /health": "Check API status.",
                "POST /metrics": "Read model metrics.",
                "POST /predict": "Predict Pokemon cards or train and predict from custom price history.",
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

        if "history" in payload:
            history = payload["history"]
            if not isinstance(history, list) or not history:
                raise ValueError("history must be a non-empty list of price records.")
            if len(history) > MAX_CUSTOM_HISTORY_ROWS:
                raise ValueError(f"history cannot contain more than {MAX_CUSTOM_HISTORY_ROWS} rows.")

            date_col = payload.get("date_col", "date")
            price_col = payload.get("price_col", "price")
            entity_col = payload.get("entity_col") or None
            item = payload.get("item")

            custom_df = pd.DataFrame(history)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                csv_path = temp_path / "custom_history.csv"
                model_dir = temp_path / "model"
                custom_df.to_csv(csv_path, index=False)

                df = load_pricing_data(csv_path, date_col, price_col)
                result = train_models(df, date_col, price_col, model_dir, entity_col)
                predictions = predict_tomorrow(df, model_dir, item)

            return jsonify(
                {
                    "mode": "custom_history",
                    "best_model": result.model_name,
                    "metrics": result.metrics,
                    "predictions": predictions,
                    "history_rows": len(df),
                }
            ), 200

        df = load_pricing_data(DATA_PATH, DATE_COLUMN, PRICE_COLUMN, CSV_SEPARATOR)
        return jsonify({"mode": "pokemon", "predictions": predict_tomorrow(df, MODEL_DIR, payload.get("item"))}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
