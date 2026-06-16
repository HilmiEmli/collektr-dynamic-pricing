from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

from src.config import CSV_SEPARATOR, DATA_PATH, DATE_COLUMN, ENTITY_COLUMN, MODEL_DIR, PRICE_COLUMN
from src.dynamic_pricing import load_pricing_data, predict_tomorrow, train_models


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
MAX_CUSTOM_HISTORY_ROWS = 5000
DATE_COLUMN_CANDIDATES = ("date", "updated_at", "created_at", "timestamp", "datetime")
PRICE_COLUMN_CANDIDATES = ("price", "market", "market_price", "current_price", "value")
ENTITY_COLUMN_CANDIDATES = ("product", "item", "name", "card", "sku")


def infer_column(columns: list[str], candidates: tuple[str, ...], label: str) -> str:
    normalized = {column.casefold(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"Could not infer the {label} column. Use the wrapped JSON format and provide {label}_col explicitly."
    )


def parse_custom_payload(payload: Any) -> tuple[list[dict[str, Any]], str, str, str | None, str | None]:
    if isinstance(payload, list):
        history = payload
        options: dict[str, Any] = {}
    elif isinstance(payload, dict) and "history" in payload:
        history = payload["history"]
        options = payload
    else:
        raise ValueError("Custom input must be a JSON array or an object containing a history array.")

    if not isinstance(history, list) or not history:
        raise ValueError("history must be a non-empty JSON array of price records.")
    if len(history) > MAX_CUSTOM_HISTORY_ROWS:
        raise ValueError(f"history cannot contain more than {MAX_CUSTOM_HISTORY_ROWS} rows.")
    if not all(isinstance(record, dict) for record in history):
        raise ValueError("Every history array entry must be a JSON object.")

    columns = list(pd.DataFrame(history).columns)
    if not columns:
        raise ValueError("History records must contain fields.")

    date_col = options.get("date_col") or infer_column(columns, DATE_COLUMN_CANDIDATES, "date")
    price_col = options.get("price_col") or infer_column(columns, PRICE_COLUMN_CANDIDATES, "price")
    normalized_columns = {column.casefold(): column for column in columns}
    entity_col = options.get("entity_col") or next(
        (normalized_columns[candidate] for candidate in ENTITY_COLUMN_CANDIDATES if candidate in normalized_columns),
        None,
    )
    item = options.get("item")

    missing = {date_col, price_col} - set(columns)
    if missing:
        raise ValueError(f"History records are missing required fields: {', '.join(sorted(missing))}")
    if entity_col and entity_col not in columns:
        raise ValueError(f"History records are missing entity_col: {entity_col}")

    return history, date_col, price_col, entity_col, item


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
        started_at = time.perf_counter()
        df = load_pricing_data(DATA_PATH, DATE_COLUMN, PRICE_COLUMN, CSV_SEPARATOR)
        result = train_models(df, DATE_COLUMN, PRICE_COLUMN, MODEL_DIR, ENTITY_COLUMN)
        return jsonify(
            {
                "best_model": result.model_name,
                "metrics": result.metrics,
                "training_seconds": round(time.perf_counter() - started_at, 3),
            }
        ), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/predict")
def predict() -> tuple[Any, int]:
    try:
        request_started_at = time.perf_counter()
        payload = request.get_json(silent=True)
        if payload is None:
            payload = {}

        if isinstance(payload, list) or (isinstance(payload, dict) and "history" in payload):
            history, date_col, price_col, entity_col, item = parse_custom_payload(payload)
            custom_df = pd.DataFrame(history)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                csv_path = temp_path / "custom_history.csv"
                model_dir = temp_path / "model"
                custom_df.to_csv(csv_path, index=False)

                df = load_pricing_data(csv_path, date_col, price_col)
                training_started_at = time.perf_counter()
                result = train_models(df, date_col, price_col, model_dir, entity_col)
                training_seconds = time.perf_counter() - training_started_at
                prediction_started_at = time.perf_counter()
                predictions = predict_tomorrow(df, model_dir, item)
                prediction_seconds = time.perf_counter() - prediction_started_at

            return jsonify(
                {
                    "mode": "custom_history",
                    "best_model": result.model_name,
                    "metrics": result.metrics,
                    "predictions": predictions,
                    "history_rows": len(df),
                    "training_seconds": round(training_seconds, 3),
                    "prediction_seconds": round(prediction_seconds, 3),
                    "total_seconds": round(time.perf_counter() - request_started_at, 3),
                }
            ), 200

        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object or a JSON history array.")

        df = load_pricing_data(DATA_PATH, DATE_COLUMN, PRICE_COLUMN, CSV_SEPARATOR)
        prediction_started_at = time.perf_counter()
        predictions = predict_tomorrow(df, MODEL_DIR, payload.get("item"))
        prediction_seconds = time.perf_counter() - prediction_started_at
        return jsonify(
            {
                "mode": "pokemon",
                "predictions": predictions,
                "prediction_seconds": round(prediction_seconds, 3),
                "total_seconds": round(time.perf_counter() - request_started_at, 3),
            }
        ), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
