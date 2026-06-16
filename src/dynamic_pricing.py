from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - handled by requirements in normal use
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


MODEL_FILE = "best_model.joblib"
METRICS_FILE = "metrics.json"


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    metrics: dict[str, dict[str, float]]
    feature_columns: list[str]


def load_pricing_data(csv_path: Path, date_col: str, price_col: str, sep: str | None = None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    read_kwargs: dict[str, Any] = {"sep": sep} if sep else {"sep": None, "engine": "python"}
    df = pd.read_csv(csv_path, **read_kwargs)
    missing = {date_col, price_col} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col]).sort_values(date_col).reset_index(drop=True)

    if len(df) < 30:
        raise ValueError("At least 30 historical rows are recommended for time-series training.")

    return df


def build_features(df: pd.DataFrame, date_col: str, price_col: str, entity_col: str | None = None) -> pd.DataFrame:
    featured = df.copy()
    sort_columns = [entity_col, date_col] if entity_col else [date_col]
    featured = featured.sort_values(sort_columns).reset_index(drop=True)
    date_series = featured[date_col]

    featured["day_of_week"] = date_series.dt.dayofweek
    featured["day_of_month"] = date_series.dt.day
    featured["month"] = date_series.dt.month
    featured["is_weekend"] = date_series.dt.dayofweek.isin([5, 6]).astype(int)

    if entity_col:
        categories = sorted(featured[entity_col].dropna().astype(str).unique())
        featured["entity_code"] = pd.Categorical(featured[entity_col].astype(str), categories=categories).codes
        grouped_price = featured.groupby(entity_col, sort=False)[price_col]
        featured["price_lag_1"] = grouped_price.shift(1)
        featured["price_lag_7"] = grouped_price.shift(7)
        featured["price_rolling_mean_7"] = grouped_price.transform(lambda series: series.shift(1).rolling(7).mean())
        featured["price_rolling_std_7"] = grouped_price.transform(lambda series: series.shift(1).rolling(7).std())
        featured["price_change_1"] = grouped_price.diff(1)
        featured["tomorrow_price"] = grouped_price.shift(-1)
    else:
        featured["price_lag_1"] = featured[price_col].shift(1)
        featured["price_lag_7"] = featured[price_col].shift(7)
        featured["price_rolling_mean_7"] = featured[price_col].shift(1).rolling(7).mean()
        featured["price_rolling_std_7"] = featured[price_col].shift(1).rolling(7).std()
        featured["price_change_1"] = featured[price_col].diff(1)
        featured["tomorrow_price"] = featured[price_col].shift(-1)

    excluded_for_lags = {price_col, "tomorrow_price", "entity_code"}
    if entity_col:
        excluded_for_lags.add(entity_col)

    for column in numeric_feature_columns(featured, exclude=excluded_for_lags):
        if column in {"day_of_week", "day_of_month", "month", "is_weekend", "price_lag_1", "price_lag_7"}:
            continue
        if entity_col:
            grouped_column = featured.groupby(entity_col, sort=False)[column]
            featured[f"{column}_lag_1"] = grouped_column.shift(1)
            featured[f"{column}_rolling_mean_7"] = grouped_column.transform(
                lambda series: series.shift(1).rolling(7).mean()
            )
        else:
            featured[f"{column}_lag_1"] = featured[column].shift(1)
            featured[f"{column}_rolling_mean_7"] = featured[column].shift(1).rolling(7).mean()

    return featured


def numeric_feature_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [column for column in numeric_cols if column not in exclude]


def prepare_training_frame(
    df: pd.DataFrame, date_col: str, price_col: str, entity_col: str | None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    featured = build_features(df, date_col, price_col, entity_col)
    model_frame = featured.dropna().reset_index(drop=True)
    feature_columns = numeric_feature_columns(model_frame, exclude={price_col, "tomorrow_price"})

    if model_frame.empty:
        raise ValueError("Not enough complete rows after feature engineering. Add more history.")

    return model_frame[feature_columns], model_frame["tomorrow_price"], feature_columns


def train_models(
    df: pd.DataFrame, date_col: str, price_col: str, model_dir: Path, entity_col: str | None = None
) -> TrainingResult:
    if entity_col and entity_col not in df.columns:
        raise ValueError(f"Entity column not found: {entity_col}")

    x, y, feature_columns = prepare_training_frame(df, date_col, price_col, entity_col)
    split_index = max(int(len(x) * 0.8), 1)

    if split_index >= len(x):
        raise ValueError("Not enough rows to create a validation split.")

    x_train, x_valid = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_valid = y.iloc[:split_index], y.iloc[split_index:]

    models: dict[str, Any] = {
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=1,
        )
    }

    if XGBRegressor is None:
        raise ImportError("xgboost is required. Install dependencies with: pip install -r requirements.txt") from XGBOOST_IMPORT_ERROR

    models["xgboost"] = XGBRegressor(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )

    metrics: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_valid)
        metrics[name] = {
            "mae": float(mean_absolute_error(y_valid, predictions)),
            "rmse": float(mean_squared_error(y_valid, predictions) ** 0.5),
            "r2": float(r2_score(y_valid, predictions)),
        }
        fitted_models[name] = model

    best_model_name = min(metrics, key=lambda name: metrics[name]["mae"])
    artifact = {
        "model": fitted_models[best_model_name],
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "date_col": date_col,
        "price_col": price_col,
        "entity_col": entity_col,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / MODEL_FILE)
    (model_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainingResult(best_model_name, metrics, feature_columns)


def predict_future_prices(
    df: pd.DataFrame, model_dir: Path, item: str | None = None, horizon: int = 7
) -> list[dict[str, Any]]:
    artifact_path = model_dir / MODEL_FILE
    if not artifact_path.exists():
        raise FileNotFoundError(f"No saved model found at {artifact_path}. Run training first.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if horizon > 30:
        raise ValueError("horizon cannot be greater than 30")

    artifact = joblib.load(artifact_path)
    date_col = artifact["date_col"]
    price_col = artifact["price_col"]
    entity_col = artifact.get("entity_col")
    feature_columns = artifact["feature_columns"]

    if item and not entity_col:
        raise ValueError("--item can only be used with a model trained using --entity-col")

    working_df = df.copy().sort_values([entity_col, date_col] if entity_col else [date_col]).reset_index(drop=True)
    if item and working_df[entity_col].astype(str).str.casefold().eq(item.casefold()).sum() == 0:
        raise ValueError(f"No rows found for item: {item}")

    results_by_key: dict[str, dict[str, Any]] = {}
    numeric_columns = working_df.select_dtypes(include=[np.number]).columns.tolist()

    for step in range(1, horizon + 1):
        featured = build_features(working_df, date_col, price_col, entity_col)
        if item:
            prediction_pool = featured[featured[entity_col].astype(str).str.casefold() == item.casefold()]
        else:
            prediction_pool = featured

        if entity_col:
            latest_rows = prediction_pool.sort_values(date_col).groupby(entity_col, sort=False).tail(1)
        else:
            latest_rows = prediction_pool.sort_values(date_col).tail(1)

        latest_features = latest_rows[feature_columns]
        if latest_features.isna().any(axis=None):
            missing_columns = latest_features.columns[latest_features.isna().any()].tolist()
            raise ValueError(f"Latest row has missing engineered features: {', '.join(missing_columns)}")

        predictions = artifact["model"].predict(latest_features)
        new_rows: list[pd.Series] = []
        for (_, row), prediction in zip(latest_rows.iterrows(), predictions):
            latest_date = pd.to_datetime(row[date_col])
            prediction_date = latest_date + pd.Timedelta(days=1)
            key = str(row[entity_col]) if entity_col else "__single_series__"
            predicted_price = round(float(prediction), 2)

            if key not in results_by_key:
                output = {
                    "model_name": artifact["model_name"],
                    "latest_date": latest_date.date().isoformat(),
                    "prediction_date": prediction_date.date().isoformat(),
                    "predicted_price": predicted_price,
                    "forecast": [],
                }
                if entity_col:
                    output["item"] = key
                results_by_key[key] = output

            results_by_key[key]["forecast"].append(
                {
                    "day": step,
                    "prediction_date": prediction_date.date().isoformat(),
                    "predicted_price": predicted_price,
                }
            )

            original_columns = working_df.columns
            new_row = row.reindex(original_columns).copy()
            new_row[date_col] = prediction_date
            new_row[price_col] = predicted_price
            for column in numeric_columns:
                if column != price_col and pd.isna(new_row[column]):
                    new_row[column] = row.get(column)
            new_rows.append(new_row)

        if new_rows:
            working_df = pd.concat([working_df, pd.DataFrame(new_rows)], ignore_index=True)

    return list(results_by_key.values())


def predict_tomorrow(df: pd.DataFrame, model_dir: Path, item: str | None = None) -> list[dict[str, Any]]:
    return predict_future_prices(df, model_dir, item, horizon=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI dynamic pricing models and predict tomorrow's price.")
    parser.add_argument("--data", required=True, type=Path, help="Path to historical pricing CSV.")
    parser.add_argument("--date-col", default="date", help="Date column name.")
    parser.add_argument("--price-col", default="price", help="Price column name.")
    parser.add_argument("--sep", default=None, help="CSV delimiter. By default pandas will auto-detect it.")
    parser.add_argument("--entity-col", default=None, help="Optional item/card/product column for grouped forecasting.")
    parser.add_argument("--item", default=None, help="Predict only one item when --entity-col was used for training.")
    parser.add_argument("--model-dir", default=Path("models"), type=Path, help="Directory for saved model artifacts.")
    parser.add_argument("--train", action="store_true", help="Train XGBoost and Random Forest models.")
    parser.add_argument("--predict", action="store_true", help="Predict tomorrow's price.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train and not args.predict:
        raise SystemExit("Choose at least one action: --train or --predict")

    df = load_pricing_data(args.data, args.date_col, args.price_col, args.sep)

    if args.train:
        result = train_models(df, args.date_col, args.price_col, args.model_dir, args.entity_col)
        print(f"Best model: {result.model_name}")
        for model_name, scores in result.metrics.items():
            print(
                f"{model_name}: MAE={scores['mae']:.3f}, "
                f"RMSE={scores['rmse']:.3f}, R2={scores['r2']:.3f}"
            )

    if args.predict:
        predictions = predict_tomorrow(df, args.model_dir, args.item)
        for prediction in predictions:
            item_prefix = f"{prediction['item']} - " if "item" in prediction else ""
            print(
                f"{item_prefix}predicted price for {prediction['prediction_date']} "
                f"using {prediction['model_name']}: {prediction['predicted_price']:.2f}"
            )


if __name__ == "__main__":
    main()
