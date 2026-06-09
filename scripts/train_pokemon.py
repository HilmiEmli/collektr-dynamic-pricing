from src.config import CSV_SEPARATOR, DATA_PATH, DATE_COLUMN, ENTITY_COLUMN, MODEL_DIR, PRICE_COLUMN
from src.dynamic_pricing import load_pricing_data, predict_tomorrow, train_models


def main() -> None:
    df = load_pricing_data(DATA_PATH, DATE_COLUMN, PRICE_COLUMN, CSV_SEPARATOR)
    result = train_models(df, DATE_COLUMN, PRICE_COLUMN, MODEL_DIR, ENTITY_COLUMN)

    print(f"Best model: {result.model_name}")
    for name, scores in result.metrics.items():
        print(f"{name}: MAE={scores['mae']:.3f}, RMSE={scores['rmse']:.3f}, R2={scores['r2']:.3f}")

    for prediction in predict_tomorrow(df, MODEL_DIR):
        print(f"{prediction['item']}: {prediction['predicted_price']:.2f}")


if __name__ == "__main__":
    main()
