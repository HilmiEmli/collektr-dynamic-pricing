from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from src.config import API_URL, CSV_SEPARATOR, DATA_PATH, DATE_COLUMN, ENTITY_COLUMN, PRICE_COLUMN
from src.dynamic_pricing import load_pricing_data


def post_api(endpoint: str, payload: dict | None = None) -> dict:
    response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=120)
    body = response.json()
    if response.status_code >= 400:
        raise RuntimeError(body.get("error", f"API returned HTTP {response.status_code}"))
    return body


def main() -> None:
    st.set_page_config(page_title="Pokemon AI Pricing", page_icon=".", layout="wide")
    st.title("Pokemon AI Dynamic Pricing")

    df = load_pricing_data(DATA_PATH, DATE_COLUMN, PRICE_COLUMN, CSV_SEPARATOR)
    cards = sorted(df[ENTITY_COLUMN].dropna().astype(str).unique())

    with st.sidebar:
        st.header("Price Prediction")
        selected = st.selectbox("Card", ["All cards", *cards])
        predict_clicked = st.button("Predict price", use_container_width=True, type="primary")
        train_clicked = st.button("Retrain models", use_container_width=True)
        st.caption(f"API: {API_URL}")

    if train_clicked:
        with st.spinner("Training Random Forest and XGBoost..."):
            result = post_api("/train")
        st.success(f"Best model: {result['best_model']}")

    latest_date = pd.to_datetime(df[DATE_COLUMN]).max()
    cols = st.columns(3)
    cols[0].metric("Rows", f"{len(df):,}")
    cols[1].metric("Cards", f"{len(cards):,}")
    cols[2].metric("Latest data", latest_date.date().isoformat())

    predictions_tab, metrics_tab, history_tab = st.tabs(["Predictions", "Model Metrics", "Price History"])

    with predictions_tab:
        if predict_clicked:
            item = None if selected == "All cards" else selected
            try:
                predictions = post_api("/predict", {"item": item})["predictions"]
                prediction_df = pd.DataFrame(predictions)
                st.dataframe(
                    prediction_df.sort_values("predicted_price", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as exc:
                st.error(str(exc))
        else:
            st.info("Select a card and click Predict price.")

    with metrics_tab:
        try:
            metrics = post_api("/metrics")["metrics"]
            metric_df = pd.DataFrame([{"model": name, **scores} for name, scores in metrics.items()])
            st.dataframe(metric_df.sort_values("mae"), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))

    with history_tab:
        history = df if selected == "All cards" else df[df[ENTITY_COLUMN] == selected]
        if selected == "All cards":
            history = history.groupby(DATE_COLUMN, as_index=False)[PRICE_COLUMN].mean()
        st.line_chart(history.sort_values(DATE_COLUMN), x=DATE_COLUMN, y=PRICE_COLUMN, use_container_width=True)


if __name__ == "__main__":
    main()
