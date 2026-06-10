from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from src.config import API_URL, CSV_SEPARATOR, DATA_PATH


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=CSV_SEPARATOR)
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df["market"] = pd.to_numeric(df["market"], errors="coerce")
    df["market_low"] = pd.to_numeric(df["market_low"], errors="coerce")
    return df.dropna(subset=["name", "updated_at", "market", "market_low"]).sort_values(["name", "updated_at"])


def post_api(endpoint: str, payload: dict | None = None) -> dict:
    response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=120)
    body = response.json()
    if response.status_code >= 400:
        raise RuntimeError(body.get("error", f"API returned HTTP {response.status_code}"))
    return body


def latest_market(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.groupby("name", sort=False).tail(1).copy()
    previous = (
        df.groupby("name", sort=False)
        .tail(2)
        .groupby("name", sort=False)
        .head(1)[["name", "market"]]
        .rename(columns={"market": "previous_market"})
    )
    latest = latest.merge(previous, on="name", how="left")
    latest["change"] = latest["market"] - latest["previous_market"]
    latest["change_pct"] = latest["change"] / latest["previous_market"] * 100
    latest["spread"] = latest["market"] - latest["market_low"]
    return latest.sort_values("market", ascending=False).reset_index(drop=True)


def money(value: float) -> str:
    return f"${value:,.2f}"


def main() -> None:
    st.set_page_config(page_title="Pokemon Card Market", page_icon=".", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.25);}
        [data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 6px;
            padding: .8rem 1rem;
        }
        div[data-testid="stMetricLabel"] {font-size: .8rem;}
        h1, h2, h3 {letter-spacing: 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    market = latest_market(df)
    cards = sorted(df["name"].unique())

    with st.sidebar:
        st.title("Card Market")
        st.caption("Daily Pokemon card price monitor")
        selected_card = st.selectbox("Card", cards, index=cards.index("Alakazam") if "Alakazam" in cards else 0)
        range_days = st.select_slider("History", options=[7, 14, 30, 60, 90], value=30, format_func=lambda x: f"{x} days")
        st.divider()
        st.metric("Cards tracked", len(cards))
        st.metric("Latest update", df["updated_at"].max().date().isoformat())

    card_history = df[df["name"] == selected_card].sort_values("updated_at").tail(range_days)
    current = market[market["name"] == selected_card].iloc[0]
    prediction: dict | None = None
    prediction_error: str | None = None
    try:
        prediction = post_api("/predict", {"item": selected_card})["predictions"][0]
    except Exception as exc:
        prediction_error = str(exc)

    title_col, status_col = st.columns([4, 1])
    with title_col:
        st.title(selected_card)
        st.caption(f"Card ID {int(current['id'])} | Updated {current['updated_at']:%B %d, %Y}")
    with status_col:
        direction = "Up" if current["change"] > 0 else "Down" if current["change"] < 0 else "Flat"
        st.metric("Daily movement", direction, f"{current['change_pct']:+.2f}%")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Current market", money(current["market"]), money(current["change"]))
    metric_cols[1].metric("Market low", money(current["market_low"]))
    metric_cols[2].metric("Price spread", money(current["spread"]))
    metric_cols[3].metric("Period average", money(card_history["market"].mean()))
    if prediction:
        prediction_change = prediction["predicted_price"] - current["market"]
        metric_cols[4].metric("Predicted next price", money(prediction["predicted_price"]), money(prediction_change))
    else:
        metric_cols[4].metric("Predicted next price", "Unavailable")

    overview_tab, prediction_tab, history_tab, market_tab, metrics_tab = st.tabs(
        ["Overview", "Prediction", "Price History", "All Cards", "Model Metrics"]
    )

    with overview_tab:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Market vs market low")
            chart_df = card_history.set_index("updated_at")[["market", "market_low"]].rename(
                columns={"market": "Market price", "market_low": "Market low"}
            )
            st.line_chart(chart_df, use_container_width=True, height=340)
        with right:
            st.subheader("Price range")
            range_rows = pd.DataFrame(
                {
                    "Measure": ["Latest market", "Latest low", "Period high", "Period low", "Period average"],
                    "Price": [
                        current["market"],
                        current["market_low"],
                        card_history["market"].max(),
                        card_history["market"].min(),
                        card_history["market"].mean(),
                    ],
                }
            )
            st.dataframe(
                range_rows,
                use_container_width=True,
                hide_index=True,
                column_config={"Price": st.column_config.NumberColumn("Price", format="$%.2f")},
            )

    with prediction_tab:
        if prediction:
            prediction_change = prediction["predicted_price"] - current["market"]
            prediction_pct = prediction_change / current["market"] * 100
            prediction_cols = st.columns(4)
            prediction_cols[0].metric("Current market", money(current["market"]))
            prediction_cols[1].metric("Predicted price", money(prediction["predicted_price"]))
            prediction_cols[2].metric("Expected change", money(prediction_change), f"{prediction_pct:+.2f}%")
            prediction_cols[3].metric("Prediction date", prediction["prediction_date"])
            st.caption(f"Prediction generated using {prediction['model_name'].replace('_', ' ').title()}.")
        else:
            st.error(f"Prediction is unavailable: {prediction_error}")

    with history_tab:
        history_view = card_history[["updated_at", "market", "market_low"]].sort_values("updated_at", ascending=False)
        st.dataframe(
            history_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "updated_at": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                "market": st.column_config.NumberColumn("Market", format="$%.2f"),
                "market_low": st.column_config.NumberColumn("Market Low", format="$%.2f"),
            },
        )

    with market_tab:
        table = market[["name", "market", "market_low", "change", "change_pct", "updated_at"]].copy()
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "name": "Card",
                "market": st.column_config.NumberColumn("Current Market", format="$%.2f"),
                "market_low": st.column_config.NumberColumn("Market Low", format="$%.2f"),
                "change": st.column_config.NumberColumn("Daily Change", format="$%.2f"),
                "change_pct": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                "updated_at": st.column_config.DatetimeColumn("Updated", format="YYYY-MM-DD"),
            },
        )

    with metrics_tab:
        try:
            metrics = post_api("/metrics")["metrics"]
            metric_df = pd.DataFrame([{"model": name, **scores} for name, scores in metrics.items()])
            st.dataframe(metric_df.sort_values("mae"), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
