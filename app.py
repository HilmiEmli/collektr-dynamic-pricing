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


def default_listings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"card": "Alakazam", "quantity": 2, "purchase_cost": 24.00, "minimum_price": 30.00, "auto_pricing": True},
            {"card": "Blastoise", "quantity": 1, "purchase_cost": 62.00, "minimum_price": 90.00, "auto_pricing": True},
            {"card": "Raichu", "quantity": 3, "purchase_cost": 21.00, "minimum_price": 28.00, "auto_pricing": True},
        ]
    )



def calculate_listings(listings: pd.DataFrame, market: pd.DataFrame, predictions: dict[str, float]) -> pd.DataFrame:
    calculated = listings.copy()
    market_by_name = market.set_index("name")
    calculated["current_market"] = calculated["card"].map(market_by_name["market"])
    calculated["market_low"] = calculated["card"].map(market_by_name["market_low"])
    calculated["predicted_price"] = calculated["card"].map(predictions)
    calculated["listing_price"] = calculated[["current_market", "minimum_price"]].max(axis=1)
    calculated.loc[~calculated["auto_pricing"], "listing_price"] = calculated.loc[
        ~calculated["auto_pricing"], "minimum_price"
    ]
    calculated["profit_each"] = calculated["listing_price"] - calculated["purchase_cost"]
    calculated["estimated_profit"] = calculated["profit_each"] * calculated["quantity"]
    calculated["status"] = calculated.apply(
        lambda row: "Auto pricing off"
        if not row["auto_pricing"]
        else "Minimum active"
        if row["minimum_price"] > row["current_market"]
        else "Following market",
        axis=1,
    )
    return calculated


def render_buyer(df: pd.DataFrame) -> None:
    market = latest_market(df)
    cards = sorted(df["name"].unique())

    with st.sidebar:
        st.subheader("Buyer Market")
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
    prediction_seconds: float | None = None
    try:
        prediction_response = post_api("/predict", {"item": selected_card})
        prediction = prediction_response["predictions"][0]
        prediction_seconds = prediction_response.get("prediction_seconds")
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
            if prediction_seconds is not None:
                st.caption(f"Prediction completed in {prediction_seconds:.3f} seconds.")
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


def render_seller(df: pd.DataFrame) -> None:
    market = latest_market(df)
    cards = sorted(df["name"].unique())

    if "seller_listings" not in st.session_state:
        st.session_state.seller_listings = default_listings()

    try:
        prediction_rows = post_api("/predict", {})["predictions"]
        predictions = {row["item"]: row["predicted_price"] for row in prediction_rows}
    except Exception:
        predictions = {}

    listings = calculate_listings(st.session_state.seller_listings, market, predictions)

    with st.sidebar:
        st.subheader("Seller Pricing")
        st.caption("Automatic market-following prices with seller protection")
        selected_listing = st.selectbox("View listing", listings["card"].tolist())
        st.divider()
        st.caption("Listing price rule")
        st.write("Uses the higher of the market price or your minimum price.")
        st.caption("Demo listings are stored only for this browser session.")

    active_value = (listings["listing_price"] * listings["quantity"]).sum()
    total_profit = listings["estimated_profit"].sum()
    floors_active = int((listings["status"] == "Minimum active").sum())

    st.title("Seller Pricing Console")
    summary_top = st.columns(2)
    summary_bottom = st.columns(2)
    summary_top[0].metric("Active listings", len(listings))
    summary_top[1].metric("Listing value", money(active_value))
    summary_bottom[0].metric("Estimated profit", money(total_profit))
    summary_bottom[1].metric("Minimum floors active", floors_active)

    listings_tab, add_tab, detail_tab, alerts_tab = st.tabs(["Listings", "Add Listing", "Listing Detail", "Alerts"])

    with listings_tab:
        display = listings[
            [
                "card",
                "quantity",
                "current_market",
                "minimum_price",
                "listing_price",
                "predicted_price",
                "estimated_profit",
                "status",
            ]
        ]
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "card": "Card",
                "quantity": "Quantity",
                "current_market": st.column_config.NumberColumn("Current Market", format="$%.2f"),
                "minimum_price": st.column_config.NumberColumn("Minimum Price", format="$%.2f"),
                "listing_price": st.column_config.NumberColumn("Automatic Listing Price", format="$%.2f"),
                "predicted_price": st.column_config.NumberColumn("Predicted Price", format="$%.2f"),
                "estimated_profit": st.column_config.NumberColumn("Estimated Profit", format="$%.2f"),
                "status": "Status",
            },
        )

    with add_tab:
        with st.form("add_listing"):
            form_cols = st.columns(2)
            card = form_cols[0].selectbox("Card", cards)
            quantity = form_cols[1].number_input("Quantity", min_value=1, value=1)
            purchase_cost = form_cols[0].number_input("Purchase cost", min_value=0.0, value=20.0, step=1.0)
            minimum_price = form_cols[1].number_input("Minimum acceptable price", min_value=0.0, value=25.0, step=1.0)
            auto_pricing = st.toggle("Enable automatic market pricing", value=True)
            submitted = st.form_submit_button("Add listing", type="primary")

        if submitted:
            new_listing = pd.DataFrame(
                [
                    {
                        "card": card,
                        "quantity": int(quantity),
                        "purchase_cost": float(purchase_cost),
                        "minimum_price": float(minimum_price),
                        "auto_pricing": auto_pricing,
                    }
                ]
            )
            st.session_state.seller_listings = pd.concat(
                [st.session_state.seller_listings, new_listing], ignore_index=True
            )
            st.rerun()

    with detail_tab:
        listing = listings[listings["card"] == selected_listing].iloc[0]
        detail_top = st.columns(3)
        detail_bottom = st.columns(2)
        detail_top[0].metric("Current market", money(listing["current_market"]))
        detail_top[1].metric("Minimum price", money(listing["minimum_price"]))
        detail_top[2].metric("Listing price", money(listing["listing_price"]))
        detail_bottom[0].metric("Predicted price", money(listing["predicted_price"]))
        detail_bottom[1].metric("Profit each", money(listing["profit_each"]))

        history = df[df["name"] == selected_listing].tail(30).set_index("updated_at")[["market", "market_low"]]
        st.line_chart(history.rename(columns={"market": "Market", "market_low": "Market low"}), use_container_width=True)
        st.caption(f"Status: {listing['status']}")

    with alerts_tab:
        alerts = listings[listings["status"] == "Minimum active"]
        if alerts.empty:
            st.success("All listings are following the market price.")
        else:
            for _, listing in alerts.iterrows():
                st.warning(
                    f"{listing['card']}: market is {money(listing['current_market'])}, "
                    f"so the minimum price of {money(listing['minimum_price'])} is protecting this listing."
                )


def render_custom_data() -> None:
    with st.sidebar:
        st.subheader("Custom Price History")
        st.caption("Upload your own CSV to train a temporary model and predict the next price.")

    st.title("Custom Price Prediction")
    uploaded = st.file_uploader("Upload price-history CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with at least 30 historical rows. The file must include a date and price column.")
        return

    try:
        custom_df = pd.read_csv(uploaded, sep=None, engine="python")
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    columns = custom_df.columns.tolist()
    if len(columns) < 2:
        st.error("The CSV needs at least a date column and price column.")
        return

    controls = st.columns(3)
    date_default = columns.index("date") if "date" in columns else 0
    price_default = columns.index("price") if "price" in columns else min(1, len(columns) - 1)
    date_col = controls[0].selectbox("Date column", columns, index=date_default)
    price_col = controls[1].selectbox("Price column", columns, index=price_default)
    entity_options = ["None", *columns]
    entity_col_choice = controls[2].selectbox("Item column", entity_options)
    entity_col = None if entity_col_choice == "None" else entity_col_choice

    item: str | None = None
    if entity_col:
        items = sorted(custom_df[entity_col].dropna().astype(str).unique())
        item = st.selectbox("Item to predict", items)

    custom_request_key = (
        uploaded.name,
        len(custom_df),
        tuple(columns),
        date_col,
        price_col,
        entity_col,
        item,
    )
    if st.session_state.get("custom_request_key") != custom_request_key:
        st.session_state.custom_request_key = custom_request_key
        st.session_state.pop("custom_prediction_result", None)

    try:
        chart_df = custom_df.copy()
        chart_df[date_col] = pd.to_datetime(chart_df[date_col])
        chart_df[price_col] = pd.to_numeric(chart_df[price_col], errors="coerce")
        if entity_col and item:
            chart_df = chart_df[chart_df[entity_col].astype(str) == item]
        chart_df = chart_df.dropna(subset=[date_col, price_col]).sort_values(date_col)
    except Exception as exc:
        st.error(f"Could not prepare the selected columns: {exc}")
        return

    summary_cols = st.columns(3)
    summary_cols[0].metric("History rows", len(chart_df))
    summary_cols[1].metric("Latest price", money(chart_df[price_col].iloc[-1]) if not chart_df.empty else "Unavailable")
    summary_cols[2].metric(
        "Latest date",
        chart_df[date_col].max().date().isoformat() if not chart_df.empty else "Unavailable",
    )

    st.subheader("Price history")
    st.line_chart(chart_df, x=date_col, y=price_col, use_container_width=True)

    if st.button("Train and predict custom data", type="primary"):
        with st.spinner("Training Random Forest and XGBoost on the uploaded history..."):
            try:
                payload = {
                    "history": custom_df.where(pd.notna(custom_df), None).to_dict(orient="records"),
                    "date_col": date_col,
                    "price_col": price_col,
                    "entity_col": entity_col,
                    "item": item,
                }
                result = post_api("/predict", payload)
                if result.get("mode") != "custom_history":
                    raise RuntimeError(
                        "The API returned the standard Pokemon prediction response. "
                        "Restart or redeploy the API so it includes custom-history prediction support."
                    )
                st.session_state.custom_prediction_result = result
            except Exception as exc:
                st.error(str(exc))

    result = st.session_state.get("custom_prediction_result")
    if result:
        predictions = result.get("predictions", [])
        if not predictions:
            st.error("The API response did not include a prediction.")
            return

        prediction = predictions[0]
        latest_price = float(chart_df[price_col].iloc[-1])
        predicted_price = float(prediction["predicted_price"])
        model_name = result.get("best_model") or prediction.get("model_name") or "unknown"
        result_cols = st.columns(4)
        result_cols[0].metric("Latest price", money(latest_price))
        result_cols[1].metric("Predicted next price", money(predicted_price), money(predicted_price - latest_price))
        result_cols[2].metric("Prediction date", prediction["prediction_date"])
        result_cols[3].metric("Best model", model_name.replace("_", " ").title())

        timing_cols = st.columns(3)
        timing_cols[0].metric("Training time", f"{result.get('training_seconds', 0):.3f} sec")
        timing_cols[1].metric("Prediction time", f"{result.get('prediction_seconds', 0):.3f} sec")
        timing_cols[2].metric("Total request time", f"{result.get('total_seconds', 0):.3f} sec")

        metrics = result.get("metrics", {})
        if metrics:
            metric_df = pd.DataFrame([{"model": name, **scores} for name, scores in metrics.items()])
            st.subheader("Temporary model metrics")
            st.dataframe(metric_df.sort_values("mae"), use_container_width=True, hide_index=True)


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

    with st.sidebar:
        st.title("Pokemon Market")
        workspace = st.radio("Workspace", ["Buyer", "Seller", "Custom Data"])

    df = load_data()
    if workspace == "Seller":
        render_seller(df)
    elif workspace == "Custom Data":
        render_custom_data()
    else:
        render_buyer(df)


if __name__ == "__main__":
    main()
