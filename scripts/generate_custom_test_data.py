from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path(__file__).resolve().parent.parent / "data" / "custom_price_history_sample.csv"


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    products = [
        ("Collector Box A", 42.0, 0.08),
        ("Premium Pack B", 18.0, 0.03),
        ("Limited Card C", 75.0, 0.14),
    ]

    rows: list[dict[str, object]] = []
    for product, starting_price, daily_trend in products:
        prices = (
            starting_price
            + np.arange(len(dates)) * daily_trend
            + np.sin(np.arange(len(dates)) / 5) * 1.5
            + rng.normal(0, 0.55, len(dates))
        )
        demand = 150 - prices * 1.2 + rng.normal(0, 5, len(dates))

        for date, price, demand_value in zip(dates, prices, demand):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "product": product,
                    "price": round(float(price), 2),
                    "demand": max(round(float(demand_value)), 1),
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
