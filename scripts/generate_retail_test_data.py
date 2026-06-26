from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_OUTPUT = OUTPUT_DIR / "retail_price_history_sample.csv"
JSON_OUTPUT = OUTPUT_DIR / "retail_price_history_sample.json"


def main() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-02-01", periods=75, freq="D")
    items = [
        ("Wireless Charger", "ELEC-1001", 24.0, 0.035, 220),
        ("Gaming Mouse", "ELEC-1002", 39.0, -0.015, 145),
        ("Desk Lamp", "HOME-2001", 31.0, 0.02, 175),
        ("Travel Tumbler", "HOME-2002", 18.0, 0.01, 260),
    ]

    rows: list[dict[str, object]] = []
    for item_name, sku, base_price, trend, base_inventory in items:
        seasonal = np.sin(np.arange(len(dates)) / 6) * 1.15
        noise = rng.normal(0, 0.45, len(dates))
        price = base_price + np.arange(len(dates)) * trend + seasonal + noise
        units_sold = np.maximum(5, 90 - price * 1.4 + rng.normal(0, 6, len(dates)))
        inventory = np.maximum(0, base_inventory - np.cumsum(units_sold * 0.12)).round()

        for date, price_value, sold_value, inventory_value in zip(dates, price, units_sold, inventory):
            rows.append(
                {
                    "timestamp": date.date().isoformat(),
                    "sku": sku,
                    "item": item_name,
                    "current_price": round(float(price_value), 2),
                    "units_sold": int(round(float(sold_value))),
                    "inventory": int(inventory_value),
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUTPUT, index=False)
    df.to_json(JSON_OUTPUT, orient="records", indent=2)
    print(f"Wrote {len(df)} rows to {CSV_OUTPUT}")
    print(f"Wrote {len(df)} records to {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
