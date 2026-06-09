from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "pkm_card_prices.csv"
MODEL_DIR = BASE_DIR / "models" / "pokemon"
API_URL = os.getenv("PRICING_API_URL", "http://127.0.0.1:8000")

DATE_COLUMN = "updated_at"
PRICE_COLUMN = "market"
ENTITY_COLUMN = "name"
CSV_SEPARATOR = ";"
