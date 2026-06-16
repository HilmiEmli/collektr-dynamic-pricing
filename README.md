# Pokemon AI Dynamic Pricing

This project trains Random Forest and XGBoost models using Pokemon card market-price history. It includes a Flask API and Streamlit dashboard.

Predictions return a 7-day forecast by default.

The dashboard includes:

- A buyer workspace for market monitoring and AI price predictions
- A seller workspace for minimum-price protection and automatic market-following listing prices
- A custom-data workspace where users upload their own price history and receive a prediction

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python -m scripts.train_pokemon
```

The best model is saved to `models/pokemon/best_model.joblib`.

## Run

Start the API:

```powershell
python api.py
```

Start the UI in another terminal:

```powershell
python -m streamlit run app.py
```

Open `http://localhost:8501`.

## Deploy to Google Cloud Run

The included `Dockerfile` trains the Pokemon model during the image build, starts the Flask API internally, and exposes the Streamlit UI on Cloud Run's assigned port.

Deploy from the repository root:

```powershell
gcloud run deploy collektr-dynamic-pricing `
  --source . `
  --region asia-southeast1 `
  --allow-unauthenticated
```

When deploying through the Google Cloud console, select this GitHub repository and keep the Dockerfile path as:

```text
Dockerfile
```

Deploy the Flask API as a separate public Cloud Run service:

```powershell
gcloud.cmd builds submit --config cloudbuild-api.yaml

gcloud.cmd run deploy collektr-pricing-api `
  --image "gcr.io/YOUR_PROJECT_ID/collektr-pricing-api" `
  --region asia-southeast1 `
  --allow-unauthenticated `
  --port 8080
```

## API

```text
GET  /health
POST /metrics
POST /predict
POST /train
```

Predict one card:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"item":"Alakazam"}'
```

Predict all cards by sending `{}`. Add `"horizon": 7` to control the forecast length up to 30 days.

Predict using a JSON array directly:

```python
import json
import requests

with open("my_price_history.json", encoding="utf-8") as file:
    history = json.load(file)

response = requests.post(
    "https://YOUR_API_URL/predict",
    json=history,
)

print(response.json())
```

The API automatically recognizes common field names:

```text
Date: date, updated_at, created_at, timestamp, datetime
Price: price, market, market_price, current_price, value
Item: product, item, name, card, sku
```

Use a wrapped JSON object when field names are different or when predicting one selected item:

```json
{
  "history": [
    {"recorded_on": "2026-01-01", "product_code": "A", "selling_price": 42.10},
    {"recorded_on": "2026-01-02", "product_code": "A", "selling_price": 43.20}
  ],
  "date_col": "recorded_on",
  "price_col": "selling_price",
  "entity_col": "product_code",
  "item": "A"
}
```

Custom history must contain at least 30 records. Training is temporary and does not overwrite the shared Pokemon model.

The Custom Data workspace accepts uploaded JSON arrays, pasted JSON arrays, and CSV files.

## Dataset

The included `data/pkm_card_prices.csv` uses:

- `name`: card name
- `updated_at`: price date
- `market`: prediction target
- `market_low`: additional price feature
