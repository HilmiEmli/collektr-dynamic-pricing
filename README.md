# Pokemon AI Dynamic Pricing

This project trains Random Forest and XGBoost models using Pokemon card market-price history. It includes a Flask API and Streamlit dashboard.

The dashboard includes:

- A buyer workspace for market monitoring and AI price predictions
- A seller workspace for minimum-price protection and automatic market-following listing prices

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

Predict all cards by sending `{}`.

## Dataset

The included `data/pkm_card_prices.csv` uses:

- `name`: card name
- `updated_at`: price date
- `market`: prediction target
- `market_low`: additional price feature
