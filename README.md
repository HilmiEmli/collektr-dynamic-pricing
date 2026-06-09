# Pokemon AI Dynamic Pricing

This project trains Random Forest and XGBoost models using Pokemon card market-price history. It includes a Flask API and Streamlit dashboard.

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
