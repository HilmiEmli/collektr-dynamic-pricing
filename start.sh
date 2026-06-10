#!/bin/sh
set -e

gunicorn --bind 127.0.0.1:8000 --workers 1 --threads 4 api:app &

exec python -m streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8080}" \
  --server.headless=true
