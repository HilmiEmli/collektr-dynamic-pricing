#!/bin/sh
set -e

gunicorn --bind 127.0.0.1:8000 --workers 1 --threads 4 api:app &

exec gunicorn --bind 0.0.0.0:"${PORT:-8080}" --workers 1 --threads 4 mobile_app:app
