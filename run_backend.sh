#!/bin/bash

# Activate virtual environment and run FastAPI backend
cd "$(dirname "$0")"
source .venv/bin/activate
uvicorn backend_main:app --reload --host 0.0.0.0 --port 8000

