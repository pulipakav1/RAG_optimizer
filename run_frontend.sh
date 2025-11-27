#!/bin/bash

# Activate virtual environment and run Streamlit frontend
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run frontend_app.py --server.port 8501

