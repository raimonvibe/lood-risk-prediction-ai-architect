#!/bin/bash
uvicorn api:app --host 0.0.0.0 --port 9000 &
streamlit run frontend.py --server.port 8501
