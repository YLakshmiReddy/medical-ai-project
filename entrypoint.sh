#!/bin/bash

# D:\medical_ai_project\entrypoint.sh

echo "Starting FastAPI application..."

# This is where the LLM will download on the first run in the Space
# It's important that this runs AFTER requirements are installed and BEFORE uvicorn starts.

# Running uvicorn to serve the FastAPI app
# --host 0.0.0.0 is crucial for Docker containers to be accessible externally
# --port 8000 matches the EXPOSE port in the Dockerfile
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

echo "FastAPI application stopped."