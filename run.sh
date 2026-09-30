#!/bin/bash
set -e

echo "Starting Candidate Search RAG Application..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Using default configuration."
    echo "Copy .env.example to .env and configure your settings."
fi

# Start with Docker Compose
docker compose up --build
