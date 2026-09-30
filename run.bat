@echo off
echo Starting Candidate Search RAG Application...

if not exist .env (
    echo Warning: .env file not found. Using default configuration.
    echo Copy .env.example to .env and configure your settings.
)

docker compose up --build
pause
