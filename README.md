# Candidate Search — Interview-Aware RAG Application

AI-powered recruitment application that searches and ranks candidates using both profile/resume data and interview evidence.

## Features

- **Natural Language Search**: Describe the candidate you need in plain English
- **Interview-Aware Ranking**: Combines profile relevance + interview evidence + structured matching
- **Document Upload**: Support for PDF, DOCX, TXT, MD, SRT, VTT files
- **AI-Powered Analysis**: Query parsing and candidate summaries via Groq LLM
- **Vector Search**: Semantic search using Qdrant + local embeddings
- **Clean UI**: Simple, recruiter-friendly interface

## Architecture

```
User Query → LLM Query Parser → Structured Requirements
                                    ↓
                    Profile Vector Retrieval (Qdrant)
                    Interview Vector Retrieval (Qdrant)
                                    ↓
                    Candidate Aggregation & Ranking
                                    ↓
                    LLM Summary Generation
                                    ↓
                              UI Display
```

## Project Structure

```
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── api/
│   │   ├── search.py           # Search endpoints
│   │   ├── upload.py           # Upload endpoints
│   │   └── candidates.py       # Candidate management
│   ├── core/
│   │   ├── config.py           # Configuration
│   │   └── logging.py          # Logging setup
│   ├── ingestion/
│   │   ├── loaders.py          # Document loaders
│   │   ├── parser.py           # Text parsing
│   │   ├── chunker.py          # Text chunking
│   │   └── pipeline.py         # Ingestion pipeline
│   ├── embeddings/
│   │   └── service.py          # Embedding service
│   ├── qdrant/
│   │   ├── client.py           # Qdrant client
│   │   ├── collections.py      # Collection management
│   │   └── search.py           # Search operations
│   ├── retrieval/
│   │   ├── profile_retriever.py
│   │   ├── interview_retriever.py
│   │   └── candidate_ranker.py
│   ├── llm/
│   │   ├── groq_client.py      # Groq API client
│   │   ├── query_parser.py     # NL query parser
│   │   └── summarizer.py       # Summary generator
│   ├── schemas/
│   │   ├── candidate.py
│   │   ├── interview.py
│   │   ├── search.py
│   │   └── responses.py
│   ├── templates/
│   │   └── index.html          # Main UI
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── data/
│   ├── candidates/             # Sample candidate profiles
│   └── interviews/             # Sample interview transcripts
├── tests/                      # Unit tests
├── scripts/
│   ├── init_qdrant.py
│   └── ingest.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── run.sh
└── run.bat
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+ (for local development)
- Groq API key

### Setup

1. **Clone and configure:**
   ```bash
   git clone <repo>
   cd candidate-rag
   cp .env.example .env
   ```

2. **Add your Groq API key to `.env`:**
   ```
   GROQ_API_KEY=your_key_here
   ```

3. **Start the application:**

   **Linux/macOS/Lightning AI:**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

   **Windows:**
   ```batch
   run.bat
   ```

   **Or manually:**
   ```bash
   docker compose up --build
   ```

4. **Open the UI:**
   ```
   http://localhost:8000
   ```

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant (requires Docker)
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Start the application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main UI |
| GET | `/health` | Health check |
| POST | `/api/search` | Search candidates |
| POST | `/api/candidates/upload` | Upload profile |
| POST | `/api/interviews/upload` | Upload interview |
| GET | `/api/candidates` | List candidates |
| GET | `/api/candidates/{id}` | Get candidate detail |
| GET | `/api/candidates/{id}/evidence` | Get candidate evidence |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant URL |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `PROFILE_WEIGHT` | `0.45` | Profile score weight |
| `INTERVIEW_WEIGHT` | `0.35` | Interview score weight |
| `STRUCTURED_WEIGHT` | `0.20` | Structured match weight |
| `CHUNK_SIZE` | `700` | Chunk size in tokens |
| `CHUNK_OVERLAP` | `100` | Chunk overlap in tokens |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_search.py -v
```

## Demo Data

Sample candidates and interviews are in `data/candidates/` and `data/interviews/`.

To ingest demo data:
```bash
python scripts/ingest.py
```

## Example Search

**Query:**
> Find a Python developer with around 5 years of experience who has strong FastAPI and software development skills, has worked with RAG and embeddings, and should have demonstrated FastAPI knowledge during the interview.

**Expected behavior:**
- Candidates with FastAPI + RAG in profile rank higher
- Candidates with strong interview evidence rank higher than those without
- Results include profile evidence, interview evidence, and AI summary

## Qdrant Collections

| Collection | Purpose |
|------------|---------|
| `candidate_profiles` | Candidate profile chunks |
| `interview_chunks` | Interview transcript chunks |

Access Qdrant dashboard at: `http://localhost:6333/dashboard`

## Troubleshooting

**Qdrant connection failed:**
- Ensure Qdrant container is running: `docker ps`
- Check QDRANT_URL in `.env`

**Embedding model download fails:**
- Check internet connection
- Model will be cached after first download

**Groq API errors:**
- Verify GROQ_API_KEY is set correctly
- Check Groq API status

## License

MIT
# CANDIDATE-rag-tESTING
