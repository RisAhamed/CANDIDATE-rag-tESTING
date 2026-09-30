You are a senior AI engineer, Python backend engineer, RAG engineer, and software architect.

I already have a **basic RAG application** in this repository, but I do not fully understand how the current implementation works. Your task is to **inspect the existing repository first, understand what is already implemented, reuse useful components where appropriate, and then develop the project into a complete working application**.

Do NOT blindly delete or rewrite everything.

First inspect the entire project structure and existing source code. Identify:

- current RAG implementation
- embedding implementation
- vector database implementation
- document ingestion
- chunking
- retrieval/search
- LLM integration
- API/backend
- UI/frontend
- configuration
- Docker configuration
- `.env` usage
- existing requirements
- existing tests
- unused/dead code
- incomplete code
- duplicated code
- incorrect assumptions

Then produce a short internal implementation assessment and proceed with the development.

The final result must be a **clean, modular, runnable Candidate Search + Interview-Aware RAG application**.

# 1. PRIMARY OBJECTIVE

Build an AI-powered recruitment/candidate search application.

The HR/recruiter should be able to:

1. Upload candidate profiles/resumes.
2. Upload candidate interview reports, interview captions/transcripts, or interview notes.
3. Process and chunk those documents.
4. Generate embeddings.
5. Store the embeddings and metadata in Qdrant.
6. Search candidates using natural-language requirements.
7. Consider BOTH:
   - candidate profile/resume information
   - interview information/captions/transcripts
8. Retrieve and rank the most relevant candidates.
9. Explain why each candidate is relevant.
10. Provide evidence from the profile and/or interview.
11. Generate a concise AI summary for the recruiter.
12. Display everything through a clean, simple UI.

Example HR query:

"I need a Python developer with around 5 years of experience who has strong FastAPI and software development experience. During the interview, the candidate should also have demonstrated practical FastAPI knowledge and experience with RAG, embeddings, and LLM applications."

The application should NOT only search resume/profile embeddings.

It MUST also search interview content.

For example:

Candidate A:

Profile:
- Python
- FastAPI
- RAG
- 5 years experience

Interview:
- Talks extensively about FastAPI
- Demonstrates API design knowledge
- Explains RAG architecture
- Has worked with embeddings

Candidate B:

Profile:
- Python
- FastAPI
- 5 years experience

Interview:
- No evidence of FastAPI experience
- Cannot explain RAG

Candidate A should receive stronger relevance because the interview provides supporting evidence.

The system must therefore combine **profile relevance + interview relevance + structured requirements**.

# 2. IMPORTANT DESIGN PRINCIPLE

This is NOT a generic document chatbot.

This is a:

**Candidate Retrieval and Ranking System using RAG + vector search + structured filtering + interview evidence.**

The final architecture should be:

User Query
    ↓
LLM Query Understanding
    ↓
Structured Candidate Requirements
    +
Semantic Query
    ↓
Profile Vector Retrieval
    +
Interview Vector Retrieval
    ↓
Candidate-level aggregation/ranking
    ↓
Top candidates
    ↓
Evidence collection
    ↓
LLM summary/explanation
    ↓
UI

# 3. TECHNOLOGY STACK

Use Python.

Backend:

- FastAPI
- Pydantic
- Uvicorn

Vector database:

- Qdrant
- Qdrant running in Docker
- Qdrant Python client

LLM:

Groq API

Use the model exactly as:

model="openai/gpt-oss-120b"

Use the Groq OpenAI-compatible endpoint:

https://api.groq.com/openai/v1

Use the existing environment variable:

GROQ_API_KEY

The key is already stored in `.env`.

NEVER hardcode the API key.

Use:

    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

The model must be configurable through `.env`, but the default must remain:

    GROQ_MODEL=openai/gpt-oss-120b

Use structured JSON output / JSON schema where appropriate for query parsing and machine-readable LLM responses.

Do not expose hidden reasoning or chain-of-thought in the UI.

Only return concise explanations and evidence-based conclusions.

# 4. EMBEDDING MODEL

Use a local embedding model initially so the application does not need a second external API just for embeddings.

Default embedding model:

    BAAI/bge-small-en-v1.5

Make it configurable through:

    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

The system must automatically determine and validate the embedding dimension rather than hardcoding an incorrect value.

The embedding dimension for the configured model must be used when creating the Qdrant collection.

If the existing project already uses another embedding model and it is working correctly, inspect it first and determine whether migration is necessary.

Do NOT mix vectors from different embedding models inside the same collection.

# 5. QDRANT ARCHITECTURE

Use Qdrant as the primary vector database.

Do not replace Qdrant with Chroma, FAISS, Pinecone, or another vector DB.

Use Docker for Qdrant.

The default local URL should be:

    http://localhost:6333

Make it configurable:

    QDRANT_URL=http://localhost:6333

Recommended collection architecture:

Collection 1:

    candidate_profiles

Collection 2:

    interview_chunks

Use `candidate_id` as the logical relationship between the two.

This is preferred over mixing profile vectors and interview vectors into an unclear single collection.

The system must be able to retrieve:

Candidate profile matches
AND
Interview-content matches

for the same candidate.

# 6. DATA MODEL

Create a clear candidate model.

Example candidate:

    candidate_id
    name
    email
    phone
    location
    current_title
    total_experience
    skills
    education
    companies
    resume_text
    profile_source
    created_at
    updated_at

Do not assume every field exists.

Make optional fields safe.

Interview model:

    interview_id
    candidate_id
    interview_date
    interviewer
    interview_stage
    transcript_text
    score
    recommendation
    source_file
    created_at

Interview chunks should also contain:

    chunk_id
    candidate_id
    interview_id
    chunk_text
    section
    timestamp
    metadata

# 7. QDRANT PAYLOAD

Each vector stored in Qdrant must have useful payload metadata.

Profile payload should contain things such as:

    candidate_id
    candidate_name
    title
    location
    experience
    skills
    source_type="profile"
    document_id

Interview payload should contain:

    candidate_id
    candidate_name
    interview_id
    interview_date
    source_type="interview"
    chunk_index
    text
    timestamp
    interview_stage

Payload must allow structured filtering.

For example:

- experience
- location
- skills
- current title
- interview stage
- candidate ID

# 8. DOCUMENT UPLOAD SYSTEM

The UI must allow the HR user to upload:

Candidate profile/resume:

- PDF
- TXT
- DOCX

Interview data:

- PDF
- TXT
- DOCX
- Markdown
- SRT
- VTT

Support plain text interview transcripts/captions.

For every uploaded document:

1. Detect document type.
2. Extract text.
3. Normalize text.
4. Remove obvious garbage.
5. Preserve meaningful headings.
6. Chunk the content.
7. Generate embeddings.
8. Store the vectors in Qdrant.
9. Store metadata.
10. Record ingestion status.

Do not embed an entire large interview transcript into one vector.

Interview transcripts must be chunked.

Profile/resume documents should also be chunked when necessary, but preserve candidate-level information.

# 9. CHUNKING

Implement configurable chunking.

Default:

- chunk size around 500-800 tokens
- overlap around 50-100 tokens

Do NOT use arbitrary character slicing if a better text chunking approach is available.

For interview transcripts, preserve useful structure such as:

- interviewer questions
- candidate answers
- topics
- timestamp
- section
- speaker if available

A chunk should ideally answer:

"What evidence does this piece of the interview contain?"

# 10. INTERVIEW-AWARE SEARCH

This is the most important part of the application.

Suppose HR asks:

"Find a Python developer with 5 years of experience who knows FastAPI and RAG and demonstrated these skills during the interview."

The system should perform:

A. Semantic profile search

Search candidate profile vectors for:

Python
FastAPI
RAG
5 years
software development

B. Semantic interview search

Search interview chunks for evidence of:

FastAPI
RAG
embeddings
API development
software engineering
practical implementation

C. Structured filtering

Apply requirements such as:

experience >= requested experience where appropriate

location if specified

job title if specified

skills if specified

D. Candidate aggregation

Group retrieved profile and interview results by:

    candidate_id

E. Candidate-level relevance scoring

Create a deterministic scoring layer.

Example conceptual score:

    final_score =
        profile_score * 0.45
        +
        interview_score * 0.35
        +
        structured_match_score * 0.20

Make the weights configurable.

Do NOT claim this score is mathematically perfect.

Normalize vector similarity scores before combining them if necessary.

The scoring implementation must be explicit, testable, and deterministic.

# 11. INTERVIEW EVIDENCE BOOST

Implement an interview evidence mechanism.

If a candidate profile strongly matches the requirement but the interview contains no supporting evidence, do NOT treat that candidate as equivalent to someone who demonstrated the skill during the interview.

Example:

Profile:

    FastAPI
    RAG

Interview:

    Candidate explains FastAPI routing,
    dependency injection,
    REST APIs,
    RAG retrieval,
    embedding generation.

That candidate receives positive interview evidence.

If the interview says nothing about FastAPI, the system should not fabricate interview evidence.

The final explanation must clearly distinguish:

    Profile evidence

from:

    Interview evidence

# 12. NATURAL LANGUAGE QUERY PARSER

Users should be able to enter natural language.

Example:

    "Python developer with 5 years experience,
     FastAPI, RAG and embeddings,
     preferably Bangalore."

Use the Groq LLM to convert the query into a structured representation.

Example schema:

    {
        "job_titles": ["Python Developer"],
        "required_skills": [
            "Python",
            "FastAPI",
            "RAG",
            "Embeddings"
        ],
        "experience_years": 5,
        "location": ["Bangalore"],
        "interview_required_skills": [
            "FastAPI",
            "RAG",
            "Embeddings"
        ],
        "employment_type": null,
        "semantic_query": "Python developer with FastAPI, RAG, embeddings..."
    }

Do not force every requirement into a hard filter.

Distinguish:

hard constraints

from:

soft semantic preferences.

For example:

"I need exactly 5 years" is a different requirement from:

"around 5 years."

The system should preserve that distinction.

# 13. SEARCH PIPELINE

Implement this flow:

    User query
       ↓
    Query parser
       ↓
    Search requirements
       ↓
    Profile embedding
       ↓
    Qdrant profile search
       ↓
    Interview embedding
       ↓
    Qdrant interview search
       ↓
    Group results by candidate_id
       ↓
    Normalize scores
       ↓
    Apply structured matching
       ↓
    Calculate final candidate relevance
       ↓
    Select top candidates
       ↓
    Fetch supporting evidence
       ↓
    LLM summary
       ↓
    UI

# 14. RESULT FORMAT

For every returned candidate, provide:

    candidate_id
    candidate_name
    current_title
    location
    experience
    overall_relevance_score

Then provide:

    Why this candidate matches

And:

    Profile evidence

And:

    Interview evidence

And:

    Matching skills

And:

    Missing / uncertain requirements

And optionally:

    Interview date
    Interview stage
    Interview score
    Source documents

Example UI result:

    # 1 — Arjun Kumar

    Python AI Engineer
    Bangalore
    5.2 years experience

    Relevance: High

    Profile evidence:
    • Python
    • FastAPI
    • RAG
    • embeddings
    • LLM applications

    Interview evidence:
    • Explained FastAPI dependency injection
    • Designed a REST API
    • Explained RAG retrieval flow
    • Discussed embedding generation and vector search

    Requirement coverage:
    ✓ Python
    ✓ FastAPI
    ✓ RAG
    ✓ Embeddings
    ✓ ~5 years experience

    Interview evidence:
    ✓ FastAPI demonstrated
    ✓ RAG knowledge demonstrated

# 15. IMPORTANT FACTUALITY RULE

The application must NEVER invent candidate information.

Never create:

- fake skills
- fake years of experience
- fake companies
- fake interview answers
- fake interview performance
- fake recommendations

Every claim shown to HR must originate from:

1. Candidate profile
2. Interview transcript/report
3. Structured candidate metadata

If the system cannot find evidence:

    "No interview evidence found."

Do not say:

    "The candidate demonstrated FastAPI"

unless actual interview evidence supports that statement.

# 16. LLM RESPONSIBILITIES

Use the Groq model primarily for:

1. Natural-language query parsing.
2. Search requirement interpretation.
3. Candidate result summarization.
4. Evidence-based explanation.
5. Optional comparison of top candidates.

Do NOT use the LLM as the primary database.

Do NOT ask the LLM to memorize the candidate database.

Qdrant is the retrieval system.

The LLM is the reasoning/generation layer operating over retrieved evidence.

# 17. RAG CONTEXT

For final generation, construct a context such as:

    Candidate Profile:
    ...

    Relevant Interview Evidence:
    ...

    Requirements:
    ...

Then ask the LLM to produce a concise recruiter-facing summary.

The LLM must only use the supplied context.

If evidence is insufficient, explicitly state that.

# 18. UI

Build a very clean, simple recruiter UI.

Do not create a complicated enterprise dashboard.

Main page:

    Candidate Search

    [ Describe the candidate you need...                ]

    [ Search ]

Below that:

    Top Matching Candidates

Each candidate should appear as a clean card.

Example:

    ┌──────────────────────────────────────────────┐
    │ Arjun Kumar                                  │
    │ Python AI Engineer                           │
    │ Bangalore • 5.2 years                       │
    │                                              │
    │ Match: 91%                                  │
    │                                              │
    │ Python • FastAPI • RAG • Embeddings         │
    │                                              │
    │ Profile: Strong                             │
    │ Interview Evidence: Strong                  │
    │                                              │
    │ [View Candidate]                             │
    └──────────────────────────────────────────────┘

Clicking the candidate should show:

    Candidate Profile

    Experience
    Skills
    Education
    Work History

    Interview Information

    Interview date
    Interview stage
    Interview score

    Relevant Interview Evidence

    [timestamp] Candidate discussed FastAPI...

    [timestamp] Candidate explained RAG...

    AI Summary

    ...

# 19. UPLOAD UI

Create a separate:

    Upload Candidates

page/section.

Allow HR to:

    Upload Profile

    Select candidate or enter candidate ID

    Upload interview report/transcript

Show ingestion status:

    ✓ Profile processed
    ✓ Embeddings generated
    ✓ Stored in Qdrant

or:

    ✗ Failed
    [View error]

Do not silently fail ingestion.

# 20. CANDIDATE MANAGEMENT

The system should maintain candidate identity.

If a profile and interview belong to:

    candidate_id = CAND_001

both must be linked.

Avoid creating duplicate candidate identities just because the uploaded filenames differ.

Support explicit candidate ID during upload.

Where safe, allow candidate identification based on filename/content, but provide a deterministic override.

# 21. DATABASE

For the MVP, Qdrant can hold vector data and payload.

If structured persistence is needed outside Qdrant, use SQLite initially.

Do NOT introduce PostgreSQL unnecessarily for the first working MVP.

Keep the architecture easy to move later to PostgreSQL.

# 22. PROJECT STRUCTURE

Create a clean structure similar to:

    candidate-rag/
    │
    ├── app/
    │   ├── main.py
    │   │
    │   ├── api/
    │   │   ├── search.py
    │   │   ├── upload.py
    │   │   └── candidates.py
    │   │
    │   ├── core/
    │   │   ├── config.py
    │   │   └── logging.py
    │   │
    │   ├── ingestion/
    │   │   ├── loaders.py
    │   │   ├── parser.py
    │   │   ├── chunker.py
    │   │   └── pipeline.py
    │   │
    │   ├── embeddings/
    │   │   └── service.py
    │   │
    │   ├── qdrant/
    │   │   ├── client.py
    │   │   ├── collections.py
    │   │   └── search.py
    │   │
    │   ├── retrieval/
    │   │   ├── profile_retriever.py
    │   │   ├── interview_retriever.py
    │   │   └── candidate_ranker.py
    │   │
    │   ├── llm/
    │   │   ├── groq_client.py
    │   │   ├── query_parser.py
    │   │   └── summarizer.py
    │   │
    │   ├── schemas/
    │   │   ├── candidate.py
    │   │   ├── interview.py
    │   │   ├── search.py
    │   │   └── responses.py
    │   │
    │   ├── templates/
    │   │   └── index.html
    │   │
    │   └── static/
    │       ├── css/
    │       └── js/
    │
    ├── data/
    │   ├── uploads/
    │   └── processed/
    │
    ├── tests/
    │   ├── test_ingestion.py
    │   ├── test_embeddings.py
    │   ├── test_search.py
    │   └── test_ranking.py
    │
    ├── scripts/
    │   ├── init_qdrant.py
    │   └── ingest.py
    │
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── .env.example
    ├── .gitignore
    ├── README.md
    ├── run.sh
    └── run.bat

Adapt this structure based on the existing repository instead of unnecessarily duplicating already-good code.

# 23. FASTAPI

Use FastAPI as the backend.

Create endpoints such as:

    GET  /
    GET  /health
    GET  /api/health

    POST /api/search

    POST /api/candidates/upload

    POST /api/interviews/upload

    GET  /api/candidates/{candidate_id}

    GET  /api/candidates/{candidate_id}/evidence

    POST /api/ingestion/reindex

Provide proper Pydantic request/response models.

Enable CORS appropriately for local development.

# 24. MAIN APPLICATION

There must be a clear main application entry point:

    app/main.py

Running:

    uvicorn app.main:app --host 0.0.0.0 --port 8000

must start the application.

Opening:

    http://localhost:8000

should show the UI.

Do not require the developer to manually run five different processes for the basic application.

# 25. DOCKER COMPOSE

Create:

    docker-compose.yml

with at minimum:

    app
    qdrant

The Qdrant container should use the official:

    qdrant/qdrant

image.

Expose:

    6333
    6334

Persist Qdrant storage using a volume.

The application should communicate with Qdrant using the Docker service name when running under Compose:

    http://qdrant:6333

while local/non-container development should support:

    http://localhost:6333

Make the Qdrant URL configurable through:

    QDRANT_URL

Do not hardcode environment-specific URLs everywhere in the code.

# 26. DOCKER STARTUP

The project should be easy to run.

Preferred command:

    docker compose up --build

This should start:

    Qdrant
    FastAPI application

and make the UI accessible.

Do not make the application responsible for manually launching Docker itself.

Docker Compose is the correct orchestration boundary.

# 27. LOCAL DESKTOP SUPPORT

The same GitHub repository must work on:

A. Lightning AI Studio

B. Windows local desktop with Docker installed

C. Linux/macOS if possible

There should be minimal environment-specific code.

Provide:

    run.bat

for Windows.

Provide:

    run.sh

for Linux/macOS/Lightning.

Example Windows usage:

    run.bat

Example Linux/Lightning usage:

    chmod +x run.sh
    ./run.sh

The README must explain this clearly.

# 28. LIGHTNING AI SUPPORT

The application will initially be developed and tested inside Lightning AI Studio.

Do not assume:

    Windows
    CUDA
    NVIDIA GPU

unless detected.

The MVP should work on CPU.

Detect available hardware and avoid hardcoded CUDA assumptions.

The code should work in a standard Lightning AI Linux environment.

Qdrant will run in Docker.

The application should connect to Qdrant using the appropriate URL depending on whether it is running:

    inside Docker Compose

or:

    directly on the host.

# 29. ENVIRONMENT VARIABLES

Create `.env.example`:

    GROQ_API_KEY=
    GROQ_MODEL=openai/gpt-oss-120b
    GROQ_BASE_URL=https://api.groq.com/openai/v1

    QDRANT_URL=http://localhost:6333

    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

    PROFILE_COLLECTION=candidate_profiles
    INTERVIEW_COLLECTION=interview_chunks

    PROFILE_WEIGHT=0.45
    INTERVIEW_WEIGHT=0.35
    STRUCTURED_WEIGHT=0.20

    TOP_K_PROFILE=20
    TOP_K_INTERVIEW=50
    FINAL_TOP_K=10

    CHUNK_SIZE=700
    CHUNK_OVERLAP=100

The actual `.env` must NOT be committed to Git.

Add `.env` to `.gitignore`.

# 30. LOGGING

Implement proper logging.

Log:

- application startup
- Qdrant connection
- collection creation
- upload received
- document parsing
- chunk count
- embedding generation
- Qdrant upsert
- search request
- retrieval counts
- candidate aggregation
- errors

Never log:

    GROQ_API_KEY

or other secrets.

# 31. HEALTH CHECK

Implement:

    GET /health

It should report the basic application status.

Also check Qdrant connectivity.

Example:

    {
        "status": "ok",
        "qdrant": "connected",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "llm": "configured"
    }

Do not expose secrets.

# 32. ERROR HANDLING

Handle:

- Qdrant unavailable
- invalid file types
- empty PDFs
- malformed documents
- embedding failures
- Groq API failures
- malformed LLM JSON
- missing candidate IDs
- duplicate ingestion
- empty search results

Show useful errors.

Never silently return an empty result when an infrastructure component failed.

Distinguish:

    "No candidates matched"

from:

    "Search failed because Qdrant is unavailable."

# 33. RE-INDEXING

Provide a way to rebuild/reindex embeddings.

For example:

    python scripts/ingest.py

and:

    python scripts/init_qdrant.py

Also expose an optional admin/reindex API if useful.

Do not require destructive deletion for every update.

Use deterministic IDs where practical to support upserts.

# 34. UPDATE BEHAVIOR

If the same candidate uploads a new profile:

update the candidate profile vector.

If a new interview is uploaded:

add/update the corresponding interview chunks.

Avoid creating duplicate chunks every time the application starts.

# 35. SEARCH RESULT EVIDENCE

For every candidate returned, keep the retrieved chunks that contributed to the candidate's result.

This allows the UI to display:

    Why this candidate matched.

The system should be able to trace:

    Candidate
      ↓
    profile match
      ↓
    interview match
      ↓
    supporting text

This traceability is essential.

# 36. CANDIDATE DETAIL PAGE

When the recruiter selects a candidate, display:

    Candidate name
    Current title
    Years of experience
    Location
    Skills
    Resume/profile summary

Then:

    Interview history

Then:

    Relevant interview evidence for the current search

Then:

    AI summary

Then:

    Original source files / filenames

# 37. SEARCH QUALITY

Do not simply return the nearest vector.

The search engine should distinguish:

    Profile similarity

    Interview similarity

    Structured requirement coverage

    Missing requirements

Create a clear ranking explanation.

Example internal result representation:

    {
        "candidate_id": "CAND_001",
        "profile_score": 0.88,
        "interview_score": 0.94,
        "structured_score": 1.00,
        "final_score": 0.925,
        "matched_skills": [...],
        "missing_skills": [...],
        "profile_evidence": [...],
        "interview_evidence": [...]
    }

# 38. AVOID SCORE MISUSE

Similarity scores from vector databases should not be presented as literal probabilities.

Do not write:

    "92% probability this person is qualified"

unless the system has actually been calibrated for probability estimation.

Use wording like:

    "Relevance score"

or:

    "Search match score"

# 39. SECURITY

Keep:

    GROQ_API_KEY

server-side.

Never expose it to browser JavaScript.

Never include it in API responses.

Never include it in logs.

Validate upload filenames.

Prevent path traversal.

Limit upload sizes.

Only permit supported extensions.

Store uploaded files under the application data directory.

# 40. TEST DATA

Create a small synthetic test dataset.

Create approximately:

    10 candidates

with:

    profiles/resumes

and:

    5 candidates with interview transcripts/reports

Use realistic but synthetic content.

Do NOT use real people's private information.

Example:

Candidate 1:
Python + FastAPI + RAG
5 years
strong interview evidence

Candidate 2:
Python + Django
5 years
weak FastAPI evidence

Candidate 3:
Java + Spring
6 years

Candidate 4:
Python + FastAPI
3 years

Candidate 5:
Python + RAG + LangChain
5 years
strong interview evidence

etc.

Use this dataset to demonstrate ranking behavior.

# 41. REQUIRED DEMO QUERIES

Create test searches for:

1.

    Python developer with 5 years experience,
    FastAPI and RAG.

2.

    Python backend engineer with strong API
    development experience.

3.

    Generative AI engineer with RAG,
    embeddings and LLM experience.

4.

    Python engineer who demonstrated
    FastAPI knowledge during the interview.

5.

    Candidate with around 5 years of experience
    who has worked with vector databases.

The final test suite should verify that interview-aware ranking changes appropriately.

# 42. UNIT TESTS

Write tests for:

- document parsing
- text chunking
- embedding generation
- collection creation
- candidate ingestion
- interview ingestion
- semantic search
- metadata filtering
- candidate aggregation
- ranking
- query parsing
- API endpoints

Mock external Groq calls in tests.

Do not make unit tests depend on an actual Groq API call.

# 43. INTEGRATION TEST

Create an integration test that:

1. Creates test candidates.
2. Ingests profiles.
3. Ingests interviews.
4. Executes a recruiter search.
5. Verifies candidate IDs are returned.
6. Verifies profile and interview evidence are included.
7. Verifies ranking is deterministic.

# 44. API RESPONSE

The search endpoint should return structured JSON like:

    {
      "query": "...",
      "interpreted_requirements": {
        "skills": [],
        "experience": null,
        "location": null
      },
      "results": [
        {
          "rank": 1,
          "candidate_id": "...",
          "name": "...",
          "title": "...",
          "experience": 5.2,
          "location": "...",
          "relevance_score": 0.91,
          "matched_skills": [],
          "missing_requirements": [],
          "profile_evidence": [],
          "interview_evidence": [],
          "summary": "..."
        }
      ],
      "result_count": 5
    }

# 45. FRONTEND

Use a simple frontend served by FastAPI.

Prefer:

- HTML
- CSS
- vanilla JavaScript

Do NOT introduce React unless the existing project already uses it and it is worth preserving.

The UI must be:

- clean
- minimal
- responsive
- easy to understand
- recruiter-focused

Do not make the interface look like a generic developer dashboard.

# 46. UI SECTIONS

Main navigation:

    Search
    Upload
    Candidates

Search page:

    natural language search box

    Search button

    result count

    candidate cards

Upload page:

    profile upload

    candidate ID

    interview upload

    ingestion status

Candidates page:

    list of candidates

    profile information

    interview availability

# 47. EMPTY SEARCH

When there are no results:

    No matching candidates were found.

Then show useful interpretation:

    Search considered:
    Python
    FastAPI
    RAG
    5 years experience

Do not fabricate candidates.

# 48. LOADING STATES

The UI should visibly show:

    Processing documents...

    Searching candidates...

    Generating summaries...

This avoids the application appearing frozen during embedding/LLM calls.

# 49. PERFORMANCE

Do not load the embedding model once per request.

Initialize the embedding model once and reuse it.

Reuse Qdrant connections.

Reuse the Groq client.

Avoid re-embedding unchanged data.

Use batching during ingestion.

Do not perform unnecessary LLM calls.

For a search:

1. Parse query once.
2. Perform vector retrieval.
3. Aggregate candidates.
4. Generate summaries only for the top candidates.

# 50. LLM CALL STRATEGY

Use the LLM only where it adds value.

Recommended:

Call 1:
Natural-language query → structured requirements.

Call 2:
Top candidate evidence → recruiter-facing summary.

Avoid one LLM call per candidate for dozens of candidates.

For the MVP, summarize only the top 5-10 candidates.

# 51. RAG PROMPTING

Create dedicated prompt templates.

Query parser prompt:

The model must extract:

- job title
- skills
- experience
- location
- interview-required skills
- hard vs soft constraints
- semantic search query

Candidate summary prompt:

Given only the retrieved candidate profile and interview evidence, explain:

- why the candidate matches
- which requirements are supported
- which evidence comes from the interview
- what is missing
- what remains uncertain

Do not invent information.

# 52. INTERVIEW CAPTION HANDLING

Interview captions may look like:

    INTERVIEWER: Tell me about FastAPI.

    CANDIDATE: I used FastAPI to build...

Preserve speaker labels when possible.

The system should prefer candidate statements as evidence of demonstrated knowledge.

Do not treat interviewer questions alone as evidence that the candidate possesses a skill.

For example:

    INTERVIEWER:
    Have you used FastAPI?

does NOT prove that the candidate knows FastAPI.

Evidence such as:

    CANDIDATE:
    I used FastAPI to implement...

should count as candidate evidence.

# 53. INTERVIEW EVIDENCE TYPES

Where possible, classify interview evidence as:

    mentioned
    explained
    demonstrated
    implemented
    uncertain

Do not claim "demonstrated" unless the transcript actually supports that interpretation.

# 54. OBSERVABILITY / DEBUG MODE

Create a developer/debug option.

When enabled, allow the developer to see:

    parsed query
    Qdrant profile hits
    Qdrant interview hits
    candidate aggregation
    scoring components

Do not expose secrets.

The normal recruiter UI should remain simple.

# 55. README

Create a complete README explaining:

1. Project overview
2. Architecture
3. Prerequisites
4. Lightning AI setup
5. Local desktop setup
6. Docker setup
7. `.env` configuration
8. Running the application
9. Uploading documents
10. Running ingestion
11. Performing searches
12. Qdrant dashboard
13. API endpoints
14. Testing
15. Troubleshooting
16. Project structure

Include exact commands.

# 56. GIT / GITHUB

The repository must be GitHub-ready.

Create:

    .gitignore

including:

    .env
    __pycache__/
    .venv/
    *.pyc
    data/uploads/
    qdrant_storage/
    logs/
    .DS_Store

Do NOT commit:

- API keys
- private candidate data
- generated embeddings
- Qdrant storage
- uploaded documents

The repository must clone and run from scratch.

# 57. DO NOT OVERENGINEER

The first milestone is a reliable MVP.

Do NOT introduce unnecessarily:

- Kubernetes
- Redis
- Kafka
- Celery
- microservices
- PostgreSQL
- React
- LangGraph
- complex agent frameworks

unless the existing repository already requires them or there is a clear reason.

Start with:

FastAPI
Qdrant
local embeddings
Groq
HTML/CSS/JS
Docker Compose

Keep the system modular so those components can be introduced later.

# 58. DO NOT DEPEND ON LANGCHAIN UNLESS JUSTIFIED

If the existing code uses LangChain, inspect it.

If useful, reuse it.

But do not force LangChain into every component.

Prefer direct, understandable implementations for:

- embeddings
- Qdrant
- retrieval
- ranking
- Groq calls

The application should be easy for a developer to understand and debug.

# 59. EXISTING CODE MIGRATION

Before changing anything:

1. Inspect all existing files.
2. Identify useful code.
3. Identify broken code.
4. Identify duplicate implementations.
5. Identify the current vector DB.
6. Identify current embedding model.
7. Identify current RAG pipeline.
8. Identify existing UI.
9. Identify current Docker setup.

Then create a migration plan.

Preserve useful functionality.

Replace broken or unnecessary components.

Do not destroy working functionality without a reason.

# 60. DEVELOPMENT ORDER

Implement in this order:

PHASE 1

Understand existing code.

PHASE 2

Create clean configuration.

PHASE 3

Connect Qdrant.

PHASE 4

Create candidate profile collection.

PHASE 5

Create interview chunk collection.

PHASE 6

Implement document loading.

PHASE 7

Implement chunking.

PHASE 8

Implement embeddings.

PHASE 9

Implement profile ingestion.

PHASE 10

Implement interview ingestion.

PHASE 11

Implement semantic profile retrieval.

PHASE 12

Implement semantic interview retrieval.

PHASE 13

Implement candidate-level aggregation.

PHASE 14

Implement structured filters.

PHASE 15

Implement Groq query parser.

PHASE 16

Implement evidence-based candidate summarization.

PHASE 17

Implement FastAPI API.

PHASE 18

Implement UI.

PHASE 19

Implement Docker Compose.

PHASE 20

Implement tests.

PHASE 21

Create synthetic demo data.

PHASE 22

Run complete end-to-end verification.

# 61. FINAL END-TO-END TEST

After implementation, actually run the application.

Do not merely write code.

Verify:

    docker compose up --build

Then verify:

    Qdrant starts

    FastAPI starts

    health endpoint works

    Qdrant connection works

    UI loads

    documents can be uploaded

    profile vectors are created

    interview vectors are created

    search works

    candidate aggregation works

    top candidates appear

    interview evidence appears

    Groq summary works

    no API key is exposed

    application can be stopped and restarted

    Qdrant data remains persistent

# 62. FINAL ACCEPTANCE CRITERIA

The application is considered complete only when this exact scenario works:

Step 1:

Upload 10 candidate profiles.

Step 2:

Upload interviews for 5 of those candidates.

Step 3:

The system processes and embeds all documents.

Step 4:

HR enters:

"Find a Python developer with around 5 years of experience who has strong FastAPI and software development skills, has worked with RAG and embeddings, and should have demonstrated FastAPI knowledge during the interview."

Step 5:

The system:

- interprets the query
- searches profile vectors
- searches interview vectors
- applies structured candidate constraints
- aggregates evidence by candidate
- ranks candidates
- returns the top relevant candidates
- summarizes each candidate
- shows profile evidence
- shows interview evidence
- clearly identifies missing/uncertain evidence

Step 6:

The UI displays the candidates in a simple recruiter-friendly format.

# 63. VERY IMPORTANT IMPLEMENTATION RULE

Do not stop after creating the architecture.

Actually implement the application.

Do not give me only:

- pseudocode
- a design document
- partial snippets
- TODO placeholders
- fake API responses

Generate the real project files.

After implementation:

1. Run tests.
2. Start Docker.
3. Start Qdrant.
4. Start FastAPI.
5. Run ingestion.
6. Run search.
7. Fix errors.
8. Verify the UI.
9. Verify the complete end-to-end workflow.

If something is incompatible with the existing repository, adapt it rather than abandoning the implementation.

# 64. FINAL DELIVERABLE

At the end, provide:

1. Final project structure.
2. Files created.
3. Files modified.
4. Files removed and why.
5. Exact commands to run on Lightning AI.
6. Exact commands to run on Windows.
7. `.env` variables required.
8. Qdrant collection names.
9. Embedding model used.
10. Groq model used.
11. API endpoints.
12. Example search query.
13. Example expected result.
14. Test results.
15. Known limitations.
16. Recommended next improvements.

Most importantly, leave the repository in a state where I can:

    git add .
    git commit
    git push

then on another machine:

    git clone <repo>
    cd <repo>
    docker compose up --build

and access:

    http://localhost:8000

with the complete candidate search application.

Do not assume that my current basic RAG implementation is correct.

Inspect it first.

Then build the complete system around the actual requirement described above.