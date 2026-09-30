from app.retriever import search_jobs


results = search_jobs(
    query="""
    Python developer who builds RAG applications,
    FastAPI services, embedding pipelines and LLM systems
    """,
    experience=5,
    skills=[
        "python",
        "fastapi",
        "rag",
        "embeddings"
    ],
    limit=10,
)


for result in results:

    print("=" * 80)

    print("Score:", result.score)

    payload = result.payload

    print("Title:", payload["title"])
    print("Company:", payload["company"])
    print("Location:", payload["location"])
    print("Experience:",
          payload["experience_min"],
          "-",
          payload["experience_max"])

    print("Skills:", payload["skills"])

    print("URL:", payload["url"])