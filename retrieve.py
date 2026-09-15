import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai


# --------------------------------
# 1. Load environment variables
# --------------------------------

load_dotenv()


# --------------------------------
# 2. Configuration
# --------------------------------

THRESHOLD = 1.5
TOP_K = 5


# --------------------------------
# 3. Load embedding model
# --------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# 4. Connect to ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="sre_docs"
)


# --------------------------------
# 5. Connect to Gemini
# --------------------------------

gemini_client = genai.Client()


# --------------------------------
# 6. Ask user
# --------------------------------

query = input("\nAsk a question: ")


# --------------------------------
# 7. Create query embedding
# --------------------------------

query_embedding = embedding_model.encode(
    query
).tolist()


# --------------------------------
# 8. Retrieve relevant chunks
# --------------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


# --------------------------------
# 9. Check relevance
# --------------------------------

best_distance = results["distances"][0][0]

print(
    f"\nBest distance: {best_distance}"
)


if best_distance > THRESHOLD:

    print(
        "\nI don't have enough information "
        "in the provided documentation."
    )

else:

    print("\nRelevant information found.")


    # --------------------------------
    # 10. Build context
    # --------------------------------

    context_parts = []

    sources = []

    for i in range(
        len(results["documents"][0])
    ):

        document = results["documents"][0][i]

        metadata = results["metadatas"][0][i]

        distance = results["distances"][0][i]


        context_parts.append(
            f"""
Source: {metadata['source']}
Section: {metadata['section']}

{document}
"""
        )


        # Store source information
        source = {
            "url": metadata["source"],
            "title": metadata["title"],
            "section": metadata["section"],
            "distance": distance
        }

        sources.append(source)


    context = "\n\n".join(
        context_parts
    )


    # --------------------------------
    # 11. Create grounded prompt
    # --------------------------------

    prompt = f"""
You are an SRE documentation assistant.

Answer the user's question using ONLY
the information provided in the context.

Rules:

1. Do not use outside knowledge.
2. Do not make up information.
3. If the context does not contain enough
   information, say:

"I don't have enough information
in the provided documentation."

4. Give a clear and concise answer.

Context:
{context}

User question:
{query}
"""


    # --------------------------------
    # 12. Generate answer
    # --------------------------------

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )


    # --------------------------------
    # 13. Display answer
    # --------------------------------

    print("\nAnswer:")

    print(response.text)


    # --------------------------------
    # 14. Display sources
    # --------------------------------

    print("\nSources:")

    displayed_sources = set()

    for source in sources:

        source_key = (
            source["url"],
            source["section"]
        )

        # Avoid duplicate sources
        if source_key in displayed_sources:
            continue

        displayed_sources.add(
            source_key
        )

        print(
            f"\n- {source['title']}"
        )

        print(
            f"  Section: {source['section']}"
        )

        print(
            f"  URL: {source['url']}"
        )