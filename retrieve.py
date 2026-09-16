from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from observability import (
    GEMINI_MODEL,
    flush,
    generation_observation,
    rag_trace,
    retrieval_output,
    retriever_observation,
    usage_details,
)


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


with rag_trace(
    query=query,
    entrypoint="cli",
    top_k=TOP_K,
) as root_span:

    # --------------------------------
    # 7. Retrieve relevant chunks
    # --------------------------------

    with retriever_observation(
        query=query,
        top_k=TOP_K
    ) as retriever_span:

        query_embedding = embedding_model.encode(
            query
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        if retriever_span:
            retriever_span.update(
                output=retrieval_output(
                    results
                )
            )


    # --------------------------------
    # 8. Check relevance
    # --------------------------------

    best_distance = results["distances"][0][0]

    print(
        f"\nBest distance: {best_distance}"
    )


    if best_distance > THRESHOLD:

        no_answer = (
            "I don't have enough information "
            "in the provided documentation."
        )

        print(
            f"\n{no_answer}"
        )

        if root_span:
            root_span.update(
                output={
                    "answer": no_answer,
                    "best_distance": best_distance
                }
            )

    else:

        print("\nRelevant information found.")


        # --------------------------------
        # 9. Build context
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
        # 10. Create grounded prompt
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
        # 11. Generate answer
        # --------------------------------

        with generation_observation(
            prompt=prompt
        ) as generation:

            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            if generation:
                generation.update(
                    output=response.text,
                    usage_details=usage_details(
                        response
                    )
                )

        if root_span:
            root_span.update(
                output={
                    "answer": response.text,
                    "sources": retrieval_output(
                        results
                    )
                }
            )


        # --------------------------------
        # 12. Display answer
        # --------------------------------

        print("\nAnswer:")

        print(response.text)


        # --------------------------------
        # 13. Display sources
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

    flush()
