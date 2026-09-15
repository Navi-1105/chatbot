import os

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

load_dotenv()


# --------------------------------
# Configuration
# --------------------------------

TOP_K = 5


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="SRE Documentation Assistant",
    page_icon="🔧",
    layout="centered"
)


# --------------------------------
# Title
# --------------------------------

st.title(" SRE Documentation Assistant")

st.write(
    "Ask questions about Site Reliability Engineering "
    "and Azure Monitor documentation."
)


# --------------------------------
# Load models / database
# --------------------------------

@st.cache_resource
def load_resources():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_collection(
        name="sre_docs"
    )

    gemini_client = genai.Client()

    return (
        embedding_model,
        collection,
        gemini_client
    )


embedding_model, collection, gemini_client = load_resources()


# --------------------------------
# User input
# --------------------------------

query = st.text_input(
    "Ask your question:",
    placeholder="What is Azure Monitor?"
)


# --------------------------------
# RAG pipeline
# --------------------------------

if query:

    with st.spinner("Searching documentation..."):

        # Convert question to embedding
        query_embedding = embedding_model.encode(
            query
        ).tolist()

        # Retrieve Top-K documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=[
                "documents",
                "metadatas"
            ]
        )

        # Build context
        context_parts = []

        for i in range(
            len(results["documents"][0])
        ):

            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i]

            context_parts.append(
                f"""
Source: {metadata['source']}
Section: {metadata['section']}

{document}
"""
            )

        context = "\n\n".join(context_parts)

        # Grounded prompt
        prompt = f"""
You are an SRE documentation assistant.

Answer the user's question using ONLY
the information provided in the context below.

If the context does not contain enough
information to answer the question,
say:

"I don't have enough information
in the provided documentation."

Do not use outside knowledge.
Do not make up information.

Context:
{context}

User question:
{query}
"""

        # Generate answer
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

    # --------------------------------
    # Display answer
    # --------------------------------

    st.subheader("Answer")

    st.write(response.text)

    # --------------------------------
    # Display sources
    # --------------------------------

    st.subheader("Sources")

    displayed_sources = set()

    for metadata in results["metadatas"][0]:

        source_key = (
            metadata["source"],
            metadata["section"]
        )

        if source_key in displayed_sources:
            continue

        displayed_sources.add(source_key)

        st.markdown(
            f"**{metadata['section']}**"
        )

        st.caption(
            metadata["source"]
        )