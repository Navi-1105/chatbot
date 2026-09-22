import prefer_app_packages  # noqa: F401  # must run before chromadb / OpenTelemetry

import os

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

from observability import (
    GEMINI_MODEL,
    flush,
    generation_observation,
    new_session_id,
    rag_trace,
    retrieval_output,
    retriever_observation,
    usage_details,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CHROMADB DEBUG INFORMATION
# ============================================================

print("========== CHROMA DEBUG ==========")

print(
    "Chroma version:",
    getattr(
        chromadb,
        "__version__",
        "UNKNOWN"
    )
)

print(
    "Chroma location:",
    chromadb.__file__
)

print(
    "PersistentClient available:",
    hasattr(
        chromadb,
        "PersistentClient"
    )
)

print("==================================\n")


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SRE Documentation Assistant",
    page_icon="🔧",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🔧 SRE Documentation Assistant"
)

st.write(
    "Ask questions about Site Reliability Engineering "
    "and Azure Monitor documentation."
)


# ============================================================
# LOAD MODELS / DATABASE
# ============================================================

@st.cache_resource
def load_resources():

    print(
        "========== LOADING RESOURCES =========="
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "Loading SentenceTransformer..."
    )

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print(
        "SentenceTransformer loaded successfully."
    )

    # --------------------------------------------------------
    # Connect to ChromaDB
    # --------------------------------------------------------

    print(
        "Connecting to ChromaDB..."
    )

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    print(
        "Chroma PersistentClient created successfully."
    )

    # --------------------------------------------------------
    # Load collection
    # --------------------------------------------------------

    collection = client.get_collection(
        name="sre_docs"
    )

    print(
        "Chroma collection loaded:",
        collection.name
    )

    # --------------------------------------------------------
    # Initialize Gemini
    # --------------------------------------------------------

    print(
        "Initializing Gemini client..."
    )

    gemini_client = genai.Client()

    print(
        "Gemini client initialized successfully."
    )

    print(
        "========================================"
    )

    return (
        embedding_model,
        collection,
        gemini_client
    )


# ============================================================
# INITIALIZE RESOURCES
# ============================================================

embedding_model, collection, gemini_client = (
    load_resources()
)


# ============================================================
# LANGFUSE SESSION
# ============================================================

if "langfuse_session_id" not in st.session_state:

    st.session_state.langfuse_session_id = (
        new_session_id()
    )


# ============================================================
# USER INPUT
# ============================================================

query = st.text_input(
    "Ask your question:",
    placeholder="What is Azure Monitor?"
)


# ============================================================
# RAG PIPELINE
# ============================================================

if query:

    with st.spinner(
        "Searching documentation..."
    ):

        # ====================================================
        # ROOT RAG TRACE
        # ====================================================

        with rag_trace(
            query=query,
            entrypoint="streamlit",
            session_id=(
                st.session_state.langfuse_session_id
            ),
            top_k=TOP_K,
        ) as root_span:

            # =================================================
            # RETRIEVAL
            # =================================================

            with retriever_observation(
                query=query,
                top_k=TOP_K
            ) as retriever_span:

                # ------------------------------------------------
                # Convert question to embedding
                # ------------------------------------------------

                query_embedding = (
                    embedding_model
                    .encode(query)
                    .tolist()
                )

                # ------------------------------------------------
                # Retrieve Top-K documents
                # ------------------------------------------------

                results = collection.query(
                    query_embeddings=[
                        query_embedding
                    ],
                    n_results=TOP_K,
                    include=[
                        "documents",
                        "metadatas",
                        "distances"
                    ]
                )

                # ------------------------------------------------
                # Log retrieval output
                # ------------------------------------------------

                if retriever_span:

                    retriever_span.update(
                        output=retrieval_output(
                            results
                        )
                    )

            # =================================================
            # BUILD CONTEXT
            # =================================================

            context_parts = []

            for i in range(
                len(
                    results["documents"][0]
                )
            ):

                document = (
                    results["documents"][0][i]
                )

                metadata = (
                    results["metadatas"][0][i]
                )

                context_parts.append(
                    f"""
Source: {metadata['source']}
Section: {metadata['section']}

{document}
"""
                )

            context = "\n\n".join(
                context_parts
            )

            # =================================================
            # GROUNDED PROMPT
            # =================================================

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

            # =================================================
            # GENERATE ANSWER
            # =================================================

            with generation_observation(
                prompt=prompt
            ) as generation:

                response = (
                    gemini_client
                    .models
                    .generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt
                    )
                )

                # ------------------------------------------------
                # Log generation information
                # ------------------------------------------------

                if generation:

                    generation.update(
                        output=response.text,
                        usage_details=usage_details(
                            response
                        )
                    )

            # =================================================
            # UPDATE ROOT TRACE
            # =================================================

            if root_span:

                root_span.update(
                    output={
                        "answer": response.text,
                        "sources": (
                            retrieval_output(
                                results
                            )
                        )
                    }
                )

            # =================================================
            # FLUSH LANGFUSE
            # =================================================

            flush()


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    st.subheader(
        "Answer"
    )

    st.write(
        response.text
    )


    # ========================================================
    # DISPLAY SOURCES
    # ========================================================

    st.subheader(
        "Sources"
    )

    displayed_sources = set()

    for metadata in results["metadatas"][0]:

        source_key = (
            metadata["source"],
            metadata["section"]
        )

        # ----------------------------------------------------
        # Avoid duplicate sources
        # ----------------------------------------------------

        if source_key in displayed_sources:
            continue

        displayed_sources.add(
            source_key
        )

        st.markdown(
            f"**{metadata['section']}**"
        )

        st.caption(
            metadata["source"]
        )