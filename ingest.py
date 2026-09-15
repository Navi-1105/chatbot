from sentence_transformers import SentenceTransformer
import chromadb
import re

# --------------------------------
# 1. Load embedding model
# --------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# --------------------------------
# 2. Connect to ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="sre_docs"
)

# --------------------------------
# 3. Read document
# --------------------------------

with open(
    "data/sre_basics.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()

# --------------------------------
# 4. Structure-aware chunking
# --------------------------------

sections = re.split(
    r"\n(?=# )",
    text
)

chunks = []

for section in sections:

    section = section.strip()

    if not section:
        continue

    # Split section into paragraphs
    paragraphs = [
        p.strip()
        for p in section.split("\n\n")
        if p.strip()
    ]

    # Keep the heading with the section content
    if len(paragraphs) > 1:

        heading = paragraphs[0]

        content = "\n\n".join(
            paragraphs[1:]
        )

        chunk = f"{heading}\n\n{content}"

    else:
        chunk = section

    chunks.append(chunk)

# --------------------------------
# 5. Display chunks
# --------------------------------

print("\nCreated chunks:")

for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i} ---")
    print(chunk)

# --------------------------------
# 6. Generate embeddings
# --------------------------------

embeddings = embedding_model.encode(
    chunks
).tolist()

# --------------------------------
# 7. Create IDs
# --------------------------------

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

# --------------------------------
# 8. Store in ChromaDB
# --------------------------------

collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print(
    f"\nSuccessfully stored {len(chunks)} chunks."
)