import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb


# --------------------------------
# 1. Documentation URLs
# --------------------------------

URLS = [
    "https://learn.microsoft.com/en-us/azure/site-reliability-engineering/",
    "https://learn.microsoft.com/en-us/azure/azure-monitor/fundamentals/overview",
]


# --------------------------------
# 2. Load embedding model
# --------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# 3. Connect to ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="sre_docs"
)


# --------------------------------
# 4. Download web page
# --------------------------------

def fetch_page(url):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    return response.text


# --------------------------------
# 5. Extract sections
# --------------------------------

def extract_sections(html, url):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # Remove unnecessary elements
    for element in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header"
    ]):
        element.decompose()

    # Page title
    title = (
        soup.title.get_text(strip=True)
        if soup.title
        else url
    )

    # Find main documentation content
    main = soup.find("main")

    if not main:
        main = soup.body

    sections = []

    current_section = None
    current_content = []

    # Extract headings, paragraphs and lists
    for element in main.find_all([
        "h1",
        "h2",
        "h3",
        "p",
        "li"
    ]):

        text = element.get_text(
            " ",
            strip=True
        )

        if not text:
            continue

        # Heading means a new section
        if element.name in [
            "h1",
            "h2",
            "h3"
        ]:

            # Save previous section
            if current_content:

                sections.append({
                    "section": current_section,
                    "content": current_content
                })

            current_section = text
            current_content = []

        else:

            current_content.append(text)

    # Save final section
    if current_content:

        sections.append({
            "section": current_section,
            "content": current_content
        })

    return title, sections


# --------------------------------
# 6. Create chunks
# --------------------------------

def create_chunks(
    title,
    sections,
    url
):

    chunks = []

    # Maximum characters per chunk
    MAX_CHARS = 1200

    # Characters carried from previous chunk
    OVERLAP_CHARS = 200

    for section in sections:

        section_name = section["section"]

        content = section["content"]

        current_chunk = ""

        for paragraph in content:

            # Build proposed chunk
            if current_chunk:

                proposed_chunk = (
                    current_chunk
                    + "\n"
                    + paragraph
                )

            else:

                proposed_chunk = paragraph


            # If it fits, keep adding
            if len(proposed_chunk) <= MAX_CHARS:

                current_chunk = proposed_chunk

            else:

                # Store current chunk
                if current_chunk:

                    chunks.append({
                        "text": (
                            f"Title: {title}\n"
                            f"Section: {section_name}\n\n"
                            f"{current_chunk}"
                        ),

                        "metadata": {
                            "source": url,
                            "title": title,
                            "section": section_name
                        }
                    })


                # Create overlap
                overlap = current_chunk[
                    -OVERLAP_CHARS:
                ]

                current_chunk = (
                    overlap
                    + "\n"
                    + paragraph
                )


        # Store final chunk
        if current_chunk:

            chunks.append({
                "text": (
                    f"Title: {title}\n"
                    f"Section: {section_name}\n\n"
                    f"{current_chunk}"
                ),

                "metadata": {
                    "source": url,
                    "title": title,
                    "section": section_name
                }
            })

    return chunks


# --------------------------------
# 7. Process all URLs
# --------------------------------

all_chunks = []

metadata = []


for url in URLS:

    print(
        f"\nFetching: {url}"
    )

    try:

        # Download webpage
        html = fetch_page(url)

        # Extract sections
        title, sections = extract_sections(
            html,
            url
        )

        print(
            f"Extracted {len(sections)} sections"
        )

        # Create chunks
        chunks = create_chunks(
            title,
            sections,
            url
        )

        print(
            f"Created {len(chunks)} chunks"
        )

        # Add chunks to global list
        for chunk in chunks:

            all_chunks.append(
                chunk["text"]
            )

            metadata.append(
                chunk["metadata"]
            )

    except Exception as e:

        print(
            f"Failed to process {url}: {e}"
        )


# --------------------------------
# 8. Display chunks for debugging
# --------------------------------

print(
    f"\nTotal chunks: {len(all_chunks)}"
)


for i, chunk in enumerate(
    all_chunks[:10]
):

    print(
        f"\n--- Chunk {i} ---"
    )

    print(
        chunk[:1000]
    )


# --------------------------------
# 9. Generate embeddings
# --------------------------------

print(
    "\nGenerating embeddings..."
)

embeddings = embedding_model.encode(
    all_chunks
).tolist()


# --------------------------------
# 10. Create unique IDs
# --------------------------------

ids = [
    f"web_chunk_{i}"
    for i in range(
        len(all_chunks)
    )
]


# --------------------------------
# 11. Store in ChromaDB
# --------------------------------

print(
    "\nStoring chunks in ChromaDB..."
)

collection.upsert(
    ids=ids,
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=metadata
)


# --------------------------------
# 12. Finished
# --------------------------------

print(
    f"\nSuccessfully stored "
    f"{len(all_chunks)} chunks in ChromaDB."
)