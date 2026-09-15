from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Linux is an operating system.",
    "Linux is a Unix-like operating system used on servers.",
    "Chocolate cake is a popular dessert."
]

# Convert text into vectors
embeddings = model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Embedding dimension:", embeddings.shape[1])

# Compare sentence 1 with sentence 2
similarity_1_2 = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]

# Compare sentence 1 with sentence 3
similarity_1_3 = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]]
)[0][0]

print("\nSimilarity between sentence 1 and 2:", similarity_1_2)
print("Similarity between sentence 1 and 3:", similarity_1_3)
