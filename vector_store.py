# vector_store.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from ingest import process_documents

# ------------------------
# Initialize embedding model
# ------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector_store(folder="documents", index_path="vector.index", mapping_path="mapping.pkl"):
    # Process all documents into chunks
    chunks = process_documents(folder)
    print(f"Total chunks: {len(chunks)}")

    # Convert chunks to embeddings
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index + mapping
    faiss.write_index(index, index_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Vector store built and saved with {len(chunks)} chunks.")

def query_vector_store(query, index_path="vector.index", mapping_path="mapping.pkl", top_k=3):
    # Load index + mapping
    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        chunks = pickle.load(f)

    # Encode query
    query_vec = embedding_model.encode([query], convert_to_numpy=True)

    # Search
    distances, indices = index.search(query_vec, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# ------------------------
# Main Test
# ------------------------
if __name__ == "__main__":
    # Build the store
    build_vector_store("documents")

    # Test query
    results = query_vector_store("What is the termination clause?")
    print("\n🔎 Top Results:")
    for r in results:
        print("-", r[:200], "...\n")
