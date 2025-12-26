import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------------
# Load FAISS index + chunks
# ------------------------
index_path = "vector.index"
mapping_path = "mapping.pkl"

index = faiss.read_index(index_path)
with open(mapping_path, "rb") as f:
    chunks = pickle.load(f)

# ------------------------
# Embedding model (same as used for FAISS)
# ------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------
# Load Falcon-1B
# ------------------------
qa_model = pipeline("text-generation", model="tiiuae/falcon-rw-1b")

# ------------------------
# FAISS search function
# ------------------------
def retrieve_relevant_clauses(query, top_k=5):
    query_vec = embedder.encode([query])
    distances, ids = index.search(query_vec, top_k)
    return [chunks[i] for i in ids[0]]

# ------------------------
# RAG function: retrieve + generate
# ------------------------
# def answer_legal_question(query, top_k=5):
#     retrieved_clauses = retrieve_relevant_clauses(query, top_k)
#     context = "\n\n".join([f"Clause: {c}" for c in retrieved_clauses])

#     prompt = (
#         f"You are a legal assistant. Use ONLY the context below to answer the question accurately.\n\n"
#         f"Context:\n{context}\n\n"
#         f"Question: {query}\nAnswer concisely:"
#     )

#     result = qa_model(prompt, max_new_tokens=150)
#     return result[0]['generated_text']

def answer_legal_question(query, top_k=5):
    retrieved_clauses = retrieve_relevant_clauses(query, top_k)
    context = " ".join(retrieved_clauses)

    prompt = (
        f"You are a legal assistant. Use ONLY the context below to answer the question concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    result = qa_model(prompt, max_new_tokens=120)
    answer = result[0]['generated_text'].strip()
    return answer

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    question = "Under what conditions can this agreement be terminated?"
    answer = answer_legal_question(question)
    print("Q:", question)
    print("A:", answer)
