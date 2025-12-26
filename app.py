import streamlit as st
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
# Embedding model
# ------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------
# Load Falcon-1B
# ------------------------
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="tiiuae/falcon-rw-1b")

qa_model = load_model()

# ------------------------
# Retrieval function
# ------------------------
def retrieve_relevant_clauses(query, top_k=5):
    query_vec = embedder.encode([query])
    distances, ids = index.search(query_vec, top_k)
    return [chunks[i] for i in ids[0]]

# ------------------------
# QA function
# ------------------------
def answer_legal_question(query, top_k=5):
    retrieved_clauses = retrieve_relevant_clauses(query, top_k)
    context = " ".join(retrieved_clauses)

    prompt = (
        f"You are a legal assistant. Use ONLY the context below to answer the question concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    result = qa_model(prompt, max_new_tokens=120)
    return result[0]['generated_text'].strip()

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("⚖️ AI Legal Assistant")

st.write("Ask questions about contracts and get concise answers from retrieved clauses.")

question = st.text_input("Enter your legal question:")

if question:
    with st.spinner("Analyzing contracts..."):
        answer = answer_legal_question(question)
    st.subheader("Answer:")
    st.write(answer)

