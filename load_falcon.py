from transformers import pipeline

# Load Falcon-1B model (will auto-download ~1.3 GB first time)
qa_model = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b"
)

# Test prompt
prompt = "Summarize this contract clause: Either party may terminate the agreement upon 30 days notice."

resp = qa_model(prompt, max_new_tokens=100)
print(resp[0]['generated_text'])
