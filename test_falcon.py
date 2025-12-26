# from transformers import pipeline

# # Load the model (this will automatically download it)
# qa_model = pipeline("text-generation", model="tiiuae/falcon-rw-1b")

# # Test prompt
# prompt = "Summarize this contract clause: Either party may terminate the agreement upon 30 days notice."

# resp = qa_model(prompt, max_new_tokens=100)

# print(resp[0]['generated_text'])

import transformers
import torch

print("Transformers version:", transformers.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
