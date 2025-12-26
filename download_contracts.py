# download_contracts.py
from datasets import load_dataset
from pathlib import Path

# Load dataset from Hugging Face
print("Downloading dataset...")
dataset = load_dataset("kozue13/Contracts")

# Check what splits are available
print(dataset)

# Create local folder for documents
outdir = Path("documents")
outdir.mkdir(exist_ok=True)

# Save first N contracts as text files
N = 5000  # adjust based on how many you want
for i, item in enumerate(dataset["train"]):
    if i >= N:
        break
    text = item.get("text", "")
    if not text.strip():
        continue
    file_path = outdir / f"contract_{i}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

print(f"✅ Saved {min(N, len(dataset['train']))} contracts to {outdir}/")
