import os
import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Path to processed chunks
input_folder = "/home/shtlp_0042/Desktop/RAG/processed_data"

# List of all MiniLM models
minilm_models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v1",
]

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="/home/shtlp_0042/Desktop/RAG/chroma_db")

# Create a collection for each MiniLM model
collections = {
    model: chroma_client.get_or_create_collection(name=model.replace("/", "_"))
    for model in minilm_models
}

# Get all processed JSON files
json_files = [f for f in os.listdir(input_folder) if f.endswith("_documents.json")]

for file_name in json_files:
    file_path = os.path.join(input_folder, file_name)

    # Load processed disease chunks
    with open(file_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Embed and store using each MiniLM model
    for model_name in minilm_models:
        model = SentenceTransformer(model_name)
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            metadata = chunk["metadata"]
            text_content = chunk["content"]

            # Generate embedding
            embedding = model.encode(text_content).tolist()

            # Store in ChromaDB
            collections[model_name].add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[metadata]
            )

    print(f"Embedded and stored chunks from {file_name}")

print("All disease chunks embedded and stored successfully! ")
