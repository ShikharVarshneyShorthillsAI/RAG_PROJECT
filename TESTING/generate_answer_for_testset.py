import os
import json
import time
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Configure Gemini API
genai.configure(api_key="AIzaSyCoe0a3mDH_EeKwZN-G8WAF4FX-e878p80")

# Load MiniLM model for query embedding
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="/home/shtlp_0042/Desktop/RAG/chroma_db")

# Load collection
collection = chroma_client.get_or_create_collection(name="sentence-transformers_all-MiniLM-L6-v2")

# Path to input test set and output folder
testset_folder = "/home/shtlp_0042/Desktop/RAG/generated_testset"  # Folder containing generated questions
output_folder = "/home/shtlp_0042/Desktop/RAG/generated_answers"  # Folder to store generated answers
chunks_folder = "/home/shtlp_0042/Desktop/RAG/processed_data"  # Folder containing processed chunks

os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

# Function to retrieve chunk content based on chunk ID
def get_chunk_content(chunk_id):
    for file_name in os.listdir(chunks_folder):
        if file_name.endswith("_documents.json"):
            file_path = os.path.join(chunks_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                chunks = json.load(file)
                for chunk in chunks:
                    if chunk["chunk_id"] == chunk_id:
                        return chunk["content"]
    return None

# Function to generate an answer using the RAG pipeline
def generate_answer_with_rag(question):
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(question).tolist()

        # Retrieve relevant chunks from ChromaDB
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        retrieved_chunk_ids = results.get("ids", [[]])[0]  # Ensure list format
        retrieved_contents = [get_chunk_content(chunk_id) for chunk_id in retrieved_chunk_ids if chunk_id]

        # Filter out None values
        retrieved_contents = [content for content in retrieved_contents if content]
        context = "\n".join(retrieved_contents) if retrieved_contents else "No relevant data found."

        # Generate response using Gemini 2.0
        if retrieved_contents:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(f"Context: {context}\nQuestion: {question}\nAnswer:")
            return response.text if response else "Failed to generate an answer."
        else:
            return "No relevant data found in the context."
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error in generating response."

# Get list of JSON files
json_files = [f for f in os.listdir(testset_folder) if f.endswith(".json")]

# Process files in batches
batch_size = 5  # Adjust based on API limits
for i in range(0, len(json_files), batch_size):
    batch = json_files[i : i + batch_size]

    for file_name in batch:
        input_file_path = os.path.join(testset_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        # Skip if the answer file already exists
        if os.path.exists(output_file_path):
            print(f"Skipping {file_name} (Already processed).")
            continue

        # Load question data
        with open(input_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        question = data.get("question", "")
        disease = data.get("disease", "")

        if not question or not disease:
            print(f"Skipping {file_name} due to missing data.")
            continue

        # Generate an answer using RAG
        answer = generate_answer_with_rag(question)

        # Store the answer in the same format
        data["answer"] = answer

        # Save the answer JSON
        with open(output_file_path, "w", encoding="utf-8") as out_file:
            json.dump(data, out_file, indent=4, ensure_ascii=False)

        print(f"Generated and saved answer for {file_name}")

    # Sleep between batches to avoid API rate limits
    print("Sleeping to avoid quota limits...")
    time.sleep(10)  # Adjust based on API constraints

print(f"All generated answers are saved in {output_folder}")
