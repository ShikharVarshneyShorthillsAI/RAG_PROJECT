import streamlit as st
import chromadb
import google.generativeai as genai
import os
import json
import datetime
from sentence_transformers import SentenceTransformer

# Configure Gemini API
genai.configure(api_key="YOUR_API_KEY")

# Load MiniLM model for query embedding
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="/home/shtlp_0042/Desktop/RAG/chroma_db")

# Load collection
collection = chroma_client.get_or_create_collection(name="sentence-transformers_all-MiniLM-L6-v2")

# Paths
chunks_folder = "/home/shtlp_0042/Desktop/RAG/processed_data"
log_file = "/home/shtlp_0042/Desktop/RAG/rag_log.json"
css_file = "/home/shtlp_0042/Desktop/RAG/style.css"

# Load external CSS
def load_css(file_name):
    with open(file_name, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Function to fetch content using chunk ID
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

# Function to log interactions
def log_interaction(user_query, retrieved_context, generated_answer):
    log_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": user_query,
        "retrieved_context": retrieved_context,
        "generated_answer": generated_answer,
    }

    # Load existing log file or create a new one
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
                if not isinstance(logs, list):
                    logs = []
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append(log_data)

    with open(log_file, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=4, ensure_ascii=False)

    # Also update session state for chat history
    st.session_state.chat_history.append({"query": user_query, "answer": generated_answer})

# Load previous chat history from log file on startup
def load_chat_history():
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
                if isinstance(logs, list):
                    return [{"query": log["query"], "answer": log["generated_answer"]} for log in logs]
            except json.JSONDecodeError:
                return []
    return []

# Load CSS for styling
load_css(css_file)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()  # Load from JSON

# Sidebar for Chat History
with st.sidebar:
    st.title("📜 Chat History")
    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            st.write(f"**Q:** {chat['query']}")
            st.write(f"**A:** {chat['answer']}")
            st.write("---")
    else:
        st.write("No chat history yet.")

    # Button to clear chat history (also clears the log file)
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        with open(log_file, "w", encoding="utf-8") as file:
            json.dump([], file)  # Reset log file
        st.rerun()

# Title and Description
st.title("🩺 Medical Q&A with RAG")
st.write("**Get accurate medical information using Retrieval-Augmented Generation (RAG).**")

# User Input Box
query = st.text_input("🔍 Enter your medical question:", placeholder="E.g., What are the symptoms of asthma?")

# Process Query
if st.button("🔎 Get Answer"):
    if query.strip():
        with st.spinner("Fetching the best answer..."):
            # Generate query embedding
            query_embedding = embedding_model.encode(query).tolist()

            # Retrieve relevant chunks
            results = collection.query(query_embeddings=[query_embedding], n_results=5)
            retrieved_chunk_ids = results.get("ids", [[]])[0]  # Ensure list format
            retrieved_contents = [get_chunk_content(chunk_id) for chunk_id in retrieved_chunk_ids if chunk_id]

            # Filter out None values
            retrieved_contents = [content for content in retrieved_contents if content]
            context = "\n".join(retrieved_contents) if retrieved_contents else "No relevant data found."

            # Generate response using Gemini 2.0
            if retrieved_contents:
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(f"Context: {context}\nQuestion: {query}\nAnswer:")
                answer = response.text if response else "Failed to generate an answer."
            else:
                answer = "No relevant data found in the context."

            # Log Interaction & Store Chat
            log_interaction(query, context, answer)

            # Display Answer
            st.subheader("📌 Answer:")
            st.markdown(f"**{answer}**", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter a valid medical question.")
