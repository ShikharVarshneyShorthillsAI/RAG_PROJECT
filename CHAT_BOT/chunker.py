import json
import os
import uuid  # To generate unique chunk IDs

# Folder containing JSON files
input_folder = "/home/shtlp_0042/Desktop/RAG/scraped_data"
output_folder = "/home/shtlp_0042/Desktop/RAG/processed_data"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all JSON files in the scraped folder
json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

for file_name in json_files:
    file_path = os.path.join(input_folder, file_name)
    
    # Extract disease name from the file name (removing .json)
    disease_name = os.path.splitext(file_name)[0]

    # Load JSON file
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Process data into chunks
    documents = []
    for category, sub_dict in data.items():
        for key, value in sub_dict.items():
            chunk = {
                "chunk_id": str(uuid.uuid4()),  # Unique ID for each chunk
                "metadata": {
                    "category": category,
                    "sub_category": key,
                    "disease": disease_name
                },
                "content": value
            }
            documents.append(chunk)

    # Save processed chunks as a JSON file
    output_file = os.path.join(output_folder, f"{disease_name}_documents.json")
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(documents, out_file, indent=4, ensure_ascii=False)

    print(f"Processed and saved: {output_file}")
