import os
import json
import time
import google.generativeai as genai
from tqdm import tqdm
from evaluate import load
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
import torch
import re

# Configure Gemini API
genai.configure(api_key="AIzaSyCoe0a3mDH_EeKwZN-G8WAF4FX-e878p80")

# Paths
test_set_folder = "/home/shtlp_0042/Desktop/RAG/generated_testset"
generated_folder = "/home/shtlp_0042/Desktop/RAG/generated_answers"
output_json_path = "/home/shtlp_0042/Desktop/RAG/evaluation_results.json"

# Load metrics
rouge_metric = load("rouge")
bleu_metric = load("bleu")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Store results
results = []

# List all files
test_files = os.listdir(test_set_folder)

# Process files with tqdm
for filename in tqdm(test_files, desc="Processing Questions"):
    test_file_path = os.path.join(test_set_folder, filename)
    generated_file_path = os.path.join(generated_folder, filename)

    if not os.path.exists(generated_file_path):
        print(f"Skipping {filename}: No generated answer found.")
        continue

    # Load JSON files
    with open(test_file_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(generated_file_path, "r", encoding="utf-8") as f:
        generated_data = json.load(f)

    # Extract data
    question = test_data.get("question", "").strip()
    disease = test_data.get("disease", "").strip()
    reference_answer = test_data.get("answer", "").strip()
    generated_answer = generated_data.get("answer", "").strip()

    # Skip if response contains an API error
    if "Error in generating response" in generated_answer or not generated_answer:
        print(f"Skipping {filename}: API error in generated response.")
        continue

    # Compute BERTScore
    P, R, F1 = bert_score([generated_answer], [reference_answer], lang="en", device=device)
    bert_score_f1 = F1.mean().item()

    # Compute ROUGE & BLEU
    rouge_score = rouge_metric.compute(predictions=[generated_answer], references=[reference_answer])["rougeL"]
    bleu_score = bleu_metric.compute(predictions=[generated_answer], references=[[reference_answer]])["bleu"]

    # Compute METEOR
    meteor = meteor_score([reference_answer.split()], generated_answer.split())

    # Compute LLM faithfulness, relevance, coherence
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Evaluate the following generated answer against the reference for faithfulness, relevance, and coherence.

    Question: {question}

    Reference Answer: {reference_answer}

    Generated Answer: {generated_answer}

    Score each from 0 (worst) to 1 (best).

    Provide output in JSON format:
    {{
      "faithfulness": <score>,
      "relevance": <score>,
      "coherence": <score>
    }}
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove leading ```json
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove trailing ```
        print(response_text)
        eval_scores = json.loads(response_text)
        
        faithfulness = eval_scores.get("faithfulness", 0)
        relevance = eval_scores.get("relevance", 0)
        coherence = eval_scores.get("coherence", 0)

    except Exception as e:
        print(f"Error in LLM evaluation for {filename}: {e}")
        faithfulness, relevance, coherence = 0, 0, 0

    # Store result
    results.append({
        "question": question,
        "disease": disease,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,
        "meteor_score": meteor,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "coherence": coherence
    })

    # Append to JSON file (one by one to avoid memory issues)
    with open(output_json_path, "a", encoding="utf-8") as f:
        json.dump(results[-1], f, indent=4)
        f.write(",\n")

    # Sleep to avoid quota limit
    time.sleep(3)

print(f"\nProcessing complete! Results saved to {output_json_path}")
