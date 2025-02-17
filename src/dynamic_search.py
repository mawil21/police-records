import os
import json
import csv
import numpy as np
import faiss
from openai import OpenAI
import io
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed


client = OpenAI()

def chatGPT_api(message_content):
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={'type': 'json_object'},
        messages=[
            {"role": "user", "content": "Provide output in valid JSON."},
            {"role": "user", "content": message_content}
        ]
    )
    raw_response = response.choices[0].message.content
    try:
        structured_response = json.loads(raw_response)
        return structured_response
    except Exception as e:
        return {"error": "Failed to parse JSON", "raw_response": raw_response}

def gpt_4o(prompt):
    return chatGPT_api(prompt)

# --- Helper Functions ---

def load_flattened_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def dict_to_text(flat_dict):
    return "\n".join([f"{k}: {v}" for k, v in flat_dict.items() if v is not None])

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)

def compute_embeddings(documents, model="text-embedding-ada-002"):
    embeddings = [get_embedding(doc, model=model) for doc in documents]
    return np.stack(embeddings)

def retrieve_documents_from_text(documents, query, k=5):
    embeddings = compute_embeddings(documents)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    query_embedding = get_embedding(query)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, k)
    retrieved = [documents[i] for i in indices[0]]
    return retrieved

# --- Process a Single File to Extract a Record ---
def process_single_file_extract_record(filepath, base_dir, desired_fields=["Case Number", "Officer Names", "Incident Dates"]):
    print(f"Processing file: {filepath}")
    try:
        data = load_flattened_json(filepath)
        documents = [dict_to_text(doc) for doc in data]
        query = "Extract records that contain information about case numbers, officer names, and incident dates."
        retrieved_docs = retrieve_documents_from_text(documents, query, k=5)
        context = "\n\n".join(retrieved_docs)
        prompt = (
            f"Below are several records extracted from police documents:\n\n{context}\n\n"
            f"Using fuzzy matching and context, extract the following fields: {', '.join(desired_fields)}. "
            "If multiple values exist for a field, list them separated by semicolons. "
            "Return the results as JSON, for example: "
            '{"Case Number": "value", "Officer Names": "value", "Incident Dates": "value"}'
        )
        response = gpt_4o(prompt)
        record = response  # Expecting a dict

        # Compute the relative path and extract the parent folder (third-to-last element)
        relative_path = os.path.relpath(filepath, base_dir)
        parts = os.path.normpath(relative_path).split(os.sep)
        if len(parts) >= 3:
            file_id = parts[-3]
        else:
            file_id = relative_path
        record["File Name"] = file_id
        record["Error"] = ""
        print(f"Finished processing file: {filepath}")
        return record
    except Exception as e:
        err_trace = traceback.format_exc()
        print(f"Error processing file {filepath}:\n{err_trace}")
        relative_path = os.path.relpath(filepath, base_dir)
        parts = os.path.normpath(relative_path).split(os.sep)
        if len(parts) >= 3:
            file_id = parts[-3]
        else:
            file_id = relative_path
        return {
            "File Name": file_id,
            "Case Number": "",
            "Officer Names": "",
            "Incident Dates": "",
            "Error": err_trace
        }

# --- Process Multiple Files Concurrently and Combine into One CSV ---
def process_structured_extraction_directories(input_dir, desired_fields=["Case Number", "Officer Names", "Incident Dates"]):
    filepaths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                filepaths.append(os.path.join(root, file))
    print(f"Processing {len(filepaths)} files.")
    
    records = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_single_file_extract_record, fp, input_dir, desired_fields) for fp in filepaths]
        for future in as_completed(futures):
            try:
                rec = future.result()
                records.append(rec)
                print("File processed successfully.")
            except Exception as e:
                print("Error processing a file:", e)
    return records

def write_records_to_csv(records, output_csv_path, desired_fields=["File Name", "Case Number", "Officer Names", "Incident Dates", "Error"]):
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=desired_fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

# --- Main Entry Point ---
if __name__ == "__main__":
    input_dir = "../tst/structured_extraction_flattened_gpt_4o"  # Directory with flattened JSON files
    output_csv_path = "../tst/processed_files_csv/police_records_rag_results.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    desired_fields = ["Case Number", "Officer Names", "Incident Dates"]
    csv_fields = ["File Name", "Case Number", "Officer Names", "Incident Dates", "Error"]
    
    records = process_structured_extraction_directories(input_dir, desired_fields)
    write_records_to_csv(records, output_csv_path, csv_fields)
    print(f"CSV file saved as {output_csv_path}")
