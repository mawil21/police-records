import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def flatten_json(data, parent_key='', sep='.'):
    """Recursively flattens a nested JSON object using dot notation."""
    items = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep=sep))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}[{i}]"
            items.update(flatten_json(v, new_key, sep=sep))
    else:
        items[parent_key] = data
    return items

def extract_all_gpt_4o(data):
    """
    Recursively searches the JSON data and returns a list of all values 
    found under the key 'gpt_4o_response'.
    """
    responses = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k == "gpt_4o_response":
                responses.append(v)
            else:
                responses.extend(extract_all_gpt_4o(v))
    elif isinstance(data, list):
        for item in data:
            responses.extend(extract_all_gpt_4o(item))
    return responses

def flatten_json_template(full_gpt_response_path, model_name="gpt_4o"):
    """
    Loads the JSON from the given file, extracts all 'gpt_4o_response' objects,
    flattens each one, and returns a list of flattened dictionaries.
    The model_name parameter is kept for compatibility.
    """
    with open(full_gpt_response_path, "r") as infile:
        data = json.load(infile)
    
    # Extract all objects under "gpt_4o_response"
    responses = extract_all_gpt_4o(data)
    
    # Flatten each extracted response
    flattened_responses = [flatten_json(response) for response in responses]
    return flattened_responses

###################################
# Multi-File Processing Functions
###################################

def process_single_file(sub_dir, file, input_extraction_dir, output_flattened_dir, model_name="gpt_4o"):
    """
    Process a single file.
    Build the full path to the input file, extract and flatten all "gpt_4o_response" objects,
    and save the flattened output as a JSON file.
    """
    full_gpt_response_path = os.path.join(input_extraction_dir, sub_dir, file)
   
    combined_data = flatten_json_template(full_gpt_response_path, model_name)
    
    base_filename = os.path.splitext(file)[0]
    output_subdir = os.path.join(output_flattened_dir, sub_dir, base_filename)
    os.makedirs(output_subdir, exist_ok=True)
    
    output_path = os.path.join(output_subdir, "flattened_structured_data.json")
    with open(output_path, "w") as outfile:
        json.dump(combined_data, outfile, indent=4)
    
    print(f"Processed file. Output saved to {output_path}")

def process_structured_extraction_directories(input_extraction_dir, output_flattened_dir, model_name="gpt_4o"):
    """
    Walk through the input extraction directory and process each JSON file concurrently.
    """
    tasks = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for root, _, files in os.walk(input_extraction_dir):
            sub_dir = os.path.relpath(root, input_extraction_dir)
            for file in files:
                if file.endswith('.json'):
                    tasks.append(
                        executor.submit(
                            process_single_file, sub_dir, file,
                            input_extraction_dir, output_flattened_dir,
                            model_name
                        )
                    )
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print("Error processing file:", e)

###################################
# Main Entry Point
###################################

if __name__ == "__main__":
    input_extraction_dir = "../tst/structured_extraction_template_gpt_4o"
    output_flattened_dir = "../tst/structured_extraction_flattened_gpt_4o"
    chosen_model_name = "gpt_4o"
    
    process_structured_extraction_directories(input_extraction_dir,
                                              output_flattened_dir,
                                              model_name=chosen_model_name)

