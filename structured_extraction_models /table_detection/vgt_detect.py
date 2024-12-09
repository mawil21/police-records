import os
import json
import subprocess


def vgt_for_heading_extraction(pdf_path, output_dir):
    # Call the Docker service using curl
    curl_command = f"curl -X POST -F 'file=@{pdf_path}' -F 'extraction_format=markdown' localhost:5060"

    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(output_dir, f"{base_filename}_dla.json")

    # Parse the output and save as JSON
    with open(out_path, 'w', encoding='utf-8') as file:
        json.dump(json.loads(result.stdout), file, indent=4)

    return out_path

input_dir = "../../tst/saved_pdfs"
output_dir = "../../tst/table_detection_analysis/vgt/output"
os.makedirs(output_dir, exist_ok=True)

MAX_PAGES = 10

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, filename)
            # Now run the heading extraction on the truncated PDF
            dla_path = vgt_for_heading_extraction(pdf_path, output_dir)
            print(f"Heading extraction results saved to {dla_path}")

           
