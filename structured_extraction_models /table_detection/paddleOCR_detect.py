import os
import json
import time
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PPStructure

input_dir = "../../tst/saved_pdfs"
output_dir = "../../tst/table_detection_analysis/paddleOCR/output"
os.makedirs(output_dir, exist_ok=True)

# Initialize PPStructure
table_engine = PPStructure(show_log=True, image_orientation=True)

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, filename)

            # Convert PDF pages to images at a higher DPI
            pages = convert_from_path(pdf_path, dpi=200)

            pdf_output_data = {
                "file": filename,
                "pages": []
            }

            # Process only the first 10 pages
            for page_index, page_image in enumerate(pages[:10]):
                page_image_np = np.array(page_image)

                # Record start time
                start_time = time.time()
                results = table_engine(page_image_np)
                # Record end time
                end_time = time.time()
                elapsed_time = end_time - start_time

                tables_output = []
                for res in results:
                    if res['type'] == 'table':
                        html_content = res['res']['html']
                        cell_boxes = res['res'].get('cell_boxes', [])

                        table_data = {
                            "bbox": res.get('bbox', []),
                            "raw-text": html_content
                        }

                        tables_output.append(table_data)

                pdf_output_data["pages"].append({
                    "page_number": page_index + 1,
                    "tables": tables_output,
                    "elapsed_time": elapsed_time
                })

            json_filename = f"{os.path.splitext(filename)[0]}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pdf_output_data, f, indent=4)
