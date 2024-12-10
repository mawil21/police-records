import os
import json
import time
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import torch
import easyocr

input_dir = "../../tst/saved_pdfs"
output_dir = "../../tst/phrase_extraction_analysis/easyocr/output"
os.makedirs(output_dir, exist_ok=True)

MAX_PAGES = 10

# Initialize EasyOCR reader for English with CPU-only
reader = easyocr.Reader(['en'], gpu=False)

def process_page(page_image):
    img_np = np.array(page_image)
    results = reader.readtext(img_np, detail=1)

    phrases = []
    for res in results:
        print(f"Detected text: {res}")
        box, text, confidence = res
        # Convert box coordinates and confidence to Python built-in types
        normalized_box = [[int(coord[0]), int(coord[1])] for coord in box]
        phrases.append({
            "text": str(text),
            "bbox": normalized_box,
            "confidence": float(confidence)
        })
    return phrases

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, filename)
            pages = convert_from_path(pdf_path, dpi=200)

            pdf_output_data = {
                "file": filename,
                "pages": []
            }

            for page_index, page_image in enumerate(pages[:MAX_PAGES]):
                start_time = time.time()

                page_image = page_image.convert("RGB")
                phrases = process_page(page_image)

                elapsed_time = time.time() - start_time

                pdf_output_data["pages"].append({
                    "page_number": page_index + 1,
                    "phrases": phrases,
                    "elapsed_time": elapsed_time
                })

            json_filename = f"{os.path.splitext(filename)[0]}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pdf_output_data, f, indent=4)

            print(f"Processed {pdf_path}. Results saved to {json_path}.")