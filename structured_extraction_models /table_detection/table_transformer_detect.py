import os
import json
import time
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
import pytesseract
import base64

input_dir = "../../tst/saved_pdfs"
output_dir = "../../tst/table_detection_analysis/table_transformer/output"
os.makedirs(output_dir, exist_ok=True)

MAX_PAGES = 10
MODEL_NAME = "microsoft/table-transformer-detection"

# Load the Table Transformer model and feature extractor
feature_extractor = DetrFeatureExtractor.from_pretrained(MODEL_NAME)
model = TableTransformerForObjectDetection.from_pretrained(MODEL_NAME)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

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
                page_image = page_image.convert("RGB")
                page_width, page_height = page_image.size

                start_time = time.time()

                # Preprocess image for model
                inputs = feature_extractor(page_image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)

                threshold = 0.7
                processed_outputs = feature_extractor.post_process_object_detection(
                    outputs, target_sizes=[(page_height, page_width)]
                )[0]

                id2label = model.config.id2label
                boxes_output = []
                for score, label, box in zip(
                    processed_outputs["scores"],
                    processed_outputs["labels"],
                    processed_outputs["boxes"]
                ):
                    if score > threshold and id2label[label.item()] == "table":
                        bbox = box.tolist()  # [y_min, x_min, y_max, x_max]

                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = bbox[1], bbox[0], bbox[3], bbox[2]

                        # Crop the table region from the page image
                        table_region = page_image.crop((x1, y1, x2, y2))

                        # Run OCR on the cropped table image
                        table_text = pytesseract.image_to_string(table_region)

                        # Optionally encode this text in Base64 or just store as plain text
                        # For Base64:
                        encoded_text = table_text

                        boxes_output.append({
                            "bbox": [x1, y1, x2, y2],
                            "score": score.item(),
                            "label": id2label[label.item()],
                            "encoded_text": encoded_text
                        })

                end_time = time.time()
                elapsed_time = end_time - start_time

                pdf_output_data["pages"].append({
                    "page_number": page_index + 1,
                    "tables": boxes_output,
                    "elapsed_time": elapsed_time
                })

            # Save JSON output for this PDF
            json_filename = f"{os.path.splitext(filename)[0]}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pdf_output_data, f, indent=4)

            print(f"Processed {pdf_path}. Results saved to {json_path}.")
