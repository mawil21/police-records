import os
import json
import time
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, DonutProcessor

input_dir = "../../tst/saved_pdfs"
output_dir = "../../tst/form_detection_analysis/donut/output"
os.makedirs(output_dir, exist_ok=True)

MAX_PAGES = 10

# Choose a Donut model fine-tuned on forms (e.g., FUNSD)
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"

device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def extract_key_values_from_result(result_json):
    # The FUNSD-finetuned Donut model typically outputs a JSON structure like:
    # {
    #   "form": [
    #       {"key": "Name", "value": "John Doe", "bbox_key": [...], "bbox_value": [...]},
    #       ...
    #   ]
    # }
    # The exact structure depends on the model. Adjust parsing accordingly.
    
    # If the model returns a `form` field with a list of key-value pairs:
    key_value_pairs = []
    if "form" in result_json:
        for field in result_json["form"]:
            key_text = field.get("key", "")
            value_text = field.get("value", "")
            # bounding boxes might be optional depending on the model
            key_bbox = field.get("bbox_key", [])
            value_bbox = field.get("bbox_value", [])
            key_value_pairs.append({
                "key_text": key_text.strip(),
                "key_bbox": key_bbox,
                "value_text": value_text.strip(),
                "value_bbox": value_bbox
            })
    return key_value_pairs

def process_page(page_image):
    # Convert image to RGB (if not already)
    page_image = page_image.convert("RGB")
    
    # Preprocess image for Donut
    pixel_values = processor(images=page_image, return_tensors="pt").pixel_values.to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(pixel_values, max_length=1024)
    sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Attempt to parse sequence as JSON
    try:
        result_json = json.loads(sequence)
    except json.JSONDecodeError:
        # If model doesn't return perfect JSON, store raw sequence
        result_json = {"raw_result": sequence}

    # Extract key-value pairs
    key_value_pairs = extract_key_values_from_result(result_json)
    return key_value_pairs

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
                kv_pairs = process_page(page_image)
                elapsed_time = time.time() - start_time

                pdf_output_data["pages"].append({
                    "page_number": page_index + 1,
                    "key_value_pairs": kv_pairs,
                    "elapsed_time": elapsed_time
                })

            json_filename = f"{os.path.splitext(filename)[0]}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pdf_output_data, f, indent=4)

            print(f"Processed {pdf_path}. Results saved to {json_path}.")
