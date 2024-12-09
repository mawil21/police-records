import os
import json
import time
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from paddleocr import PPStructure

input_dir = "../../tst/saved_pdfs"
output_dir = "../../tst/form_detection_analysis/paddleOCR/output"
os.makedirs(output_dir, exist_ok=True)

MAX_PAGES = 10

# Enable KIE. Note: If you want KIE to work, ensure that you have the KIE models downloaded.
# Also, setting `layout=False` and `ocr=True` might conflict based on version. Try `layout=True` if needed.
# According to the warning, if layout is false, ocr is automatically set to false.
# To get key-value pairs (KIE), layout analysis is often needed, so set layout=True.
ocr = PPStructure(layout=True, kie=True, table=False, ocr=True, lang='en')

def process_page(image):
    # Convert PIL Image to NumPy array
    img_np = np.array(image)
    # Run PPStructure on the given image
    result = ocr(img_np)
    
    key_value_pairs = []
    # result should contain elements with 'type'='form' for key-value extraction
    for element in result:
        if element['type'] == 'form':
            # 'res' contains form fields
            form_fields = element.get('res', [])
            for field in form_fields:
                key_info = field.get('key', {})
                value_info = field.get('value', {})
                key_text = key_info.get('text', '').strip()
                value_text = value_info.get('text', '').strip()
                key_bbox = key_info.get('bbox', [])
                value_bbox = value_info.get('bbox', [])

                if key_text and value_text:
                    key_value_pairs.append({
                        "key_text": key_text,
                        "key_bbox": key_bbox,
                        "value_text": value_text,
                        "value_bbox": value_bbox
                    })

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
                
                page_image = page_image.convert("RGB")
                key_value_pairs = process_page(page_image)
                
                elapsed_time = time.time() - start_time
                
                pdf_output_data["pages"].append({
                    "page_number": page_index + 1,
                    "key_value_pairs": key_value_pairs,
                    "elapsed_time": elapsed_time
                })

            # Save JSON
            json_filename = f"{os.path.splitext(filename)[0]}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pdf_output_data, f, indent=4)

            print(f"Processed {pdf_path}. Results saved to {json_path}.")
