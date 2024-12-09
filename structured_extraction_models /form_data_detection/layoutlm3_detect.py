import os
import json
import time
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import math

input_dir = "../../tst/saved_pdfs"
output_dir = "../../tst/form_detection_analysis/layoutlmv3/output"
os.makedirs(output_dir, exist_ok=True)

MAX_PAGES = 10
MODEL_NAME = "nielsr/layoutlmv3-finetuned-funsd"  # A model fine-tuned on FUNSD

device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)  
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def run_ocr_and_get_words_boxes(image):
    # Run OCR with detailed layout info
    config = r'--psm 11 --oem 3'
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
    
    words = []
    boxes = []
    width, height = image.size
    
    for i, text in enumerate(data['text']):
        if text.strip():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            words.append(text)
            # Normalize bounding boxes to 0-1000 scale
            x1 = int((x / width) * 1000)
            y1 = int((y / height) * 1000)
            x2 = int(((x + w) / width) * 1000)
            y2 = int(((y + h) / height) * 1000)
            boxes.append([x1, y1, x2, y2])
    return words, boxes

def group_entities_with_boxes(words, boxes, labels):
    entities = []
    current_entity = {"type": None, "text": [], "boxes": []}
    for word, box, label in zip(words, boxes, labels):
        if label == 'O':
            if current_entity["type"]:
                x1 = min(b[0] for b in current_entity["boxes"])
                y1 = min(b[1] for b in current_entity["boxes"])
                x2 = max(b[2] for b in current_entity["boxes"])
                y2 = max(b[3] for b in current_entity["boxes"])
                entities.append({
                    "type": current_entity["type"],
                    "text": " ".join(current_entity["text"]),
                    "bbox": [x1, y1, x2, y2]
                })
                current_entity = {"type": None, "text": [], "boxes": []}
        else:
            label_type = label.split('-')[-1] if '-' in label else None
            if label.startswith('B-'):
                # start new entity
                if current_entity["type"]:
                    # finalize previous entity first
                    x1 = min(b[0] for b in current_entity["boxes"])
                    y1 = min(b[1] for b in current_entity["boxes"])
                    x2 = max(b[2] for b in current_entity["boxes"])
                    y2 = max(b[3] for b in current_entity["boxes"])
                    entities.append({
                        "type": current_entity["type"],
                        "text": " ".join(current_entity["text"]),
                        "bbox": [x1, y1, x2, y2]
                    })
                current_entity = {"type": label_type, "text": [word], "boxes": [box]}
            elif label.startswith('I-') and current_entity["type"] == label_type:
                current_entity["text"].append(word)
                current_entity["boxes"].append(box)

    if current_entity["type"]:
        x1 = min(b[0] for b in current_entity["boxes"])
        y1 = min(b[1] for b in current_entity["boxes"])
        x2 = max(b[2] for b in current_entity["boxes"])
        y2 = max(b[3] for b in current_entity["boxes"])
        entities.append({
            "type": current_entity["type"],
            "text": " ".join(current_entity["text"]),
            "bbox": [x1, y1, x2, y2]
        })
    return entities

def remap_labels(entities):
    # FUNSD labels: "QUESTION" ~ KEY, "ANSWER" ~ VALUE
    label_map = {
        "QUESTION": "KEY",
        "ANSWER": "VALUE"
    }
    for e in entities:
        if e["type"] in label_map:
            e["type"] = label_map[e["type"]]
    return entities

def center_of_bbox(bbox):
    # bbox: [x1, y1, x2, y2]
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return x_center, y_center

def pair_keys_and_values(entities):
    # Separate keys and values
    keys = [e for e in entities if e["type"] == "KEY"]
    values = [e for e in entities if e["type"] == "VALUE"]

    # Attempt to match each KEY to the closest VALUE
    key_value_pairs = []
    used_values = set()

    for key in keys:
        kx, ky = center_of_bbox(key["bbox"])
        # Find closest value
        best_val = None
        best_dist = float('inf')
        for i, val in enumerate(values):
            if i in used_values:
                continue
            vx, vy = center_of_bbox(val["bbox"])
            dist = math.sqrt((kx - vx)**2 + (ky - vy)**2)
            if dist < best_dist:
                best_dist = dist
                best_val = (i, val)
        if best_val is not None:
            used_values.add(best_val[0])
            key_value_pairs.append({
                "key_text": key["text"],
                "key_bbox": key["bbox"],
                "value_text": best_val[1]["text"],
                "value_bbox": best_val[1]["bbox"]
            })

    # Unmatched keys
    unmatched_keys = [k for k in keys if k not in [None] and k["text"] not in [p["key_text"] for p in key_value_pairs]]
    # Unmatched values
    unmatched_values = [v for i, v in enumerate(values) if i not in used_values]

    return key_value_pairs, unmatched_keys, unmatched_values

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
                words, boxes = run_ocr_and_get_words_boxes(page_image)

                if not words:
                    pdf_output_data["pages"].append({
                        "page_number": page_index+1,
                        "key_value_pairs": [],
                        "unmatched_keys": [],
                        "unmatched_values": [],
                        "elapsed_time": time.time() - start_time
                    })
                    continue

                encoding = processor(
                    text=words,
                    boxes=boxes,
                    images=page_image,
                    return_tensors="pt",
                    truncation=True
                )
                for k, v in encoding.items():
                    encoding[k] = v.to(device)

                with torch.no_grad():
                    outputs = model(**encoding)

                predictions = outputs.logits.argmax(-1).squeeze().tolist()
                labels = [model.config.id2label[id_] for id_ in predictions[:len(words)]]

                entities = group_entities_with_boxes(words, boxes, labels)
                entities = remap_labels(entities)  # Map QUESTION -> KEY, ANSWER -> VALUE

                key_value_pairs, unmatched_keys, unmatched_values = pair_keys_and_values(entities)

                elapsed_time = time.time() - start_time

                pdf_output_data["pages"].append({
                    "page_number": page_index + 1,
                    "key_value_pairs": key_value_pairs,
                    "unmatched_keys": unmatched_keys,
                    "unmatched_values": unmatched_values,
                    "elapsed_time": elapsed_time
                })

            json_filename = f"{os.path.splitext(filename)[0]}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pdf_output_data, f, indent=4)

            print(f"Processed {pdf_path}. Results saved to {json_path}.")
