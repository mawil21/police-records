import fitz
from PIL import Image
import io
import cv2
import numpy as np 
import json 
import os
from docling.document_converter import DocumentConverter


class Docling_Structured_Extraction:
    def __init__(self):
        pass

    # Helper function to convert TableCell to a dictionary
    def table_cell_to_dict(self, cell):
        return {
            "bbox": {
                "l": cell.bbox.l if cell.bbox else None,
                "t": cell.bbox.t if cell.bbox else None,
                "r": cell.bbox.r if cell.bbox else None,
                "b": cell.bbox.b if cell.bbox else None,
                "coord_origin": cell.bbox.coord_origin.name if cell.bbox else None,
            },
            "row_span": cell.row_span,
            "col_span": cell.col_span,
            "start_row_offset_idx": cell.start_row_offset_idx,
            "end_row_offset_idx": cell.end_row_offset_idx,
            "start_col_offset_idx": cell.start_col_offset_idx,
            "end_col_offset_idx": cell.end_col_offset_idx,
            "text": cell.text,
            "column_header": cell.column_header,
            "row_header": cell.row_header,
            "row_section": cell.row_section,
        }

    # Function to convert the table data structure to JSON-compatible format
    def table_to_json(self, table):
        return {
            "page_no": table.prov[0].page_no,
            "bbox": {
                "l": table.prov[0].bbox.l,
                "t": table.prov[0].bbox.t,
                "r": table.prov[0].bbox.r,
                "b": table.prov[0].bbox.b,
                "coord_origin": table.prov[0].bbox.coord_origin.name,
            },
            "cells": [[self.table_cell_to_dict(cell) for cell in row] for row in table.data.grid],
            "num_rows": table.data.num_rows,
            "num_cols": table.data.num_cols,
        }


    # Function to extract tables from a document
    def extract_tables(self, pdf):
        doc_converter = DocumentConverter()
        conv_res = doc_converter.convert(pdf)
        return conv_res.document.tables if conv_res.document.tables else []

    # Function to extract key-value pairs from a document
    def extract_key_value_pairs(self, pdf):
        doc_converter = DocumentConverter()
        conv_res = doc_converter.convert(pdf)
        return conv_res.document.key_value_region if conv_res.document.key_value_region else []


    # Function to extract the structure data from a document through docling 
    def extract_document_structure(self, document):
        docling_doc = DocumentConverter().convert(document).document
        structured_data = []
        for item, _ in docling_doc.iterate_items():
            page_num, item_label, bbox = item.prov[0].page_no, item.label, item.prov[0].bbox
            structured_data.append((page_num, item_label, bbox))
        return structured_data 


    # Helper function to check if bounding box is valid
    def check_valid_bbox(self, bbox, page):
        # Ensure the bbox is within the page and has non-zero width/height
        rect = fitz.Rect(bbox)
        rect.intersect(page.rect)  # Clip bbox to the page rect
        return not rect.is_empty and rect.width > 0 and rect.height > 0


    # Function that uses structured data and returns cropped image
    def page_segment_retrieval(self, doc, structured_data, extraction_dir_name):
        images = {}
        with fitz.open(doc) as pdf:
            for structured_data in structured_data:
                page_number = structured_data[0] # Docling Page number is 1-indexed
                bbox = structured_data[2]
            
                page = pdf[page_number - 1]
                pix = page.get_pixmap()
                page_image = np.array(Image.open(io.BytesIO(pix.tobytes("ppm"))))[:,:,::-1]
                
                left, top, right, bottom = bbox.l, bbox.t, bbox.r, bbox.b
                
                page_height = page_image.shape[0]
            
                # Ensure the crop coordinates are within the image bounds
                top_cv = max(0, min(int(page_height - bottom), int(page_height - top)))
                bottom_cv = min(page_height, max(int(page_height - bottom), int(page_height - top)))
                
                left = max(0, int(left))
                right = min(page_image.shape[1], int(right))

                # Check for point-like bounding box (zero width or height)
                if left == right or top_cv == bottom_cv:
                    print(f"Skipping point-like bounding box for page {page_number}: ({left}, {top_cv}, {right}, {bottom_cv})")
                    continue  # Skip this bounding box if it represents a point
                # Check if the crop area is valid (non-zero width and height)
                if top_cv < bottom_cv and left < right:
                    # Crop the image using OpenCV
                    cropped_image = page_image[top_cv:bottom_cv, left:right]

                    # Add the cropped image to the dictionary if it is not empty
                    if cropped_image.size > 0:
                        if page_number not in images:
                            images[page_number] = [cropped_image]
                        else:
                            images[page_number].append(cropped_image)
                else:
                    print(f"Skipping empty or invalid crop box for page {page_number}: {(left, top_cv, right, bottom_cv)}")
            # Save each image by page number
        for page_num, images in images.items():
            # Create a subfolder for each page within the PDF's output directory
            page_dir = os.path.join(extraction_dir_name, f"page_{page_num}")
            os.makedirs(page_dir, exist_ok=True)
            
            # Save each image in the page's folder
            for i, image_data in enumerate(images):
                # Define a unique filename for each cropped image
                image_filename = f"structured_item_{i}.png"
                image_path = os.path.join(page_dir, image_filename)

                # Check if the image is non-empty before saving
                if image_data.size > 0:
                    cv2.imwrite(image_path, image_data)
                    print(f"Saved {image_filename} in {page_dir}")
                else:
                    print(f"Skipping empty image for page {page_num}, item {i}")
        return images 



    # Function to process and save a single PDF
    def process_pdf(self, pdf_file, input_dir, output_dir):
        # Compute the relative path without creating directories for each PDF
        relative_path = os.path.relpath(pdf_file, input_dir)
        json_file_name = os.path.splitext(relative_path)[0] + '.json'
        json_output_path = os.path.join(output_dir, json_file_name)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

        print(f"Processing {relative_path} and saving JSON to {json_output_path}")

        # Process the PDF with docling and save the JSON output
        tables = self.extract_tables(pdf_file)
        # Convert each table to JSON and collect them in a list
        all_tables_json = [self.table_to_json(table) for table in tables]
        with open(json_output_path, 'w') as file:
            json.dump(all_tables_json, file, indent=4)



    # Function to process all PDFs in a given directory
    def process_pdf_directory(self, input_dir, output_dir):
        for root, _, files in os.walk(input_dir):
            print(f"Processing files in {root}")
            for file in files:
                if file.endswith('.pdf'):
                    full_pdf_path = os.path.join(root, file)
                    print(f"PDF File Processing {full_pdf_path}")
                    self.process_pdf(full_pdf_path, input_dir, output_dir)

# GOAL is to extract the table and key-value pairs from the PDFs
if __name__ == "__main__":
    input_pdf_dir = "../../structured_extraction_baselines/azure_document_intelligence/tst/saved_pdfs"  # Directory containing the PDFs to process
    output_pdf_dir = "../../tst/docling_table_detection" # Directory to save the extraction results of PDF documents
    extractor = Docling_Structured_Extraction()
    extractor.process_pdf_directory(input_pdf_dir, output_pdf_dir)


