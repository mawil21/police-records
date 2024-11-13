import fitz  # PyMuPDF
import os
from PIL import Image, ImageDraw
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import json 

class Azure_Document_Intelligence_Extraction:
    def __init__(self):
        pass
    
    # Initialize the Azure Document Analysis Client
    def initialize_azure_client(self, endpoint, key):
        return DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    def _format_polygon(self, polygon):
        if not polygon:
            return "N/A"
        return ", ".join([f"[{polygon[i]}, {polygon[i + 1]}]" for i in range(0, len(polygon), 2)])

    def _format_bounding_region(self, bounding_regions):
        if not bounding_regions:
            return "N/A"
        return ", ".join(
            f"Page #{region.page_number}: {self._format_polygon(region.polygon)}" for region in bounding_regions
        )

    def _format_bounding_box(self, bounding_regions):
        if not bounding_regions:
            return None
        polygon = bounding_regions[0].polygon
        x_coords = [point.x for point in polygon]
        y_coords = [point.y for point in polygon]
        return {
            "l": min(x_coords),
            "t": min(y_coords),
            "r": max(x_coords),
            "b": max(y_coords)
        }

    # Function to extract tables with customized bounding box format
    def extract_tables(self, result):
        tables = []
        if result.tables:
            for table in result.tables:
                table_data = {
                    "page_no": table.bounding_regions[0].page_number if table.bounding_regions else "N/A",
                    "bbox": self._format_bounding_box(table.bounding_regions),
                    "cells": []
                }
                for cell in table.cells:
                    cell_data = {
                        "bbox": self._format_bounding_box(cell.bounding_regions),
                        "text": cell.content,
                        
                    }
                    table_data["cells"].append(cell_data)
                tables.append(table_data)
        return tables

    def extract_kv_pairs(self, result):
        kv_pairs = []
        if result.key_value_pairs:
            for kv_pair in result.key_value_pairs:
                # Initialize page number variables
                key_page_number = None
                value_page_number = None
                
                # Extract key content, bounding box, and page number if available
                key_content = kv_pair.key.content if kv_pair.key else ""
                key_bounding_box = {}
                if kv_pair.key and kv_pair.key.bounding_regions:
                    polygon = kv_pair.key.bounding_regions[0].polygon
                    key_page_number = kv_pair.key.bounding_regions[0].page_number  # Extract page number from bounding regions
                    x_coords = [point.x for point in polygon]
                    y_coords = [point.y for point in polygon]
                    key_bounding_box = {
                        "left": min(x_coords),
                        "top": min(y_coords),
                        "right": max(x_coords),
                        "bottom": max(y_coords)
                    }

                # Extract value content, bounding box, and page number if available
                value_content = kv_pair.value.content if kv_pair.value else ""
                value_bounding_box = {}
                if kv_pair.value and kv_pair.value.bounding_regions:
                    polygon = kv_pair.value.bounding_regions[0].polygon
                    value_page_number = kv_pair.value.bounding_regions[0].page_number  # Extract page number from bounding regions
                    x_coords = [point.x for point in polygon]
                    y_coords = [point.y for point in polygon]
                    value_bounding_box = {
                        "left": min(x_coords),
                        "top": min(y_coords),
                        "right": max(x_coords),
                        "bottom": max(y_coords)
                    }
                    print(
                        f"Value '{value_content}' found within "
                        f"'{_format_bounding_region(kv_pair.value.bounding_regions)}' bounding regions on page {value_page_number}\n"
                    )

                # Append extracted key-value pair to the list, including page numbers
                kv_pairs.append({
                    "page_number": key_page_number or value_page_number,  # Use key page if available, otherwise value page
                    "key_bounding_box": key_bounding_box,
                    "value_bounding_box": value_bounding_box,
                    "key": key_content,
                    "value": value_content,
                })

        return kv_pairs




    # Function to draw bounding boxes on an image
    def annotate_page(self, page, layout_data):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)

        page_height = page.rect.height  # Get the height of the page in points

        # Draw bounding boxes for lines using 4-vertex quadrilaterals
        for line in layout_data.lines:
            if line.polygon and len(line.polygon) == 4:
                # Convert inches to points and adjust Y-coordinates to flip origin from bottom-left to top-left
                bbox = [(point.x * 72, (point.y * 72)) for point in line.polygon]
                draw.polygon(bbox, outline="blue", width=2)  # Draw box with blue outline

        # Draw bounding boxes for selection marks (if any) using quadrilateral order
        for mark in layout_data.selection_marks:
            if mark.polygon and len(mark.polygon) == 4:
                bbox = [(point.x * 72, (point.y * 72)) for point in mark.polygon]
                color = "green" if mark.state == "selected" else "red"
                draw.polygon(bbox, outline=color, width=2)  # Draw box with color based on state

        # Check if tables are present and annotate them
        if hasattr(layout_data, 'tables'):
            for table in layout_data.tables:
                for cell in table.cells:
                    if cell.polygon and len(cell.polygon) == 4:
                        bbox = [(point.x * 72, (point.y * 72)) for point in cell.polygon]
                        draw.polygon(bbox, outline="purple", width=2)  # Draw box with purple outline

        return img

    # Function to process and save a single PDF with annotations
    def process_pdf_with_annotations(self, pdf_file, output_dir, azure_client, page_limit=5):
        extracted_pdf_dir = os.path.join(output_dir, "saved_pdfs")
        annotated_pdf_dir = os.path.join(output_dir, "annotated_pdfs")
        
        os.makedirs(extracted_pdf_dir, exist_ok=True)
        os.makedirs(annotated_pdf_dir, exist_ok=True)
        
        output_pdf_file = os.path.join(extracted_pdf_dir, os.path.basename(pdf_file))
        with fitz.open(pdf_file) as pdf:
            pdf_extract = fitz.open()
            for page_num in range(0, min(page_limit, len(pdf))):
                pdf_extract.insert_pdf(pdf, from_page=page_num, to_page=page_num)
            pdf_extract.save(output_pdf_file)

        # Analyze PDF layout with Azure Document Intelligence
        with open(output_pdf_file, "rb") as f:
            poller = azure_client.begin_analyze_document("prebuilt-document", document=f)
            result = poller.result()

        # Annotate each page and save as a new PDF
        annotated_pdf_path = os.path.join(annotated_pdf_dir, f"annotated_{os.path.basename(pdf_file)}")
        with fitz.open(output_pdf_file) as pdf:
            images = []
            for page_num, page in enumerate(pdf.pages()):
                if page_num >= page_limit:
                    break
                layout_data = result.pages[page_num]
                annotated_img = self.annotate_page(page, layout_data)
                images.append(annotated_img)
            
            if images:
                images[0].save(
                    annotated_pdf_path, "PDF", save_all=True, append_images=images[1:]
                )

        print(f"Annotated PDF has been saved at {annotated_pdf_path}")



    # Function to process and save a single PDF
    def process_pdf(self, pdf_file, input_dir, output_dir, azure_client):
        # Compute the relative path without creating directories for each PDF
        relative_path = os.path.relpath(pdf_file, input_dir)
        json_file_name = os.path.splitext(relative_path)[0] + '.json'
        json_output_path = os.path.join(output_dir, json_file_name)
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

        print(f"Processing {relative_path} and saving JSON to {json_output_path}")

        
        # Analyze PDF layout with Azure Document Intelligence
        with open(pdf_file, "rb") as f:
            poller = azure_client.begin_analyze_document("prebuilt-document", document=f)
            result = poller.result()
        
        # Process the PDF with docling and save the JSON output
        kv_pairs = self.extract_tables(result)
        with open(json_output_path, 'w') as file:
            json.dump(kv_pairs, file, indent=4)



    # Function to process all PDFs in a given directory
    def process_pdf_directory(self, input_dir, output_dir, azure_client):
        for root, _, files in os.walk(input_dir):
            print(f"Processing files in {root}")
            for file in files:
                if file.endswith('.pdf'):
                    full_pdf_path = os.path.join(root, file)
                    print(f"PDF File Processing {full_pdf_path}")
                    self.process_pdf(full_pdf_path, input_dir, output_dir, azure_client)


# GOAL is to extract the table and key-value pairs from the PDFs
if __name__ == "__main__":
    input_pdf_dir = "../../structured_extraction_baselines/azure_document_intelligence/tst/saved_pdfs"  # Directory containing the PDFs to process
    output_pdf_dir = "../../tst/document_intelligence_tables" # Directory to save the extraction results of PDF documents
    
    # Initialize the Azure client
    azure_endpoint = "" # add your endpoint
    azure_key = "" # add your key
    extractor = Azure_Document_Intelligence_Extraction()
    azure_client = extractor.initialize_azure_client(azure_endpoint, azure_key)
    extractor.process_pdf_directory(input_pdf_dir, output_pdf_dir, azure_client)
