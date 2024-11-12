from docling.document_converter import DocumentConverter
import os
import fitz
from PIL import Image
import io
import cv2
import numpy as np 
from docling_pipeline.docling_structured_extraction import extract_tables, extract_key_value_pairs

# Function to process and save a single PDF
def process_pdf(pdf_file, input_dir, output_dir):
    # Compute the relative path
    relative_path = os.path.relpath(pdf_file, input_dir)
    print(f"Processing {relative_path}")
    extraction_dir_name = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(extraction_dir_name), exist_ok=True)

    #Process the PDF with docling
    tables = extract_tables(pdf_file) # json output 
    print(f"Extracted {len(tables)} tables from {relative_path}")
    key_value_pairs = extract_key_value_pairs(pdf_file) # json output 
    print(f"Extracted {len(key_value_pairs)} key-value pairs from {relative_path}")


# Function to process all PDFs in a given directory
def process_pdf_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        print(f"Processing files in {root}")
        for file in files:
            if file.endswith('.pdf'):
                full_pdf_path = os.path.join(root, file)
                print(f"PDF File Processing {full_pdf_path}")
                process_pdf(full_pdf_path, input_dir, output_dir)

# Main function to process the PDFs in the input directory
if __name__ == "__main__":
    input_pdf_dir = "../tst/master_dataset"  # Directory containing the PDFs to process
    output_pdf_dir = "../tst/pipeline_results" # Directory to save the extraction results of PDF documents
    process_pdf_directory(input_pdf_dir, output_pdf_dir)
