from docling.document_converter import DocumentConverter
import os
import fitz
from PIL import Image
import io
import cv2
import numpy as np 
from models.docling_structured_extraction import Docling_Structured_Extraction
from models.document_intelligence_extraction import Azure_Document_Intelligence_Extraction

# Initialize extraction classes
docling_extractor = Docling_Structured_Extraction()
azure_extractor = Azure_Document_Intelligence_Extraction()




# Function to extract structured data from a PDF
def structued_extraction(pdf_file):
    tables = docling_extractor.extract_tables(pdf_file) # json output
    key_value_pairs = azure_extractor.extract_kv_pairs(pdf_file) # json output
    lines = azure_extractor.extract_lines(pdf_file) # json output
    return tables, key_value_pairs, lines

# Function to process and save a single PDF
def process_pdf(pdf_file, input_dir, output_dir):
    # Compute the relative path
    relative_path = os.path.relpath(pdf_file, input_dir)
    print(f"Processing {relative_path}")
    extraction_dir_name = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(extraction_dir_name), exist_ok=True)

    #Process the PDF with docling
    tables, kv_pairs, lines = structued_extraction(pdf_file)



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
