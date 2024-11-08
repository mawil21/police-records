from docling.document_converter import DocumentConverter
import os
import fitz
from PIL import Image


# Function to extract the structure data from a document through docling 
def extract_document_structure(document):
    docling_doc = DocumentConverter().convert(document).document
    structured_data = []
    for item, _ in docling_doc.iterate_items():
        page_num, item_label, bbox = item.prov[0].page_no, item.label, item.prov[0].bbox
        structured_data.append((page_num, item_label, bbox))
    return structured_data 


# Function to process and save a single PDF
def process_pdf(pdf_file, input_dir, output_dir):
    # Compute the relative path
    relative_path = os.path.relpath(pdf_file, input_dir)
    print(f"Processing {relative_path}")
    extraction_dir_name = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(extraction_dir_name), exist_ok=True)

    #Process the PDF with docling
    structured_data = extract_document_structure(pdf_file)

    

# Function to process all PDFs in a given directory
def process_pdf_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        print(f"Processing files in {root}")
        for file in files:
            if file.endswith('.pdf'):
                full_pdf_path = os.path.join(root, file)
                print(f"PDF File Processing {full_pdf_path}")
                process_pdf(full_pdf_path, input_dir, output_dir)



if __name__ == "__main__":
    input_pdf_dir = "../tst/master_dataset"  # Directory containing the PDFs to process
    output_pdf_dir = "../tst/extraction_results" # Directory to save the extraction results of PDF documents
    process_pdf_directory(input_pdf_dir, output_pdf_dir)
