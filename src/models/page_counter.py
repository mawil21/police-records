import os
from PyPDF2 import PdfReader

def process_structured_extraction_directories(pdf_dir):
    total_pages = 0
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                try:
                    reader = PdfReader(pdf_path)
                    total_pages += len(reader.pages)
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")
    return total_pages

# Example usage:
if __name__ == "__main__":
    directory = "../tst/master_dataset"
    total_pages = process_structured_extraction_directories(directory)
    print(f"Total number of pages in all PDFs: {total_pages}")
