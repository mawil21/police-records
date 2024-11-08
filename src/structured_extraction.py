from docling.document_converter import DocumentConverter
import os
import fitz
from PIL import Image
import io
import cv2
import numpy as np 

# Function to extract the structure data from a document through docling 
def extract_document_structure(document):
    docling_doc = DocumentConverter().convert(document).document
    structured_data = []
    for item, _ in docling_doc.iterate_items():
        page_num, item_label, bbox = item.prov[0].page_no, item.label, item.prov[0].bbox
        structured_data.append((page_num, item_label, bbox))
    return structured_data 


# Helper function to check if bounding box is valid
def check_valid_bbox(bbox, page):
    # Ensure the bbox is within the page and has non-zero width/height
    rect = fitz.Rect(bbox)
    rect.intersect(page.rect)  # Clip bbox to the page rect
    return not rect.is_empty and rect.width > 0 and rect.height > 0

# Function that uses structured data and returns cropped image
def page_segment_retrieval(doc, structured_data):
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
    return images 

# Function to process and save a single PDF
def process_pdf(pdf_file, input_dir, output_dir):
    # Compute the relative path
    relative_path = os.path.relpath(pdf_file, input_dir)
    print(f"Processing {relative_path}")
    extraction_dir_name = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(extraction_dir_name), exist_ok=True)

    #Process the PDF with docling
    structured_data = extract_document_structure(pdf_file)
    images_by_page = page_segment_retrieval(pdf_file, structured_data)
    
    # Save each image by page number
    for page_num, images in images_by_page.items():
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
    output_pdf_dir = "../tst/extraction_results" # Directory to save the extraction results of PDF documents
    process_pdf_directory(input_pdf_dir, output_pdf_dir)
