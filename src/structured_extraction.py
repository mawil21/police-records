from docling.document_converter import DocumentConverter
import os
import fitz
from PIL import Image
import io
import cv2
import numpy as np 
from models.docling_structured_extraction import Docling_Structured_Extraction
from models.document_intelligence_extraction import Azure_Document_Intelligence_Extraction
import json 
# Initialize extraction classes
docling_extractor = Docling_Structured_Extraction()
azure_extractor = Azure_Document_Intelligence_Extraction()


############################################################################################################
# Helper Functions to Refactor JSON Data
############################################################################################################

def refactor_tables_json(json_data):
        refactored_data = {}
        for item in json_data:
            page_no = item.get('page_no')
            if page_no not in refactored_data:
                refactored_data[page_no] = []

            bbox = item.get('bbox', {})
            point_x = bbox.get('point.x', [])
            point_y = bbox.get('point.y', [])

            # Pair every x in point.x with every y in point.y
            coordinates = list(zip(point_x, point_y))


            refactored_data[page_no].append({
                'bbox': coordinates,  # Assign the paired coordinates to bbox
                'content' : item.get('cells', [])
            })

        return refactored_data

def refactor_key_value_pairs(json_data):
    refactored_data = {}
    
    for item in json_data:
        page_no = item.get('page_number')
        if page_no not in refactored_data:
            refactored_data[page_no] = []

        # Helper function to pair coordinates
        def pair_coordinates(bbox):
            point_x = bbox.get('point.x', [])
            point_y = bbox.get('point.y', [])
            return list(zip(point_x, point_y))

        # Process key and value bounding boxes
        key_bbox_coordinates = pair_coordinates(item.get('key_bounding_box', {}))
        value_bbox_coordinates = pair_coordinates(item.get('value_bounding_box', {}))
        print(f"Key Bounding Box Coordinates: {key_bbox_coordinates}")
        print(f"Value Bounding Box Coordinates: {value_bbox_coordinates}")
        x1y1, x2y2, x3y3, x4y4 = None, None, None, None
        if not value_bbox_coordinates:
            x1y1 = key_bbox_coordinates[0]
            x2y2 = key_bbox_coordinates[1]
            x3y3 = key_bbox_coordinates[2]
            x4y4 = key_bbox_coordinates[3]
        else:
            x1y1 = key_bbox_coordinates[0]
            x2y2 = value_bbox_coordinates[1]
            x3y3 = value_bbox_coordinates[2]
            x4y4 = key_bbox_coordinates[3]

        # Combine the key and value content
        key_content = item.get('key', [])
        value_content = item.get('value', [])
        content = None 
        if not value_content:
            content = key_content
        else:
            content = f"{key_content} {value_content}"

        # Append the refactored key-value pair data
        refactored_data[page_no].append({
            'bbox' : [x1y1, x2y2, x3y3, x4y4],
            'content' : content
        })

    return refactored_data

        
        
# Helper function to refactor the lines json -- delete all pages that don't have tables or kv pairs identified
def refactor_lines_json(json_data, table_pages, kv_pages):
    refactored_data = {}
    
    for item in json_data:
        page_no = item.get('page_no') or item.get('page_number')
        
        if page_no in table_pages or page_no in kv_pages:
            if page_no not in refactored_data:
                refactored_data[page_no] = []
            
            contents = item.get('content', [])
            print(f"Contents: {contents}")
            if not contents: 
                raise ValueError("No content found in the lines json")
            for content in contents:
                bbox = content.get('bbox', {})
                point_x = bbox.get('point.x', [])
                point_y = bbox.get('point.y', [])

                # Pair every x in point.x with every y in point.y
                coordinates = list(zip(point_x, point_y))

                text = content.get('text', "")
                
                if not text:
                    text = ""
                            
                refactored_data[page_no].append({
                    'bbox': coordinates,  # Assign the paired coordinates to bbox
                    'content' : text
                })
    return refactored_data


def format_extraction(table_json_path, kv_json_path, lines_json_path, output_extraction_dir):
    
    with open(table_json_path, 'r') as file:
        tables = refactor_tables_json(json.load(file))
    with open(kv_json_path, 'r') as file:
        kv_pairs = refactor_key_value_pairs(json.load(file))
    with open(lines_json_path, 'r') as file:
        lines = refactor_lines_json(json.load(file), tables, kv_pairs)
    
    # Dump the refactored data to the output directory
    with open(os.path.join(output_extraction_dir, "cleaned_table.json"), 'w') as file:
        json.dump(tables, file, indent=4)
    with open(os.path.join(output_extraction_dir, "cleaned_kv_pairs.json"), 'w') as file:
        json.dump(kv_pairs, file, indent=4)
    with open(os.path.join(output_extraction_dir, "cleaned_lines.json"), 'w') as file:
        json.dump(lines, file, indent=4)

   


    return tables, kv_pairs, lines


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
    # input_pdf_dir = "../tst/master_dataset"  # Directory containing the PDFs to process
    # output_pdf_dir = "../tst/pipeline_results" # Directory to save the extraction results of PDF documents
    # process_pdf_directory(input_pdf_dir, output_pdf_dir)



    ############################################################################################################
    # Example JSON Path 
    ############################################################################################################

    table_json_path = "../tst/document_intelligence_tables/1715882152079-fun/1715882152079-fun-1.json"
    kv_json_path = "../tst/document_intelligence_kv_pairs/1715882152079-fun/1715882152079-fun-1.json"
    lines_json_path = "../tst/document_intelligence_lines/1715882152079-fun/1715882152079-fun-1.json"
    output_extraction_dir = "../tst/pipeline_results"
    table, kv_pairs, lines = format_extraction(table_json_path, kv_json_path, lines_json_path, output_extraction_dir)
    
   



    
