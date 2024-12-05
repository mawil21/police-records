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
from shapely.geometry import Polygon

# Initialize extraction classes
docling_extractor = Docling_Structured_Extraction()
azure_extractor = Azure_Document_Intelligence_Extraction()

############################################################################################################
# Helper Functions to For Boundary Box Cases
############################################################################################################
def bbox_equivalence(bbox1, bbox2):
    if bbox1 == bbox2:
        return False
    return bbox1 == bbox2


def find_lines_above_region(bbox1, bbox2):
    """
    Determine if bbox1 is higher than bbox2 on the y-axis.

    Parameters:
        bbox1: Dictionary containing the 'point.y' of the first bounding box.
        bbox2: Dictionary containing the 'point.y' of the second bounding box.

    Returns:
        bool: True if bbox1 is higher than bbox2, False otherwise.
    """
    # Extract y-values from the bounding boxes
    # bbox1_y_values = [point[1] for point in bbox1]
    # bbox2_y_values = [point[1] for point in bbox2]

    # # Find the minimum y-values (top-most points) for both bounding boxes
    # bbox1_top = min(bbox1_y_values)
    # bbox2_top = min(bbox2_y_values)

    # # bbox1 is higher if its top-most y-value is smaller
    # return bbox1_top < bbox2_top
    x0, y0 = bbox1[0]
    x1, y1 = bbox1[1]
    x2, y2 = bbox1[2]
    x3, y3 = bbox1[3]

    x4, y4 = bbox2[0]
    x5, y5 = bbox2[1]
    x6, y6 = bbox2[2]
    x7, y7 = bbox2[3]

    

    lowest_bbox_1_y_axis = max(y0, y1)
    highest_bbox_2_y_axis = max(y6, y7)
    return lowest_bbox_1_y_axis < highest_bbox_2_y_axis


def check_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.

    Parameters:
        bbox1 (list of tuples): Coordinates of the first bounding box.
        bbox2 (list of tuples): Coordinates of the second bounding box.

    Returns:
        bool: True if the bounding boxes overlap, False otherwise.
    """
    # Create polygons from bounding boxes
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)

    # Check if they overlap
    return poly1.intersects(poly2)


############################################################################################################
# Helper Functions to Refactor JSON Data
############################################################################################################

def refactor_tables_json(json_data):
        refactored_data = {}
        for item in json_data:
            print(item)
            page_no = item.get('page_no')
            if page_no not in refactored_data:
                refactored_data[page_no] = []

            bbox = item.get('table_bounding_region', {})
            point_x = bbox.get('point.x', [])
            point_y = bbox.get('point.y', [])

            # Pair every x in point.x with every y in point.y
            coordinates = list(zip(point_x, point_y))

            refactored_data[page_no].append({
                'type': 'table',
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
            'type' : 'kv_pair',
            'bbox' : [x1y1, x2y2, x3y3, x4y4],
            'content' : content
        })

    return refactored_data

                
# Helper function to refactor the lines json --
# Delete all pages that don't have tables or kv pairs identified
def refactor_lines_json(json_data, table_pages, kv_pages):
    refactored_data = {}
    
    for item in json_data:
        page_no = item.get('page_no') or item.get('page_number')
        
        if page_no in table_pages or page_no in kv_pages:
            if page_no not in refactored_data:
                refactored_data[page_no] = []
            
            contents = item.get('content', [])
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


def refactor_and_persist_json_data(table_json_path, kv_json_path, lines_json_path):
    with open(table_json_path, 'r') as file:
        tables = refactor_tables_json(json.load(file).get('tables', []))
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





# Bug 1: I need to preprocess the data on lines such that we take out tables and key-value pairs. 
# Bug 2: I need to find a way to format the tables correctly 


def format_extraction(table_json_path, kv_json_path, lines_json_path, output_extraction_dir):
    
    tables, kv_pairs, lines = refactor_and_persist_json_data(table_json_path, kv_json_path, lines_json_path)
    
    final_output = {} 

    combined_sets = {}
    for page in lines.keys():
        page_k_lines = lines.get(page, [])
        
        page_tables = tables.get(page, []) # Get the tables for the page = consider this v1 set. Remember the bbox is bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 
        page_kv_pairs = kv_pairs.get(page, [])  # Get the kv pairs for the page = consider this v2 set. Remember the bbox is bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 

        print(f"Page {page} has {len(page_k_lines)} lines, {len(page_tables)} tables, and {len(page_kv_pairs)} key-value pairs")
        # print(f"Page {page} has {page_k_lines}")
       
        # TODO: Merge the tables and kv pairs into a single list and sort by the y-coordinate and x-coordinate
        combined_data = page_tables + page_kv_pairs
        combined_data.sort(key=lambda item: (item['bbox'][0][1], item['bbox'][0][0])) # Sort by y-coordinate and x-coordinate = consider this v3 set. 
        

        copy_combined_data = combined_data.copy()
        
        combined_sets[page] = {
            'combined_data': copy_combined_data
        }


        

        ### Finding: Edge Case: Key-Value Pairs can also be in the same region as the table! Example is fun-1 and fun-2. ###

        if len(page_tables) == 0 and len(page_kv_pairs) == 0:
            print(f"Page {page} has no tables or key-value pairs")
            continue
        else:
            content = [] 
            restricted_regions_bbox = []
            restricted_regions_text = [] 
            for data in combined_data:
                restricted_regions_bbox.append(data['bbox'])
                restricted_regions_text.append(data['content'])
            
            page_k_lines_copy = [line for line in page_k_lines]
            while len(combined_data) != 0:
                data = combined_data.pop(0)
                restricted_region = data['bbox'] # Get the first item in the list and remove it from the list
                # Find all items that are above (y-axis) the restricted region from the page_k_lines
                k_line_content = [] 
                # print(f"Page K Lines: {page_k_lines}")

                while len(page_k_lines_copy) != 0:
                    line = page_k_lines_copy.pop(0)
                    bbox1 = line.get('bbox', {})
                    print(f"Line: {line.get('content', '')} has bbox: {bbox1}")
                    print(f"Restricted Region: {restricted_region}")
                    if not bbox1:
                        raise ValueError("No bounding box found in the line")
                    if bbox_equivalence(line, restricted_regions_bbox) or check_overlap(bbox1, restricted_region):
                        print(f"Line: {line.get('content', '')} is in the restricted region")
                        break
                    # Check if the line is above the restricted region
                    if find_lines_above_region(bbox1, restricted_region):
                        k_line_content.append(line.get('content', ""))
                    else:
                        break  # Exit the loop for lines that don't match the conditions
                # print(f"K Line Content: {page_k_lines_copy}")

                
                
                content.append({
                    "k-lines" : '\n'.join(k_line_content),
                    "type" : data['type'],
                    "content" : data['content']
                })
               
              
                
            final_output[page] = content 
    with open(os.path.join(output_extraction_dir, "combined_tables+kv.json"), 'w') as file:
            json.dump(combined_sets, file, indent=4)

    with open(os.path.join(output_extraction_dir, "final_output.json"), 'w') as file:
        json.dump(final_output, file, indent=4)
    return final_output





# Function to extract structured data from a PDF
def structured_extraction(pdf_file):
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
    tables, kv_pairs, lines = structured_extraction(pdf_file)



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
    extraction = format_extraction(table_json_path, kv_json_path, lines_json_path, output_extraction_dir)
    
   
    


    
